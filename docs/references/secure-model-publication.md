# Signed Model Publication and Verified vLLM Startup

This Phase 1 workflow cryptographically binds the promoted model checkpoint to its promotion
record. The serving process verifies the Cosign signature and every checkpoint hash before it
imports or starts vLLM.

It does not claim network-enforced zero egress. Offline library settings prevent accidental model
downloads, while host firewall or Kubernetes network policy remains a separate deployment control.

## Security boundary

The signed `model-release.json` descriptor contains:

- the model and manifest identities;
- the path, size, and SHA-256 digest of every checkpoint file;
- the promotion-record SHA-256 and canonical record digest;
- the requirement that the recorded decision is `promoted`.

The private key stays outside Git and outside the serving host. Provision only `cosign.pub` to the
serving host through a trusted, out-of-band path. Committing a replacement public key beside a
replacement signature must not be sufficient to change the production trust root.

## 1. Install Cosign

Install Cosign 3.0.6 or newer using the official Sigstore installation instructions. Cosign 3 uses
the standardized bundle format and requires `--bundle` for blob signatures. Check the binary before
using it:

```bash
cosign version
```

## 2. Generate and separate the signing key

Run this on the signing workstation, not inside the vLLM container:

```bash
install -d -m 0700 /workspace/idrkd-signing
cd /workspace/idrkd-signing
cosign generate-key-pair
chmod 0600 cosign.key
```

Protect the private-key password with `COSIGN_PASSWORD` or an interactive prompt. A KMS URI can be
passed to `--key` instead of a local private key. Copy `cosign.pub` to the serving host using a
trusted channel, for example `/workspace/idrkd-trust/cosign.pub`.

## 3. Sign the promoted release

Run from the repository containing the promoted checkpoint and schema-v2 promotion evidence:

```bash
cd /workspace/IDRKD
source .venv/bin/activate

RELEASE_DIR=eval/releases/phi4-mini-llmc-awq-v1
CHECKPOINT=artifacts/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq

idrkd-release sign \
  --checkpoint "$CHECKPOINT" \
  --promotion-record "$RELEASE_DIR/promotion-record.json" \
  --key /workspace/idrkd-signing/cosign.key \
  --descriptor "$RELEASE_DIR/model-release.json" \
  --bundle "$RELEASE_DIR/model-release.sigstore.json"
```

Commit and publish only `model-release.json` and `model-release.sigstore.json`. Never add
`cosign.key`, `COSIGN_PASSWORD`, or the vLLM API key to Git.

## 4. Verify without starting vLLM

Verification is offline and fails on an invalid signature, a rejected or modified promotion record,
an invalid manifest digest, added or removed checkpoint files, symbolic links, or changed bytes:

```bash
idrkd-release verify \
  --checkpoint "$CHECKPOINT" \
  --promotion-record "$RELEASE_DIR/promotion-record.json" \
  --public-key /workspace/idrkd-trust/cosign.pub \
  --descriptor "$RELEASE_DIR/model-release.json" \
  --bundle "$RELEASE_DIR/model-release.sigstore.json"
```

## 5. Start verified vLLM natively

Install the editable IDRKD package without changing the validated vLLM dependencies, set a strong
API key, and use the secure launcher instead of invoking `vllm serve` directly:

```bash
source /workspace/.venv-vllm/bin/activate
uv pip install --no-deps -e /workspace/IDRKD

export IDRKD_VLLM_API_KEY="$(openssl rand -hex 32)"

idrkd-secure-serve \
  --checkpoint /workspace/IDRKD/artifacts/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --promotion-record /workspace/IDRKD/eval/releases/phi4-mini-llmc-awq-v1/promotion-record.json \
  --descriptor /workspace/IDRKD/eval/releases/phi4-mini-llmc-awq-v1/model-release.json \
  --bundle /workspace/IDRKD/eval/releases/phi4-mini-llmc-awq-v1/model-release.sigstore.json \
  --public-key /workspace/idrkd-trust/cosign.pub \
  --served-model-name idrkd-phi4-mini-dpo-tooljson-split-v2-llmc-awq
```

The launcher passes the API key to vLLM through `VLLM_API_KEY`, sets Hugging Face, Transformers,
and Datasets to offline mode, and replaces itself with vLLM only after verification succeeds.

## 6. Start the hardened Compose profile

The secure image pins vLLM 0.27.1 and Cosign 3.0.6 by immutable multi-platform image digest. The
service runs as vLLM's non-root UID, drops all Linux capabilities, enables
`no-new-privileges`, uses a read-only root filesystem and model mount, and publishes port 8000 only
on localhost.

```bash
cd /workspace/IDRKD

install -d -m 0700 /workspace/idrkd-secrets
openssl rand -hex 32 > /workspace/idrkd-secrets/vllm-api-key
chmod 0600 /workspace/idrkd-secrets/vllm-api-key

export IDRKD_COSIGN_PUBLIC_KEY_PATH=/workspace/idrkd-trust/cosign.pub
export IDRKD_VLLM_API_KEY_PATH=/workspace/idrkd-secrets/vllm-api-key
export IDRKD_SLM_MODEL_ROOT=/workspace/IDRKD/artifacts/models
export IDRKD_RELEASE_EVIDENCE_ROOT=/workspace/IDRKD/eval/releases/phi4-mini-llmc-awq-v1

docker compose -f docker/docker-compose.yml --profile slm-server build slm-server
docker compose -f docker/docker-compose.yml --profile slm-server up -d slm-server
```

Authenticated readiness check:

```bash
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $(cat /workspace/idrkd-secrets/vllm-api-key)" \
  | jq .
```

If any verification step fails, the container exits before vLLM allocates the model on the GPU.

## Deferred controls

The following remain explicitly outside Phase 1:

- network-enforced zero egress;
- TLS ingress and external identity-aware proxying;
- Kubernetes or OCI publication;
- managed KMS policy, key rotation, and revocation;
- multi-node production rollout.
