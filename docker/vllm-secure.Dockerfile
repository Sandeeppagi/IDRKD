ARG COSIGN_IMAGE=ghcr.io/sigstore/cosign/cosign:v3.0.6@sha256:de9c65609e6bde17e6b48de485ee788407c9502fa08b8f4459f595b21f56cd00
ARG VLLM_IMAGE=vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967

FROM ${COSIGN_IMAGE} AS cosign
FROM ${VLLM_IMAGE}

USER root
COPY --from=cosign /ko-app/cosign /usr/local/bin/cosign
COPY pyproject.toml /opt/idrkd/pyproject.toml
COPY src /opt/idrkd/src
RUN python3 -m pip install --no-cache-dir --no-deps /opt/idrkd \
    && chmod 0755 /usr/local/bin/cosign

USER 2000:0
ENTRYPOINT ["idrkd-secure-serve"]
