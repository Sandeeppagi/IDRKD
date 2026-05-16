from __future__ import annotations

from pathlib import Path


PATH = Path("/Users/sandeeppagi/Projects/IDRKD/docs/project-plan/IDRKD_Project_Plan_v2_updated.html")


def replace_all(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    text = PATH.read_text()

    text = replace_all(
        text,
        {
            "<title>IDRKD Project Completion Plan v2</title>": "<title>IDRKD Project Completion Plan v3</title>",
            "IDRKD System · Project Completion Plan v2 · Aligned to HLD/LLD v3": "IDRKD System · Project Completion Plan v3 · Aligned to HLD/LLD v3",
            "Period: <span>10 May → 5 Jul 2026</span>": "Period: <span>16 May → 5 Jul 2026</span>",
            "10 May – 5 Jul 2026": "16 May – 5 Jul 2026",
            "10 May → 5 Jul 2026": "16 May → 5 Jul 2026",
            "10–16 May": "16–22 May",
            "17–23 May": "23–29 May",
            "24–30 May": "30 May–5 Jun",
            "31 May–6 Jun": "6–12 Jun",
            "7–13 Jun": "13–19 Jun",
            "14–20 Jun": "20–26 Jun",
            "21–27 Jun": "27 Jun–3 Jul",
            "28 Jun–5 Jul": "4–5 Jul",
            "dates:'10 May – 23 May 2026'": "dates:'16 May – 29 May 2026'",
            "dates:'24 May – 6 Jun 2026'": "dates:'30 May – 12 Jun 2026'",
            "dates:'31 May – 20 Jun 2026'": "dates:'6 Jun – 26 Jun 2026'",
            "dates:'21 Jun – 4 Jul 2026'": "dates:'27 Jun – 5 Jul 2026'",
            "dates:'21 Jun – 5 Jul 2026'": "dates:'27 Jun – 5 Jul 2026'",
            "dates:'5 Jul 2026 (and beyond)'": "dates:'4–5 Jul 2026'",
            "duration:'2 weeks', color:C[3]": "duration:'9 days', color:C[3]",
            "duration:'Parallel with SLM', color:C[4]": "duration:'Parallel with SLM wrap-up', color:C[4]",
            "duration:'Buffer + stretch', color:C[5]": "duration:'2-day closeout', color:C[5]",
            "References</div><div class=\"sc-val\" style=\"color:var(--c6)\">16+</div>": "References</div><div class=\"sc-val\" style=\"color:var(--c6)\">39+</div>",
            "Incl. Graphify baseline": "Checked links + expanded topic material",
            "All references from the dissertation grouped by theme, including Graphify as the P1 prior-art baseline added in HLD v3.": "Expanded and link-checked reference set grouped by project topic, including official specifications, implementation docs, and research papers for each phase.",
            "https://docs.anthropic.com/mcp": "https://modelcontextprotocol.io/specification/",
            "source:'Anthropic, 2024'": "source:'Model Context Protocol project · latest spec 2025-11-25'",
            "url:'https://github.com/a2aproject/A2A'": "url:'https://a2a-protocol.org/latest/specification/'",
            "source:'Google / Linux Foundation, April 2026'": "source:'A2A Protocol project · v1.0, April 2026'",
            "source:'Google / Linux Foundation — official a2a-sdk [UPDATED HLD v3]'": "source:'A2A Protocol project · official spec and SDK line [checked]'", 
            "url:'https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258'": "url:'https://a2a-protocol.org/latest/announcing-1.0/'",
            "source:'Tifrea et al., 2025'": "source:'Zha et al. · arXiv 2023'",
            "style=\"color:var(--c3)\">★ Mid-Sem Report: 16–21 Jun (W6)</div>": "style=\"color:var(--c3)\">★ Mid-Sem Report: 16–21 Jun (W5–W6)</div>",
        },
    )

    phase_ref_insertions = {
        "{num:16,title:'Software Engineering at Google: Lessons Learned from Programming Over Time',source:'Winters, Manshreck & Wright · O\\'Reilly, 2020 · p.22',url:'https://abseil.io/resources/swe-book'},": "{num:16,title:'Software Engineering at Google: Lessons Learned from Programming Over Time',source:'Winters, Manshreck & Wright · O\\'Reilly, 2020 · p.22',url:'https://abseil.io/resources/swe-book'},\n      {num:17,title:'Tree-sitter documentation: incremental parsing and language grammars',source:'Tree-sitter project · official docs [checked]',url:'https://tree-sitter.github.io/tree-sitter/'},\n      {num:18,title:'Neo4j Graph Data Science: PageRank, Louvain, BFS and path algorithms',source:'Neo4j GDS Manual v2026.04 [checked]',url:'https://neo4j.com/docs/graph-data-science/current/'},\n      {num:19,title:'Apache Kafka documentation: producers, consumers, partitioning and operations',source:'Apache Kafka official docs [checked]',url:'https://kafka.apache.org/documentation/'},\n      {num:20,title:'OpenTelemetry documentation: traces, metrics, logs and context propagation',source:'OpenTelemetry official docs [checked]',url:'https://opentelemetry.io/docs/'},\n      {num:21,title:'SpanBERT: Improving Pre-training by Representing and Predicting Spans',source:'Joshi et al. · TACL 2020',url:'https://arxiv.org/abs/1907.10529'},",
        "{num:13,title:'AlignScore: Evaluating Factual Consistency with a Unified Alignment Function',source:'Zha et al. · arXiv 2023',url:'https://arxiv.org/abs/2305.16739'},": "{num:13,title:'AlignScore: Evaluating Factual Consistency with a Unified Alignment Function',source:'Zha et al. · arXiv 2023',url:'https://arxiv.org/abs/2305.16739'},\n      {num:22,title:'BGE-M3: Multi-lingual, multi-function, multi-granularity text embeddings',source:'Chen et al. · arXiv 2024',url:'https://arxiv.org/abs/2402.03216'},\n      {num:23,title:'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',source:'Lewis et al. · NeurIPS 2020',url:'https://arxiv.org/abs/2005.11401'},\n      {num:24,title:'LangGraph documentation: stateful multi-agent workflows',source:'LangChain AI · official docs [checked]',url:'https://langchain-ai.github.io/langgraph/'},\n      {num:25,title:'pgvector: vector similarity search for PostgreSQL',source:'pgvector project · official repository [checked]',url:'https://github.com/pgvector/pgvector'},",
        "{num:12,title:'WebGPT: Browser-Assisted Question-Answering',source:'Nakano et al. · arXiv 2112',url:'https://arxiv.org/abs/2112.09332'},": "{num:12,title:'WebGPT: Browser-Assisted Question-Answering',source:'Nakano et al. · arXiv 2112',url:'https://arxiv.org/abs/2112.09332'},\n      {num:26,title:'A2A Protocol v1.0 announcement: production-ready agent-to-agent standard',source:'A2A Protocol project · April 2026 [checked]',url:'https://a2a-protocol.org/latest/announcing-1.0/'},\n      {num:27,title:'A2A and MCP: complementary protocol layers',source:'A2A Protocol documentation [checked]',url:'https://a2a-protocol.org/latest/topics/a2a-and-mcp/'},\n      {num:28,title:'MCP security and trust guidance',source:'Model Context Protocol specification [checked]',url:'https://modelcontextprotocol.io/specification/'},\n      {num:29,title:'Microsoft AutoGen documentation',source:'Microsoft · official docs [checked]',url:'https://microsoft.github.io/autogen/stable/'},",
        "{num:15,title:'MemGPT: Towards LLMs as Operating Systems',source:'Packer et al. · NeurIPS 2023',url:'https://arxiv.org/abs/2310.08560'},": "{num:15,title:'MemGPT: Towards LLMs as Operating Systems',source:'Packer et al. · NeurIPS 2023',url:'https://arxiv.org/abs/2310.08560'},\n      {num:30,title:'Phi-4-mini technical report: 3.8B SLM, 200K vocab and GQA',source:'Microsoft Research · official publication [checked]',url:'https://www.microsoft.com/en-us/research/publication/phi-4-mini-technical-report-compact-yet-powerful-multimodal-language-models-via-mixture-of-loras/'},\n      {num:31,title:'vLLM documentation: serving and OpenAI-compatible endpoints',source:'vLLM official docs [checked]',url:'https://docs.vllm.ai/'},\n      {num:32,title:'PEFT documentation: LoRA and QLoRA adapters',source:'Hugging Face PEFT docs',url:'https://huggingface.co/docs/peft/index'},\n      {num:33,title:'AWQ: Activation-aware Weight Quantization for LLM compression',source:'Lin et al. · MLSys 2024',url:'https://arxiv.org/abs/2306.00978'},",
        "{num:14,title:'DeBERTa: Decoding-Enhanced BERT with Disentangled Attention',source:'He et al. · ICLR 2021',url:'https://arxiv.org/abs/2006.03654'},\n    ]\n  },\n  {\n    id:6": "{num:14,title:'DeBERTa: Decoding-Enhanced BERT with Disentangled Attention',source:'He et al. · ICLR 2021',url:'https://arxiv.org/abs/2006.03654'},\n      {num:18,title:'Neo4j Graph Data Science: Louvain community detection and centroid drift support',source:'Neo4j GDS Manual v2026.04 [checked]',url:'https://neo4j.com/docs/graph-data-science/current/'},\n      {num:19,title:'Apache Kafka documentation: event streams and consumer lag operations',source:'Apache Kafka official docs [checked]',url:'https://kafka.apache.org/documentation/'},\n      {num:20,title:'OpenTelemetry documentation: SLO-oriented traces and metrics',source:'OpenTelemetry official docs [checked]',url:'https://opentelemetry.io/docs/'},\n      {num:34,title:'Celery documentation: distributed task queues and retries',source:'Celery project docs',url:'https://docs.celeryq.dev/en/stable/'},\n    ]\n  },\n  {\n    id:6",
        "{num:'G',title:'Graphify — prior-art baseline for P1 ingestion comparison',source:'graphify.net, April 2026',url:'https://graphify.net'},": "{num:'G',title:'Graphify — prior-art baseline for P1 ingestion comparison',source:'graphify.net, April 2026',url:'https://graphify.net'},\n      {num:35,title:'Berkeley Function Calling Leaderboard (BFCL)',source:'UC Berkeley Gorilla project',url:'https://gorilla.cs.berkeley.edu/leaderboard.html'},\n      {num:36,title:'Bootstrap confidence intervals and resampling methods',source:'Efron & Tibshirani · foundational statistical reference',url:'https://doi.org/10.1201/9780429246593'},\n      {num:37,title:'Wilcoxon signed-rank test',source:'NIST Engineering Statistics Handbook',url:'https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm'},\n      {num:38,title:'Reproducibility checklist for machine learning research',source:'Papers with Code',url:'https://paperswithcode.com/rc2021'},",
    }
    text = replace_all(text, phase_ref_insertions)

    all_refs = """const ALL_REFS=[
  {num:'G',title:'Graphify - open-source knowledge graph skill for AI coding assistants',source:'graphify.net, April 2026 [P1 prior-art baseline]',url:'https://graphify.net',theme:'Code Intelligence (P1)'},
  {num:1,title:'Google Code Wiki: Accelerating code understanding',source:'Google Developers Blog, 2025',url:'https://developers.googleblog.com/introducing-code-wiki-accelerating-your-code-understanding/',theme:'Code Intelligence (P1)'},
  {num:16,title:'Software Engineering at Google: Lessons Learned from Programming Over Time',source:"Winters, Manshreck & Wright · O'Reilly, 2020",url:'https://abseil.io/resources/swe-book',theme:'Code Intelligence (P1)'},
  {num:17,title:'Tree-sitter documentation: incremental parsing and language grammars',source:'Tree-sitter project · official docs [checked]',url:'https://tree-sitter.github.io/tree-sitter/',theme:'Code Intelligence (P1)'},
  {num:18,title:'Neo4j Graph Data Science: PageRank, Louvain, BFS and paths',source:'Neo4j GDS Manual v2026.04 [checked]',url:'https://neo4j.com/docs/graph-data-science/current/',theme:'Code Intelligence (P1)'},
  {num:19,title:'Apache Kafka documentation: event streams and operations',source:'Apache Kafka official docs [checked]',url:'https://kafka.apache.org/documentation/',theme:'Code Intelligence (P1)'},
  {num:20,title:'OpenTelemetry documentation: traces, metrics, logs and propagation',source:'OpenTelemetry official docs [checked]',url:'https://opentelemetry.io/docs/',theme:'Code Intelligence (P1)'},
  {num:21,title:'SpanBERT: Improving Pre-training by Representing and Predicting Spans',source:'Joshi et al. · TACL 2020',url:'https://arxiv.org/abs/1907.10529',theme:'Code Intelligence (P1)'},
  {num:2,title:'Model Context Protocol (MCP) Specification',source:'Model Context Protocol project · latest spec 2025-11-25 [checked]',url:'https://modelcontextprotocol.io/specification/',theme:'MCP & A2A (P4)'},
  {num:3,title:'Agent-to-Agent (A2A) Protocol v1.0 specification',source:'A2A Protocol project · official spec [checked]',url:'https://a2a-protocol.org/latest/specification/',theme:'MCP & A2A (P4)'},
  {num:26,title:'A2A Protocol v1.0 announcement',source:'A2A Protocol project · April 2026 [checked]',url:'https://a2a-protocol.org/latest/announcing-1.0/',theme:'MCP & A2A (P4)'},
  {num:27,title:'A2A and MCP: complementary protocol layers',source:'A2A Protocol documentation [checked]',url:'https://a2a-protocol.org/latest/topics/a2a-and-mcp/',theme:'MCP & A2A (P4)'},
  {num:28,title:'MCP security and trust guidance',source:'Model Context Protocol specification [checked]',url:'https://modelcontextprotocol.io/specification/',theme:'MCP & A2A (P4)'},
  {num:29,title:'Microsoft AutoGen documentation',source:'Microsoft · official docs [checked]',url:'https://microsoft.github.io/autogen/stable/',theme:'MCP & A2A (P4)'},
  {num:7,title:'ToolLLM: Facilitating LLMs to Master 16000+ Real-World APIs',source:'Qin et al. · ICLR 2024',url:'https://arxiv.org/abs/2307.16789',theme:'MCP & A2A (P4)'},
  {num:8,title:'Gorilla: Large Language Model Connected with Massive APIs',source:'Patil et al. · NeurIPS 2023',url:'https://arxiv.org/abs/2305.15334',theme:'MCP & A2A (P4)'},
  {num:11,title:'AgentBench: Evaluating LLMs as Agents',source:'Wang et al. · ICLR 2024',url:'https://arxiv.org/abs/2308.03688',theme:'MCP & A2A (P4)'},
  {num:12,title:'WebGPT: Browser-Assisted Question-Answering',source:'Nakano et al. · arXiv 2112',url:'https://arxiv.org/abs/2112.09332',theme:'MCP & A2A (P4)'},
  {num:4,title:'CRAG: Corrective Retrieval Augmented Generation',source:'Tworkowski et al. · ICLR 2025',url:'https://arxiv.org/abs/2401.15884',theme:'Agentic RAG (P3)'},
  {num:9,title:'Lost in the Middle: How Language Models Use Long Contexts',source:'Liu et al. · TACL 2024',url:'https://arxiv.org/abs/2307.03172',theme:'Agentic RAG (P3)'},
  {num:10,title:'MuSiQue: Multi-hop Questions via Single-hop Composition',source:'Trivedi et al. · TACL 2022',url:'https://arxiv.org/abs/2108.00573',theme:'Agentic RAG (P3)'},
  {num:22,title:'BGE-M3: Multi-lingual, multi-function, multi-granularity text embeddings',source:'Chen et al. · arXiv 2024',url:'https://arxiv.org/abs/2402.03216',theme:'Agentic RAG (P3)'},
  {num:23,title:'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',source:'Lewis et al. · NeurIPS 2020',url:'https://arxiv.org/abs/2005.11401',theme:'Agentic RAG (P3)'},
  {num:24,title:'LangGraph documentation: stateful multi-agent workflows',source:'LangChain AI · official docs [checked]',url:'https://langchain-ai.github.io/langgraph/',theme:'Agentic RAG (P3)'},
  {num:25,title:'pgvector: vector similarity search for PostgreSQL',source:'pgvector project · official repository [checked]',url:'https://github.com/pgvector/pgvector',theme:'Agentic RAG (P3)'},
  {num:5,title:'QLoRA: Efficient Finetuning of Quantised LLMs',source:'Dettmers et al. · NeurIPS 2023',url:'https://arxiv.org/abs/2305.14314',theme:'SLM Distillation (P5)'},
  {num:6,title:'Direct Preference Optimisation (DPO)',source:'Rafailov et al. · NeurIPS 2023',url:'https://arxiv.org/abs/2305.18290',theme:'SLM Distillation (P5)'},
  {num:15,title:'MemGPT: Towards LLMs as Operating Systems',source:'Packer et al. · NeurIPS 2023',url:'https://arxiv.org/abs/2310.08560',theme:'SLM Distillation (P5)'},
  {num:30,title:'Phi-4-mini technical report: 3.8B SLM, 200K vocab and GQA',source:'Microsoft Research · official publication [checked]',url:'https://www.microsoft.com/en-us/research/publication/phi-4-mini-technical-report-compact-yet-powerful-multimodal-language-models-via-mixture-of-loras/',theme:'SLM Distillation (P5)'},
  {num:31,title:'vLLM documentation: serving and OpenAI-compatible endpoints',source:'vLLM official docs [checked]',url:'https://docs.vllm.ai/',theme:'SLM Distillation (P5)'},
  {num:32,title:'PEFT documentation: LoRA and QLoRA adapters',source:'Hugging Face PEFT docs',url:'https://huggingface.co/docs/peft/index',theme:'SLM Distillation (P5)'},
  {num:33,title:'AWQ: Activation-aware Weight Quantization for LLM compression',source:'Lin et al. · MLSys 2024',url:'https://arxiv.org/abs/2306.00978',theme:'SLM Distillation (P5)'},
  {num:13,title:'AlignScore: Evaluating Factual Consistency with a Unified Alignment Function',source:'Zha et al. · arXiv 2023',url:'https://arxiv.org/abs/2305.16739',theme:'Evaluation & Reproducibility'},
  {num:14,title:'DeBERTa: Decoding-Enhanced BERT with Disentangled Attention',source:'He et al. · ICLR 2021',url:'https://arxiv.org/abs/2006.03654',theme:'Evaluation & Reproducibility'},
  {num:34,title:'Celery documentation: distributed task queues and retries',source:'Celery project docs',url:'https://docs.celeryq.dev/en/stable/',theme:'Evaluation & Reproducibility'},
  {num:35,title:'Berkeley Function Calling Leaderboard (BFCL)',source:'UC Berkeley Gorilla project',url:'https://gorilla.cs.berkeley.edu/leaderboard.html',theme:'Evaluation & Reproducibility'},
  {num:36,title:'Bootstrap confidence intervals and resampling methods',source:'Efron & Tibshirani · foundational statistical reference',url:'https://doi.org/10.1201/9780429246593',theme:'Evaluation & Reproducibility'},
  {num:37,title:'Wilcoxon signed-rank test',source:'NIST Engineering Statistics Handbook',url:'https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm',theme:'Evaluation & Reproducibility'},
  {num:38,title:'Reproducibility checklist for machine learning research',source:'Papers with Code',url:'https://paperswithcode.com/rc2021',theme:'Evaluation & Reproducibility'},
];"""

    start = text.index("const ALL_REFS=[")
    end = text.index("\n\nconst THEME_COLORS=", start)
    text = text[:start] + all_refs + text[end:]

    text = text.replace(
        "const THEME_COLORS={'Code Intelligence (P1)':C[0],'MCP & A2A (P4)':C[2],'Agentic RAG (P3)':C[1],'SLM Distillation (P5)':C[3],'Evaluation':C[4]};",
        "const THEME_COLORS={'Code Intelligence (P1)':C[0],'MCP & A2A (P4)':C[2],'Agentic RAG (P3)':C[1],'SLM Distillation (P5)':C[3],'Evaluation & Reproducibility':C[4]};",
    )

    text = text.replace("['W11','TaskBench + Buffer',C[5],[0,0,0,0,0,0,0,1]],", "['W8 close','TaskBench + Buffer',C[5],[0,0,0,0,0,0,0,1]],")
    text = text.replace("<div style=\"color:var(--c3);\">★ Mid-Sem Report: 16–21 Jun (W6)</div>", "<div style=\"color:var(--c3);\">★ Mid-Sem Report: 16–21 Jun (W5–W6)</div>")
    text = text.replace("HLD phases</div><div class=\"sc-val\" style=\"color:var(--c2)\">6</div><div class=\"sc-sub\">W1–W2 through W11+</div>", "HLD phases</div><div class=\"sc-val\" style=\"color:var(--c2)\">6</div><div class=\"sc-sub\">W1–W2 through W8 closeout</div>")

    PATH.write_text(text)


if __name__ == "__main__":
    main()
