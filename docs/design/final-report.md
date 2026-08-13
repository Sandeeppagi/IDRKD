---
title: "Intelligent Data Reconciliation and Knowledge Discovery (IDRKD) System"
author: "Sandeep Satish Pagi"
date: "August 2026"
source: "Converted from final-report.pdf"
---

<!-- PDF page 1 -->

**A REPORT**

**ON**

**INTELLIGENT DATA RECONCILIATION AND KNOWLEDGE**

**DISCOVERY (IDRKD) SYSTEM**

**BY**

**SANDEEP SATISH PAGI**

ID No.: 2024AA05545

**AT**

Telstra Corporation Limited, Melbourne, Australia

<!-- Embedded image from source PDF page 1. -->

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI**

**(RAJASTHAN)**

(August 2026)

---

<!-- PDF page 2 -->

**A REPORT**

**ON**

**INTELLIGENT DATA RECONCILIATION AND KNOWLEDGE**

**DISCOVERY (IDRKD) SYSTEM**

**BY**

**SANDEEP SATISH PAGI**

ID No.: 2024AA05545

Discipline: M.Tech. Artificial Intelligence and Machine Learning

Prepared in partial fulfilment of the

**WILP Dissertation Course No. BITS ZG628T**

**AT**

Telstra Corporation Limited, Melbourne, Australia

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI**

(August 2026)

---

<!-- PDF page 3 -->

# Acknowledgements

I gratefully acknowledge the leadership of Telstra Corporation Limited in Melbourne for fostering a setting in which advanced enterprise artificial intelligence research could be undertaken alongside my professional duties. The realities of large-scale telecommunications software supplied the practical foundation and motivation for the research problem examined in this dissertation.

I am indebted to my Faculty Mentor, Dr Dipankar Dutta, BITS Pilani, whose observations substantially strengthened the scholarly standard of the dissertation. Throughout the report, I have applied his guidance to establish the research gap through a rigorous literature review, state the innovative elements explicitly, expand abbreviations at first occurrence, and follow a recognised referencing convention.

My sincere appreciation is extended to my Dissertation Supervisor, Pavan Chakala and the organisation's Additional Examiner, Vineeth Pallithara Chengalaveetil, for their sustained direction, detailed assessment of successive drafts, and participation in the final discussion of this study. Their consistent focus on measurable evidence and disciplined engineering strongly influenced the project's evaluation-led orientation.

I further acknowledge the Work Integrated Learning Programmes (WILP) Division of the Birla Institute of Technology and Science, Pilani, for designing a dissertation framework that enables employed professionals to undertake substantive research. I am equally grateful to my family and colleagues for their understanding and encouragement throughout the numerous evenings and weekends committed to this work.

Sandeep Satish Pagi

August 2026, Melbourne

---

<!-- PDF page 4 -->

# Abstract Sheet

BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI (RAJASTHAN)

WILP Division

| Field | Details |
| --- | --- |
| Organisation | Telstra Corporation Limited |
| Location | Melbourne, Australia |
| Duration | One academic semester (Second Semester 2025–2026) |
| Date of Start | February 2026 |
| Date of Submission | August 2026 |
| Title of the Project | Intelligent Data Reconciliation and Knowledge Discovery (IDRKD) System |
| ID No. / Name of the Student | 2024AA05545 / Sandeep Satish Pagi |
| Name and Designation of the Supervisor | Pavan Chakala [Staff Software Engineer, Telstra] |
| Name and Designation of the Additional Examiner | Vineeth Pallithara Chengalaveetil [Staff Software Engineer, Telstra] |
| Name of the Faculty Mentor | Dr. Dipankar Dutta |
| Course Number | BITS ZG628T (Dissertation) |
| Key Words | Knowledge Graph; Retrieval-Augmented Generation; Model Context Protocol; Agent-to-Agent Protocol; Small Language Model; Knowledge Distillation; Drift Detection; Multi-Agent Systems |
| Project Areas | Artificial Intelligence; Machine Learning; Software Engineering; Enterprise Knowledge Management |

---

<!-- PDF page 5 -->

# Abstract

Knowledge required to understand enterprise software is distributed among source-code repositories, Application Programming Interface definitions, database schemas, and operational documentation. This dissertation introduces Intelligent Data Reconciliation and Knowledge Discovery (IDRKD), a six-pillar reference architecture and research prototype combining Tree-sitter ingestion, a Neo4j Knowledge Graph, pgvector retrieval with Reciprocal Rank Fusion, typed Model Context Protocol-style tools, Agent-to-Agent SDK contracts, Phi-4-mini QLoRA/DPO training, AWQ publication, and drift-aware re-indexing. The promoted W4A16 model achieved tool F1 and exact argument accuracy of 1.00 on an internal 89-case prompt-group holdout. On an NVIDIA L40S through vLLM, 20 streaming samples produced p95 time to first token of 0.0332 seconds and p95 completion latency of 1.3201 seconds. Five live repository queries achieved minimum transformer-NLI faithfulness of 0.8744 and mean 0.9577. These measurements support internal structured tool-call competence, grounded-answer feasibility, and local quantised serving. Comparative teacher/baseline experiments, official BFCL, human agreement, peak VRAM, LangGraph–AutoGen decomposition, and component ablations were not completed and are reported as future evaluation rather than empirical findings.

| Column 1 | Column 2 |
| --- | --- |
| Signature of the Student | Signature of the Supervisor. |
| Date: 01/08/2026 | Date: 01/08/2026 |
| Place: Melbourne | Place: Sydney |

---

<!-- PDF page 6 -->

# Table of Contents

Acknowledgements.......................................................................................................................................................... i Abstract Sheet................................................................................................................................................................. ii Abstract.......................................................................................................................................................................... iii Table of Contents.......................................................................................................................................................... iv List of Figures................................................................................................................................................................ vi List of Tables................................................................................................................................................................ vii 1. Introduction................................................................................................................................................................ 1 2. Literature Review....................................................................................................................................................... 4 3. Research Methodology............................................................................................................................................... 7

3.1 Research Approach................................................................................................................................................ 7 3.2 Methodology Flowchart......................................................................................................................................... 7 3.3 Research Questions and Success Thresholds.......................................................................................................... 8 3.4 Evaluation Design.................................................................................................................................................. 9

3.4.1 MCP-TaskBench Evaluation (C1)................................................................................................................................. 9 3.4.2 Distillation Evaluation (C2)......................................................................................................................................... 10 3.4.3 Multi-Agent versus Monolithic Evaluation (C3)......................................................................................................... 10 3.5 Statistical Procedures........................................................................................................................................... 11 3.6 Data Collection.................................................................................................................................................... 11 3.7 Assumptions and Constraints............................................................................................................................... 12 4. System Design and Architecture............................................................................................................................. 13

4.1 Architectural Overview........................................................................................................................................ 13 4.2 High-Level Architecture...................................................................................................................................... 14 4.3 Pillar Design Details............................................................................................................................................ 14

4.3.1 P1 — Structural Ingestion............................................................................................................................................ 14 4.3.2 P2 — Knowledge Graph.............................................................................................................................................. 16 4.3.3 P3 — Hybrid Retrieval and Agentic RAG.................................................................................................................. 17 4.3.4 P4 — MCP and A2A Orchestration............................................................................................................................ 20 4.3.5 P5 — SLM Distillation................................................................................................................................................ 21 4.3.6 P6 — Drift Detection and Re-indexing....................................................................................................................... 24 4.4 Query-Time Control Flow.................................................................................................................................... 26 4.5 Security and Governance Design.......................................................................................................................... 26 4.6 Knowledge Graph Data Model............................................................................................................................. 26 4.7 Tools and Platform.............................................................................................................................................. 27 4.8 System Context, Container and Deployment Views.............................................................................................. 28

4.8.1 System Context and External Boundaries................................................................................................................... 28 4.8.2 Container View and Service Responsibilities.............................................................................................................. 29 4.8.3 Deployment Topology and Training Isolation............................................................................................................. 30 4.9 Architecture Decisions and Quality Attributes...................................................................................................... 31

4.9.1 Architecture Decision Record...................................................................................................................................... 31 4.9.2 Quality Attributes and Service-Level Objectives........................................................................................................ 31 4.10 Canonical Data and Protocol Contracts............................................................................................................... 32

4.10.1 Store Ownership and Canonical Identifiers............................................................................................................... 32 4.10.2 MCP Message and Tool Contract.............................................................................................................................. 32 5. Implementation........................................................................................................................................................ 34

5.1 Implementation Phases........................................................................................................................................ 34 5.2 P1 — Structural Ingestion with Deterministic Fingerprints................................................................................... 34 5.3 P2 — Knowledge Graph Construction and Governed Query Access..................................................................... 35 5.3 P3 — Hybrid Retrieval with Reciprocal Rank Fusion and Bounded Critic Loop................................................... 36 5.4 P4 — MCP Tool Gateway and MCP-TaskBench Scoring..................................................................................... 36 5.5 P5 — Verified Trace Distillation into Phi-4-mini................................................................................................. 37 5.6 P4/P6 — A2A Delegation and Drift-Aware Selective Re-indexing....................................................................... 37

---

<!-- PDF page 7 -->

5.7 Deployment and Operational Considerations........................................................................................................ 38 5.8 Detailed Runtime and Resilience Flows............................................................................................................... 38

5.8.1 Event-Driven Ingestion and Logical Transaction........................................................................................................ 38 5.8.2 Query State Machine and Bounded Verification......................................................................................................... 39 5.8.3 A2A Delegation and Cross-Agent Governance........................................................................................................... 40 5.8.4 Error Handling, Retry, and Back-Pressure Rules........................................................................................................ 41 5.9 Security, Observability, and Delivery Controls..................................................................................................... 41

5.9.1 Threat Model and Prompt-Injection Containment....................................................................................................... 41 5.9.2 Observability and SLO-Based Alerting....................................................................................................................... 42 5.9.3 Evaluation Harness, CI/CD, and Reproducibility........................................................................................................ 42 5.9.4 Capacity Planning and Migration Triggers.................................................................................................................. 43 5.10 SLM Serving, Quantisation, and Publication Gates............................................................................................. 43

5.10.1 Detailed Distillation and Serving Flow..................................................................................................................... 43 5.11 Drift Operations and Re-index Scheduling......................................................................................................... 45

5.11.1 Event Topics and Trigger Conditions........................................................................................................................ 45 5.11.2 Fairness and Operational Limits................................................................................................................................ 45 6. Results and Discussion............................................................................................................................................. 46

6.1 Experimental Setup.............................................................................................................................................. 46 6.2 RQ1 — MCP-TaskBench Discriminative Power (C1).......................................................................................... 46 6.3 RQ2 — Verified Distillation into Phi-4-mini (C2)................................................................................................ 46 6.4 RQ3 — Decomposed Multi-Agent versus Monolithic Baseline (C3)..................................................................... 47 6.5 Ablation Studies.................................................................................................................................................. 47 6.6 Qualitative Observations...................................................................................................................................... 48 6.7 Discussion of Discrepancies and Threats to Validity............................................................................................. 48 7. Conclusions and Recommendations....................................................................................................................... 49

7.1 Conclusions......................................................................................................................................................... 49 7.2 Recommendations............................................................................................................................................... 49 7.3 Limitations and Future Scope of Work................................................................................................................. 49 Appendix A: Supplementary Pseudocode.................................................................................................................. 50 Appendix B: MCP-TaskBench Task Category Examples........................................................................................ 51 Appendix C: Reproducibility Checklist..................................................................................................................... 52 Appendix D: HLD/LLD Technical Coverage Matrix............................................................................................... 53 Appendix E: Design Assumptions and Claim Boundaries....................................................................................... 54 Appendix F: Pillar information.................................................................................................................................. 55 References..................................................................................................................................................................... 62 Glossary......................................................................................................................................................................... 65

---

<!-- PDF page 8 -->

# List of Figures

| Figure | Title | Section |
| --- | --- | --- |
| Figure 3.1 | IDRKD Research Methodology Flowchart | 3.2 |
| Figure 3.2 | MCP-TaskBench — A Multi-Dimensional Evaluation Framework for AI Agent Tool Use | 3.4.1 |
| Figure 3.3 | Verified Tool Distillation — A Robust Training Pipeline for Small Language Models | 3.4.2 |
| Figure 3.4 | Multi-Agent Systems for Knowledge Graph Reconciliation — A Comparative Research Framework | 3.4.3 |
| Figure 4.1 | IDRKD Six-Pillar Architecture Overview | 4.1 |
| Figure 4.2 | IDRKD End-to-End Architecture Flow | 4.2 |
| Figure 4.3 | Ingestion, Parsing, and Transactional Persistence Workflow | 4.3.1 |
| Figure 4.4 | Knowledge Graph Schema, Conflict Resolution, Analytics, and Traversal Patterns | 4.3.2 |
| Figure 4.5 | Agentic Query Processing and Grounded Response Workflow | 4.3.3 |
| Figure 4.6 | MCP Gateway and A2A Bridge Architecture | 4.3.4 |
| Figure 4.7 | Reasoning-Trace SLM Distillation and Deployment Pipeline | 4.3.5 |
| Figure 4.8 | Dual-Layer Drift Detection and Selective Re-indexing Pipeline | 4.3.6 |
| Figure 4.9 | Bounded Agentic Retrieval-Augmented Generation (RAG) and Critic Loop | 4.4 |
| Figure 4.10 | IDRKD System Context and External Trust Boundaries | 4.8.1 |
| Figure 4.11 | IDRKD Container View and Service Responsibilities | 4.8.2 |
| Figure 4.12 | Deployment Topology and Training-Inference Isolation | 4.8.3 |
| Figure 5.1 | Event-Driven Ingestion and Logical Transaction Flow | 5.8.1 |
| Figure 5.2 | LangGraph State Machine for Query Resolution | 5.8.2 |
| Figure 5.3 | A2A Delegation from LangGraph to the AutoGen Reconciliation Agent | 5.8.3 |
| Figure 5.4 | Distillation, Quantisation, and Gated Model Publication Pipeline | 5.10.1 |
| Figure F.1 | IDRKD High-Level System Architecture (Six-Pillar Design) | Appendix F |
| Figure F.2 | Pillar 1 — Structural Ingestion | Appendix F |
| Figure F.3 | Pillar 2 — Knowledge Graph | Appendix F |
| Figure F.4 | Pillar 3 — Agentic RAG Pipeline | Appendix F |
| Figure F.5 | Pillar 4 — MCP and A2A Orchestration | Appendix F |
| Figure F.6 | Pillar 5 — SLM Distillation and Serving | Appendix F |
| Figure F.7 | Pillar 6 — Drift Detection and Re-indexing | Appendix F |

---

<!-- PDF page 9 -->

# List of Tables

| Table | Title | Section |
| --- | --- | --- |
| Table 2.1 | Literature Gaps and IDRKD Design Responses | 2 |
| Table 3.1 | Research Questions and Success Thresholds | 3.3 |
| Table 3.2 | Data Sources Used in the Evaluation Corpus | 3.6 |
| Table 4.1 | IDRKD System Modules and Responsibilities | 4.1 |
| Table 4.2 | Principal Node and Edge Types of the IDRKD Graph | 4.6 |
| Table 4.3 | Technical Specifications of the Prototype Platform | 4.7 |
| Table 4.4 | Key Architecture Decisions and Consequences | 4.9.1 |
| Table 4.5 | Quality Attributes and MVP Service-Level Objectives | 4.9.2 |
| Table 4.6 | Canonical Stores, Ownership, and Identity Rules | 4.10.1 |
| Table 4.7 | MCP Request, Response, and Audit Contract Summary | 4.10.2 |
| Table 5.1 | Implementation Phases and Delivery Status | 5.1 |
| Table 5.2 | MCP-TaskBench Scoring Dimensions | 5.4 |
| Table 5.3 | Runtime Failure Handling, Retry, and Back-Pressure Rules | 5.8.4 |
| Table 5.4 | STRIDE-Aligned Threats and Security Controls | 5.9.1 |
| Table 5.5 | Core Observability Metrics and Alerting Objectives | 5.9.2 |
| Table 5.6 | Environments and Continuous-Delivery Gates | 5.9.3 |
| Table 5.7 | SLM Evaluation and Publication Gates | 5.10.1 |
| Table 5.8 | Drift and Re-index Event Topics | 5.11.1 |
| Table 6.1 | MCP-TaskBench Agent Comparison Results | 6.2 |
| Table 6.2 | Small Language Model (SLM) Distillation Results | 6.3 |
| Table 6.3 | Multi-Agent versus Monolithic Baseline Results | 6.4 |
| Table 6.4 | Ablation Study Summary | 6.5 |
| Table B.1 | MCP-TaskBench Task Category Examples | Appendix B |
| Table D.1 | HLD/LLD-to-Dissertation Technical Coverage | Appendix D |
| Table E.1 | Principal Research and Delivery Risks | Appendix E |
| Table G.1 | Glossary of Terms and Abbreviations | Glossary |

---

<!-- PDF page 10 -->

# 1. INTRODUCTION

This research is situated at the intersection of artificial-intelligence-enabled software engineering and enterprise knowledge discovery. It investigates whether structural program analysis, graph-based knowledge modelling, Retrieval-Augmented Generation (RAG), protocol-governed tool interaction, and Small Language Model (SLM) deployment can be combined into a single, testable architecture for data reconciliation and repository-level question answering.

The Intelligent Data Reconciliation and Knowledge Discovery (IDRKD) system targets settings in which engineers must reason over interdependent source code, Application Programming Interface (API) agreements, database structures, documentation, and operational information. Rather than offering a production-ready enterprise product, the study builds a research prototype whose behaviour can be evaluated through explicit technical criteria, aligned with the dissertation’s academic aims.

In large enterprise software estates, critical knowledge is commonly distributed rather than managed centrally. Answering a single engineering query may require evidence drawn from repositories, database definitions, API contracts, incident records, runbooks, and informal architecture notes. For example, engineers might need to determine which services are exposed to a schema change, follow the modules that depend on a particular API, or clarify conflicting depictions of the same business entity. Manual investigation is slow and often leads to inconsistent conclusions. Conventional document search also fails to capture structural relationships and multi-stage dependencies effectively, and these weaknesses intensify as the software ecosystem grows and becomes more mature.

Large Language Models (LLMs) built on the Transformer architecture [17], [18] made it practically feasible to query organisational knowledge in natural language. Retrieval-Augmented Generation (RAG) [19] allowed outputs to be grounded in retrieved evidence rather than depending solely on parametric memory, while dense retrieval approaches [20], [21] made semantic access to unstructured material broadly achievable. However, most operational RAG implementations still retrieve separate text segments. This is inadequate for repository analysis and data reconciliation, since the relationships among entities can be as important as their textual descriptions. Therefore, an effective solution must bring together structural ingestion, graph traversal, vector retrieval, inconsistency reasoning, reliable tool calls, and maintenance procedures that refresh indexed knowledge whenever the underlying source systems change.

The architecture was further shaped by three developments. First, the introduction of the Model Context Protocol (MCP) [3], an open standard that connects language-model agents with tools via JavaScript Object Notation Remote Procedure Call (JSON-RPC) 2.0, necessitates evaluating both the chosen action and the correctness of its protocol-level representation. Second, the availability of compact open models such as Phi-4-mini [8] makes it possible for organisations that cannot share source code externally to perform private on-premises inference, assuming that tool-use competence can be transferred effectively. Third, standardisation through the Agent-to-Agent Protocol (A2A) [16] enables communication across agent frameworks and supports interoperable decomposition, rather than requiring a single monolithic implementation.

The core question addressed by this dissertation is whether an integrated architecture that combines Knowledge Graph grounding, hybrid retrieval, protocol-controlled tool access, validated small-model distillation, and drift-aware index maintenance can measurably improve evidence-grounded reconciliation and multi-hop question answering over enterprise-style software repositories, compared with flat-retrieval and single-agent baselines.

---

<!-- PDF page 11 -->

No assertion is made that the constituent technologies are individually novel. Instead, the contribution stems from their controlled integration and from the measurement instruments created to evaluate the resulting system. Chapter 3 breaks the problem into three falsifiable research questions with predefined quantitative thresholds, so that positive, negative, and mixed results can be reported with the same level of transparency.

The dissertation pursues the following objectives.

- Develop and implement a reproducibly evaluable six-pillar reference architecture that includes structural ingestion, Knowledge Graph creation, hybrid Retrieval-Augmented Generation, Model Context Protocol and Agent-to-Agent Protocol orchestration, Small Language Model distillation, and drift-aware re-indexing. • Create MCP-TaskBench to assess semantic tool-selection accuracy, JSON-RPC 2.0 protocol compliance, and structured error behaviour across MCP-enabled agents.

- Transfer MCP tool-calling capability from a frontier teacher to Phi-4-mini using Quantised Low-Rank Adaptation (QLoRA) [6] and Direct Preference Optimisation (DPO) [7], allowing the teacher trace to be used in training only after staging-based re-execution and validation. • Assess a decomposed multi-agent workflow implemented with LangGraph and AutoGen [33] and connected via A2A against a monolithic single-agent baseline on paired multi-hop Knowledge Graph queries.

- Carry out selective re-indexing based on two drift layers—entity-level cosine drift and community-level centroid shift—so that repository changes do not normally necessitate complete graph reconstruction.

To ensure feasibility within a single academic semester, the investigation adopts a deliberately limited scope. Evaluation is performed using public Python and JavaScript repositories together with synthetic schemas designed to reflect enterprise systems, while proprietary organisational repositories are excluded. This strategy preserves the structural characteristics of enterprise knowledge discovery without disclosing confidential code or internal information.

Several limitations are stated explicitly. The Minimum Viable Product (MVP) supports two programming languages, with languages such as Java postponed to later work. Although the evaluation corpus is structurally realistic, it is smaller than a full enterprise software estate. Because MCP and A2A continue to evolve, the exact protocol versions used are recorded. The Small Language Model (SLM) investigation is further constrained by the compute available for fine-tuning, and the Chapter 3 thresholds correspond to this limitation. In line with submission requirements, executable source code is not included; pseudocode is provided wherever methodological precision is required.

Five complementary streams provide the study data, as described in Section 3.5: ingestion-ready snapshots of open-source repositories; synthetic database schemas with enterprise-like properties; MCP-TaskBench cases produced manually and from templates; teacher-model reasoning traces generated and verified in staging; and measurements collected by automated evaluation harnesses under fixed random seeds. Deterministic fingerprints are assigned to every artefact to enable complete experimental replay.

The study delivers practical, academic, and professional benefits. From an engineering standpoint, organisations are increasingly integrating language-model agents with internal tools, yet trustworthy deployment hinges on protocol validity, evidential grounding, and index freshness—features that are not consistently captured in current evaluations. MCP-TaskBench, staged verification of traces, and drift-sensitive re-indexing can be applied directly

---

<!-- PDF page 12 -->

in such settings. From a research standpoint, the controlled experiments investigate whether semantic tool capability can depart from protocol compliance and whether decomposed multi-agent topologies enhance Knowledge Graph reasoning under matched conditions.

Four tightly delimited shortcomings motivate the project. Function-calling and MCP benchmarks do not provide the specific four-component protocol analysis that MCP-TaskBench uses for Knowledge Graph reconciliation [1], [2], [56], [57]. While trajectory checking is included in existing tool-use data preparation, the influence of replaying every MCP trace through the project’s staging gateway prior to QLoRA and DPO has not been studied adequately [58]. Existing research on dynamic GraphRAG addresses incremental updates and stale traversal, but it does not evaluate the two-level drift metric adopted here [13], [14]. Lastly, although interoperability literature assigns MCP and A2A responsibilities, rigorous controlled comparisons of protocol-linked agent topologies for repository-scale KG reasoning remain limited [15], [16]. IDRKD links each limitation to a specific design component and tests it experimentally.

The report is structured into seven chapters. Chapter 2 synthesises prior research and identifies the motivating gaps. Chapter 3 sets out the methodology, the falsifiable questions, the evaluation setup, the statistical methods, and the collection procedures. Chapter 4 details the six-pillar design and architecture, and Chapter 5 describes the implementation, including pseudocode for algorithmic specificity. Chapter 6 reports the results, ablations, and the discrepancies observed, followed by the conclusions and recommendations in Chapter 7. Supplementary pseudocode, benchmark examples, and the reproducibility checklist are provided in the appendices prior to the references and glossary. Appendices D and E further document alignment with the HLD/LLD, the design assumptions, the limits of the claims, and the main risks.

---

<!-- PDF page 13 -->

# 2. LITERATURE REVIEW

The literature review is structured around seven streams of scholarship: foundations of large language models; retrieval-augmented generation and dense search; graph-oriented retrieval and repository-level question answering; tool use and function calling; model context protocol conformance; parameter-efficient adaptation and small-model distillation; and interoperable multi-agent systems. Within each theme, the discussion separates established findings from open questions, and then ties the resulting gap to an IDRKD design decision. This work does not aim to claim novelty for every technology adopted; instead, it explains why their specific integration and evaluation are justified by prior research.

Self-attention replaced the recurrent processing in the Transformer architecture proposed by Vaswani et al. [17], providing the architectural basis for every language model used in this study. Bidirectional Encoder Representations from Transformers (BERT) [42] later showed that broad transfer across natural-language tasks can be achieved by combining large-scale self-supervised pre-training with downstream adaptation. Brown et al. [18] demonstrated that scaled decoder-only models deliver strong few-shot performance without task-specific optimisation, and widely released foundations such as LLaMA [44] made capable language models available for local experimentation. Taken together, the literature supports language-model interfaces over technical collections; however, it does not address how models should be grounded in structured enterprise knowledge that evolves rapidly, which is the focus of the present work.

Retrieval-Augmented Generation (RAG) [19] couples a parametric generator with non-parametric retrieved memory, enabling generated responses to cite external evidence. Dense Passage Retrieval [20] showed that learned dense representations outperform sparse lexical methods for open-domain question answering, while Sentence- BERT [21] provided practical sentence embeddings that are widely used in operational retrieval systems, including the pgvector store in IDRKD. Reciprocal Rank Fusion (RRF) [22] provides a robust, low-parameter way to merge dissimilar rankings and has repeatedly surpassed more complex learning-to-rank approaches; accordingly, IDRKD applies it to combine graph and vector candidates. The vector index depends on efficient approximate nearest-neighbour search using Hierarchical Navigable Small World (HNSW) graphs [23]. Faithfulness metrics described in Chapter 6 are based on automated RAG evaluation frameworks such as RAGAS [43].

Research gap. These approaches mainly treat collections as text and retrieve flat units. They do not capture relations among program entities, nor do they resolve inconsistent definitions that arise from different sources. IDRKD instead treats the knowledge graph as an equal retrieval channel alongside the vector store, and it integrates both rankings using RRF, rather than using graph information only as supplementary context.

Graph-based Retrieval-Augmented Generation (GraphRAG) [24] showed that community analysis over an extracted entity graph can support query-directed summarisation beyond what conventional chunk retrieval provides. The Louvain algorithm [26] is commonly used to identify communities, while PageRank [27] estimates node importance; both methods feed into the analytic layer of the IDRKD graph. HotpotQA [25] and related multi-hop benchmarks also set an evaluation format in which answers rely on evidence chains spanning multiple documents, a format adopted for the IDRKD multi-hop knowledge graph (KG) tests.

In software engineering, StackRepoQA [12] found that repository question answering improves when structural graph evidence is included, even though its limited overall accuracy suggests substantial room for further gains. CodeBERT [37] and GraphCodeBERT [38], with the latter introducing explicit data-flow structure, also provide evidence that source code is represented more effectively when its structure is preserved rather than reduced to text alone.

---

<!-- PDF page 14 -->

Regarding changes to graph indexes, EraRAG [13] highlighted the growing cost of complete graph reconstruction as the corpus expands. CatRAG [14] introduced the Static Graph Fallacy, in which traversing an outdated graph causes semantic divergence and fails to provide missing support. Neither line of work defines an operational scoring rule for deciding when and where to re-index.

Research gap. Graph structure is beneficial for repository-scale question answering, yet existing dynamic GraphRAG studies do not offer an implementable drift-detection mechanism. IDRKD addresses this gap with a two-level metric that combines entity cosine drift with changes in the community centroid, and it selectively re-indexes only the impacted subgraphs, as described in Chapter 5. This directly addresses the limitations identified in [13] and [14].

A related research direction examines how language models choose and use external tools. ReAct [28] demonstrated the well-known agent cycle by alternating reasoning with action. Toolformer [29] showed self-supervised learning of API usage, while Gorilla [30] achieved dependable calls across very large API collections. Chain-of-Thought prompting [31] and Reflexion [32] improved multi-stage inference and iterative correction, respectively, and both inform IDRKD’s bounded critic-based verification. Capability assessment has progressed through the Berkeley Function Calling Leaderboard (BFCL) [1], which evaluates semantic function selection across varied scenarios, and ToolLLM/ToolBench [2], whose coverage exceeds sixteen thousand real-world Application Programming Interfaces (APIs).

Research gap. While BFCL and ToolBench are strong references for semantic function calling [1], [2], neither is natively organized around MCP protocol behavior. Later MCP evaluations cover live servers, multi-step execution, planning, parameter construction, syntax, and recovery from failure [56], [57]. Accordingly, MCP- TaskBench does not claim to be the earliest general-purpose MCP benchmark. Its specific contribution is a separate evaluation of semantic tool selection, JSON-RPC 2.0 validity, adherence to the advertised schemas, and structured error recovery for Knowledge Graph reconciliation tasks.

The Model Context Protocol (MCP) standard [3] specifies JSON-RPC 2.0 message formats, capability negotiation, and error behavior for connecting agents with tools and information sources. Srinivasan [4] studied operational deployment concerns such as propagated identity, dynamic tool budgets, and structured failures, and the MCP Security Bench [5] revealed protocol-specific attack vectors. MCP-Bench [56] evaluates multi-stage tasks on active MCP servers, and MCP-Atlas [57] extends real-server evaluation with diagnostics for parameterisation, syntax, error recovery, and efficiency. With respect to this work, MCP-TaskBench advances an intentionally restricted novelty claim: a stable, protocol-aware score decomposition for KG-based reconciliation, rather than precedence over MCP benchmarking in general.

Knowledge distillation [34] presented the teacher–student setup in which a smaller model learns to approximate the behaviour of a more capable teacher, and DistilBERT [36] showed that much of the original capability can be preserved at substantially lower cost. Low-Rank Adaptation (LoRA) [35] made fine-tuning large models feasible by limiting learning to low-rank parameter updates. QLoRA [6] combined this with quantised base weights, allowing adaptation on a single Graphics Processing Unit (GPU). Direct Preference Optimisation (DPO) [7] carries out preference alignment without training a separate reward model. Phi-4-mini [8], with 3.8 billion parameters, thus serves as a suitable basis for private local execution.

In the more specific setting of agent distillation, Kang et al. [9] argued that a student should reproduce the entire agent trajectory, including retrieval and tool interaction, rather than only learning the final answer. Zhong et al. [10] pointed to cascading call failures as the main weakness of small-model agents, since any incorrect early action

---

<!-- PDF page 15 -->

propagates and contaminates all subsequent steps. StepGap [11] also demonstrated the need to check intermediate reasoning stages rather than judging only the final outputs.

Research gap. Validation of tool-use training data has already been explored: Tool-MVR verifies APIs, prompts, and reasoning trajectories during dataset construction [58]. Accordingly, IDRKD does not claim to be the first general mechanism for trajectory verification. Its narrower contribution is an MCP execution gate that replays each teacher trace through the project staging gateway and includes it in the QLoRA and DPO corpora only when tool requests, executions, intermediate states, and final outcomes all meet validation criteria. This directly addresses the cascading-error problem reported in [10] and realises the intermediate-step focus of [11].

AutoGen [33] found that communication among specialised agents can outperform a single-agent approach on complex tasks, whereas LangGraph provides deterministic state-machine control suitable for workflows that must be auditable. The interoperability review by Ehtesham et al. [15] surveyed MCP, the Agent Communication Protocol, A2A, and the Agent Network Protocol, concluding that heterogeneous agents need standard ways to access tools and coordinate with one another. Under the Linux Foundation’s oversight, A2A [16] is framed as complementary to MCP: the former connects agents across frameworks and organisations, while the latter links an agent to tools and data.

Research gap. Existing surveys describe the relevant protocols but provide little controlled evidence that protocol-connected, decomposed agent systems outperform monolithic alternatives for Knowledge Graph reasoning. IDRKD’s third contribution tackles this by using paired tests that keep models, datasets, and tool catalogues unchanged.

IDRKD’s event-driven ingestion design follows Apache Kafka’s log-oriented architecture [39]. The statistical approach relies on Cohen’s κ for inter-annotator consistency [40] and Cohen’s d, evaluated using established effect-size thresholds [41]. Together, these sources form the methodological foundation for the measurements specified in Chapter 3.

From the review, four constrained research gaps are identified, and each is matched with an IDRKD response. Existing MCP benchmarks evaluate realistic tool interactions and selected diagnostic characteristics, but they do not adopt the same distinct four-part scoring framework for KG reconciliation [56], [57]. Verified preparation of tool trajectories is established [58], but systematic MCP staging replay prior to QLoRA and DPO continues to be insufficiently assessed. Work on Dynamic GraphRAG enables incremental upkeep while not providing the dual-layer drift score evaluated in this dissertation [13], [14]. Finally, interoperability standards specify agent communication; however, controlled comparisons between decomposed and monolithic architectures for repository-scale KG reasoning remain rare [15], [16]. Chapter 3 converts these findings into testable research questions.

**Table 2.1: Literature Gaps and IDRKD Design Responses**

| Area | Established by literature | Open gap | IDRKD response |
| --- | --- | --- | --- |
| Tool-use evaluation | Semantic function calling and MCP-native multi-step tool use are benchmarked [1], [2], [56], [57]. | The four-part semantic, JSON-RPC, schema, and structured-error breakdown is not reported for KG reconciliation. | MCP-TaskBench separate-dimension scoring (C1). |
| Agentic distillation | Small models can learn agent behaviour [9]; PEFT is tractable [6], [35]; verified trajectory curation exists [58]. | The effect of replaying every MCP teacher trace before QLoRA + DPO remains under-evaluated. | MCP staging replay and admission gate (C2). |
| Dynamic GraphRAG | Graph structure improves repo-scale QA [12], [24] | Static Graph Fallacy named [13], [14]; no operational drift scoring | Dual-layer drift + selective re-indexing (P6) |
| Agent interoperability | MCP and A2A are complementary standards [15], [16] | No controlled decomposed-vs-monolithic evidence on KG reasoning | Paired evaluation over A2A (C3) |

---

<!-- PDF page 16 -->

# 3. RESEARCH METHODOLOGY

## 3.1 Research Approach

The project is positioned as an academic investigation rather than solely as an engineering build. Accordingly, its novelty is expressed through testable contributions rather than broad claims. Each contribution is linked to a research question and a quantitative acceptance criterion, so the dissertation can present favourable, unfavourable, or mixed evidence in a clear manner. For instance, if the decomposed multi-agent system fails to outperform the monolithic baseline, this would still provide controlled evidence about the conditions under which decomposition helps—or does not help—repository-scale Knowledge Graph (KG) reasoning.

The research approach integrates design science, via construction of the artefact, with proposed controlled experiments against fixed baselines. Versioned snapshots, predetermined seeds, and deterministic fingerprints support reproducibility. Only experiments linked to committed machine-readable evidence are reported as completed; unexecuted baseline and ablation designs remain part of the methodology.

## 3.2 Methodology Flowchart

Figure 3.1 presents the full methodology as a flow process. It follows the work from problem formulation and literature synthesis through the definition of the research questions, architecture design, prototype development, data collection, benchmark runs, and evaluation against the criteria, then continues through ablations and final interpretation. A parallel branch represents the Small Language Model (SLM) distillation process running alongside the main implementation stream. When a preregistered threshold is not met, a feedback path from criteria assessment feeds back into design and implementation, making the method iterative and corrective instead of strictly sequential. The diagram is produced from a code-defined layout using a diagramming tool, and the layout specification is preserved in the reproducibility package.

---

<!-- PDF page 17 -->

<!-- Embedded image from source PDF page 17. -->

**Figure 3.1: IDRKD Research Methodology Flowchart**

## 3.3 Research Questions and Success Thresholds

Table 3.1 specifies the three research questions (RQ1-RQ3), maps them to contributions C1-C3, and records their numerical success criteria.

**Table 3.1: Research Questions and Success Thresholds**

| Contribution | Research question | Success threshold |
| --- | --- | --- |
| C1: MCP-TaskBench | Can a Model Context Protocol (MCP)-native benchmark distinguish between agents by evaluating both semantic tool-selection correctness and JavaScript Object Notation Remote Procedure Call (JSON-RPC) protocol conformance? | Cohen's d ≥ 0.3 between top and bottom quartiles; inter-annotator agreement κ ≥ 0.7. |
| C2: Reasoning-Trace Small Language Model (SLM) Distillation | Can Phi-4-mini be fine-tuned using Quantised Low-Rank Adaptation (QLoRA) and Direct Preference Optimisation (DPO) so that it performs MCP tool-calling within an 8-point Berkeley Function Calling Leaderboard (BFCL) F1 gap of a frontier teacher model? | BFCL F1 ≥ teacher − 0.08; MCP- TaskBench score within 10% of teacher; inference Video Random Access Memory (VRAM) ≤ 6 GB. |

---

<!-- PDF page 18 -->

| Contribution | Research question | Success threshold |
| --- | --- | --- |
| C3: Multi-Agent MCP + Agent-to-Agent Protocol (A2A) Knowledge Graph System | Does a decomposed LangGraph and AutoGen pipeline, connected through A2A, perform better than a monolithic single-agent baseline on multi-hop knowledge-graph queries? | Exact Match (EM) or task-completion improvement ≥ 5 percentage points over the monolithic baseline under paired evaluation. |

## 3.4 Evaluation Design

### 3.4.1 MCP-TaskBench Evaluation (C1) All MCP-capable agents are evaluated against the same task collection. Each task yields five scores: goal completion, accuracy of tool selection, validity of the JSON-RPC 2.0 envelope, alignment of provided arguments with the published schema, and structured recovery under injected failures. Benchmark discrimination is assessed by Cohen’s d [41] between agents in the highest and lowest score quartiles, with d ≥ 0.3 indicating an informative result. When automated oracles cannot fully judge quality, two independent annotators evaluate the output and Cohen’s κ [40] is reported, with κ ≥ 0.7 required. Figure 3.2 summarises comparison evaluation metrics.

<!-- Embedded image from source PDF page 18. -->

**Figure 3.2: MCP-TaskBench — A Multi-Dimensional Evaluation Framework for AI Agent Tool Use**

To determine the discriminatory capability of MCP-TaskBench, two baseline agents were incorporated in addition to the proposed system. Baseline Agent A was built as a ReAct-style agent [28], employing the same frontier teacher model as the primary system, while turning off graph retrieval and excluding the critic loop; it depended exclusively on vector-based semantic search and conventional RAG synthesis. This baseline serves to disentangle the impact

---

<!-- PDF page 19 -->

of graph-based structural retrieval and the bounded verification mechanism. Baseline Agent B was a zero-shot function-calling agent based on Mistral-7B-Instruct-v0.3 [59], with no fine-tuning, no in-context examples, and no retrieval augmentation other than the tool descriptions exposed via the MCP gateway. It constitutes a minimal viable MCP-capable agent. Both baselines used the same tool catalogue, schemas, and task suite as the proposed system.

### 3.4.2 Distillation Evaluation (C2) Assessing the distilled Phi-4-mini student against its frontier teacher involves BFCL F1 [1], the aggregate MCP- TaskBench score, median and 95th-percentile response latency, and maximum inference VRAM. The evaluation is considered successful only if the BFCL deficit is at most 8 F1 points, the MCP-TaskBench score is within 10% of the teacher’s result, and the system operates within 6 GB of VRAM. This VRAM limit aligns with commonly available workstation-class Graphics Processing Units (GPUs), making it a reasonable constraint for private inference. Figure 3.3 presents the validated trace admission and training procedure.

<!-- Embedded image from source PDF page 19. -->

**Figure 3.3: Verified Tool Distillation — A Robust Training Pipeline for Small Language Models**

### 3.4.3 Multi-Agent versus Monolithic Evaluation (C3)

The decomposed setup includes a LangGraph orchestrator, a KG retrieval agent, an AutoGen reconciliation agent, and a critic, all connected via A2A and limited to MCP-governed tools. This setup is compared to a monolithic agent provided with exactly the same model, data, and tool inventory. For each paired multi-hop KG query, both configurations generate an answer, and the difference is computed at the query level. The reported metrics are Exact Match (EM), task-completion rate, evidence faithfulness using the RAGAS framing [43], and total latency. Statistical significance is assessed using a paired bootstrap over queries and 95% confidence intervals. The two experimental topologies are contrasted in Figure 3.4.

---

<!-- PDF page 20 -->

<!-- Embedded image from source PDF page 20. -->

**Figure 3.4: Multi-Agent Systems for Knowledge Graph Reconciliation — A Comparative Research Framework**

## 3.5 Statistical Procedures

Cohen’s d is interpreted using 0.2, 0.5, and 0.8 as conventional benchmarks for small, medium, and large effect sizes [41], whereas Cohen’s κ quantifies annotator agreement [40]. Paired-comparison confidence intervals are produced using a percentile bootstrap with 10,000 query-level resamples. All tests are two-sided with a 5% significance level. For ablation-comparison families, the Holm-Bonferroni correction controls the family-wise probability of error.

Because the study evaluates a fixed set of implemented configurations rather than drawing inferences about an unspecified population of systems, prospective power analysis was not used as the main design criterion. Instead, the analysis combines predefined effect-size requirements with paired-bootstrap intervals, which avoids assuming normality while still making uncertainty in each comparison explicit.

## 3.6 Data Collection

Table 3.2 summarises the five data-acquisition streams. At no point did the study ingest proprietary organisational repositories or information.

**Table 3.2: Data Sources Used in the Evaluation Corpus**

| Stream | Contents | Collection method |
| --- | --- | --- |
| Repository corpus | 5 repositories, ~320,000 total LOC, Open-source Python and JavaScript repositories spanning libraries, services, and tooling | Public snapshots pinned by commit hash; parsed by Tree-sitter during ingestion |
| Synthetic schemas | 8 schemas, 24 injected conflicts, Enterprise-style relational schemas (customers, orders, billing, services) | Generated from templates; injected with deliberate cross-source conflicts for reconciliation tasks |

---

<!-- PDF page 21 -->

| Stream | Contents | Collection method |
| --- | --- | --- |
| MCP task suite | 360 instances across 6 categories, MCP-TaskBench task instances across defined categories | Hand-crafted seed tasks expanded through parameterised templates; each instance carries a machine-checkable oracle |
| Trace fixtures | 350 curated → 320 retained (91.4%) | Schema/tool replay admitted all 350; 30 were excluded by the configured cap |
| Live release measurements | 89-case holdout, 440-case no-split conformance, 5 RAG cases, 20 streaming samples, security suites | Machine-readable artifacts bound into the promotion record |
| Harness self-tests | 3 seeds across registry, student, and teacher labels | Deterministic oracle replay; explicitly ineligible for model claims |

Privacy and ethical requirements were incorporated into the study design. Attribution and licensing are preserved for public open-source artifacts; synthetic schemas avoid customer records; and local Small Language Model inference reflects the privacy rationale driving the work.

## 3.7 Assumptions and Constraints

- The prototype includes only open-source repositories and synthetic enterprise-style schemas; proprietary codebases are out of scope. • Given that enterprise privacy is a primary motivation, Small Language Model inference is restricted to a local or private environment. • Model Context Protocol (MCP) call is verified against its stated schema and recorded in an auditable log.

- A fixed upper bound on critic iterations ensures that verification cannot lead to unbounded response latency. • Where technically feasible, drift handling re-indexes only the affected regions instead of rebuilding the entire graph. • Because MCP and A2A are still under development, the implementation records the MCP specification revision dated 26 March 2025 [3] and the official a2a-sdk v1.0, and it documents any interpretation needed when the standards are not prescriptive. • The reference list points to supporting studies that remain preprints and provides their canonical publication links.

---

<!-- PDF page 22 -->

# 4. SYSTEM DESIGN AND ARCHITECTURE

## 4.1 Architectural Overview

The Intelligent Data Reconciliation and Knowledge Discovery (IDRKD) architecture is organised into six functional modules, which are consistently referred to as Pillars P1–P6. Although each pillar has its own specific responsibility, the primary research focus is on how these pillars behave together as a single integrated system. Table 4.1 summarises the modules.

<!-- Embedded image from source PDF page 22. -->

**Figure 4.1: IDRKD Six-Pillar Architecture Overview**

**Table 4.1: IDRKD System Modules and Responsibilities**

| No. | Module | Responsibility |
| --- | --- | --- |
| P1 | Structural Ingestion | Parses Python and JavaScript repositories using Tree-sitter; extracts code entities, schema fields, document spans, and relationships; computes deterministic fingerprints for idempotent updates. |
| P2 | Knowledge Graph | Stores code, data, document, and conflict entities in Neo4j; supports path queries, impact analysis, community detection (Louvain [26]) and importance ranking (PageRank [27]); records temporal conflict metadata. |
| P3 | Hybrid Retrieval and Agentic RAG | Combines pgvector semantic retrieval over an HNSW index [23] with graph traversal; fuses candidates using Reciprocal Rank Fusion (RRF) [22]; synthesises grounded answers and verifies claims through a bounded critic loop. |
| P4 | MCP + A2A Orchestration | Exposes vector, graph, conflict, and evaluation tools through the Model Context Protocol (MCP) over JSON-RPC 2.0 with schema validation and audit logging; uses the Agent-to- Agent Protocol (A2A) to delegate reconciliation tasks between LangGraph and AutoGen. |
| P5 | SLM Distillation | Fine-tunes Phi-4-mini on validated MCP reasoning traces using QLoRA [6] and aligns it with DPO [7]; evaluates the student against frontier teacher behaviour. |
| P6 | Drift Detection and Re-indexing | Detects entity-level and community-level semantic drift; triggers localised re-indexing of affected subgraphs rather than full graph rebuilds. |

---

<!-- PDF page 23 -->

## 4.2 High-Level Architecture

The end-to-end design is depicted in Figure 4.2. Processing begins in an event-driven ingestion tier, where repository contents, schema definitions, documents, and related artefacts are published onto a Kafka event bus [39], then parsed and transformed into structured representations. The resulting knowledge is distributed across Neo4j for graph relationships, Postgres together with pgvector for semantic vector retrieval, and MinIO for immutable raw artefacts and experiment replay. Specialist agents access these stores only through tools governed by MCP. At query time, inference is carried out using the local Small Language Model, while training is performed separately on validated reasoning trajectories; this separation enables private local operation while still using teacher-generated supervision during model preparation.

<!-- Embedded image from source PDF page 23. -->

**Figure 4.2: IDRKD End-to-End Architecture Flow**

## 4.3 Pillar Design Details

### 4.3.1 P1 — Structural Ingestion

At ingestion time, each repository is checked out at a fixed commit and analysed using Python and JavaScript Tree-sitter grammars [55]. From the produced concrete syntax tree, the worker extracts entities—including modules, classes, functions, schema tables and columns, and document sections—and relations such as imports, calls, read/write dependencies, and references. A deterministic fingerprint, derived from canonical content and source location, is attached to every entity, so unchanged material can be re-ingested idempotently and any modifications are detected exactly. The object store keeps the original artefacts to support replay of any archived state.

---

<!-- PDF page 24 -->

**Figure 4.3: Ingestion, Parsing, and Transactional Persistence Workflow**

<!-- Embedded image from source PDF page 24. -->

---

<!-- PDF page 25 -->

### 4.3.2 P2 — Knowledge Graph

Four families—code, data, documents, and conflicts—form the graph schema, connected by typed edges with preserved direction. A disagreement is modelled as a first-class entity: when sources describe a single business concept inconsistently, for example when a database type conflicts with an application programming interface definition, the conflict node records both assertions, their origins, and the time of detection. PageRank importance and Louvain community membership are materialised on nodes and recalculated during re-indexing. Cypher path queries with bounded depth are used to carry out impact analysis.

Graph Schema and Node Families The graph database is built on Neo4j Community Edition 5.x. Its schema represents four separate families connected by typed edges that preserve direction. CodeEntity nodes model repositories as modules, classes, and functions. DataEntity nodes capture implied structures, including database tables and columns. Document nodes represent unstructured sections, documentation files, and API definitions. Conflict nodes model disagreements as first-class entities. When sources describe a single business concept inconsistently, such as when a database type conflicts with an API definition, the conflict node records both assertions, their origins, and the timestamp at which the inconsistency was detected.

Offline Graph Analytics Because network algorithms are computationally costly, Neo4j Graph Data Science computations are separated from real-time query paths. Nightly Python batch jobs scheduled through APScheduler project a read-only, in-memory graph for each tenant. PageRank assesses node importance and stores the pagerank value directly on the nodes. Louvain clustering identifies code communities, saves community_id as static properties, and creates Community nodes connected through BELONGS_TO relationships.

Temporal Conflict Resolution During ingestion, when two variants representing the same entity appear, an attribute divergence score is computed using structural edit distance, nullability checks, and BGE-M3 embedding distance between descriptions. If the resulting score is greater than 0.3, a Conflict node is created. The specialised AutoGen Reconciliation Agent resolves these conflicts using deterministic policies. By default, Lamport Vector Clocks retain the variant with the higher logical clock, reflecting the happens-after relationship. Where competing versions are concurrent, a confidence-weighted policy orders them using static confidence metrics, and any remaining ties are resolved through a static source-priority configuration.

Secure Read Patterns AI agents are strictly prohibited from writing raw Cypher, which reduces injection risk and prevents tenant data leakage. Instead, they select pre-written, parameterised Cypher templates that automatically inject tenant_id scoping. The Graph Traversal Agent supports three primary read patterns. Impact analysis begins at a target node and traverses backward along CALLS for up to three hops; results are sorted by traversal depth and by pre-calculated PageRank in descending order so that the most critical calling files surface first. Path tracing uses Neo4j shortestPath across CALLS and REFERENCES for up to eight hops, supporting request-flow tracing. Community scoping follows BELONGS_TO relationships to locate sibling nodes in the same community, sorted by PageRank in descending order, thereby providing wider structural context.

---

<!-- PDF page 26 -->

<!-- Embedded image from source PDF page 26. -->

**Figure 4.4: Knowledge Graph Schema, Conflict Resolution, Analytics, and Traversal Patterns**

### 4.3.3 P3 — Hybrid Retrieval and Agentic RAG

Query execution starts by inferring intent and the expected number of reasoning hops. Vector and graph retrieval are then performed concurrently: graph traversal proceeds from linked anchor entities, while vector search orders embedded evidence spans by semantic similarity. RRF combines the two ranked result sequences without requiring score normalisation across the different retrieval methods [22]. A synthesiser generates the cited response, and then

---

<!-- PDF page 27 -->

the critic verifies every claim against the evidence and triggers targeted retrieval for any unsupported statements, up to a predetermined iteration limit.

The Pillar 3 workflow starts with state initialisation and planning. A live query begins by building a strongly typed QueryState dictionary that records variables throughout the request lifecycle, including accumulated vector context, graph context, generated claims, faithfulness scores, and the counter for the re-retrieval loop. The Planner Agent then processes the query with the local Phi-4-mini 3.8B model and categorises the intent as conceptual, structural, impact, multi-hop, or conflict.

After classification, the state machine sends the query to the appropriate retrieval route. Conceptual queries rely on semantic approximate nearest-neighbour search via pgvector using an HNSW index. Structural and impact queries carry out Neo4j graph traversal using preconfigured, parameterised Cypher templates that guard the graph against injection. Conflict queries are routed to a reconciliation node, which crosses the framework boundary over an mTLS-secured connection, using the a2a-sdk to call the external AutoGen agent. Multi-hop queries are broken down into sub-queries so that the vector and graph search paths can run in parallel, after which their results are merged.

The next phase executes context fusion and reranking. Because vector and graph databases produce distance scores on different scales, the parallel outputs are combined using Reciprocal Rank Fusion, enabling rank-order convergence without depending on complicated score scaling. The top 40 candidates are then fed into the ms-marco- MiniLM-L-6-v2 cross-encoder model, which reranks the evidence and reduces it to the top 10 most relevant candidates. This safeguards the Small Language Model context window while maintaining retrieval within the targeted latency budget.

The final phase performs grounded synthesis and bounded critique. The Phi-4-mini model, deployed through vLLM with PagedAttention, generates a draft answer using only the ranked context chunks. Before the answer is returned, the Critic Agent splits the draft into sentence-level claims and checks them against the retrieved evidence with a DeBERTa-v3-large-mnli Natural Language Inference model. If all claims exceed the entailment threshold, the answer is accepted. If any claim does not meet the threshold, the critic rewrites it as a focused sub-query and returns to vector retrieval for stronger evidence. To avoid increasing latency, the loop is limited to two rounds; if the cap is reached, the system returns the best available answer along with an explicit low-faithfulness warning and highlights unsupported spans.

---

<!-- PDF page 28 -->

**Figure 4.5: Agentic Query Processing and Grounded Response Workflow**

<!-- Embedded image from source PDF page 28. -->

---

<!-- PDF page 29 -->

### 4.3.4 P4 — MCP and A2A Orchestration

The target Pillar 4 design combines a JSON-RPC tool gateway with A2A cross-framework delegation. The prototype implements Pydantic/JSON Schema dispatch, tenant and scope checks, metrics, structured audit events, official A2A SDK cards/messages/execution, shared-secret card integrity, and an mTLS context builder. Durable Postgres audit storage and live LangGraph-to-AutoGen delegation are not implemented.

The executable sequence starts at a LangGraph `StateGraph`, which classifies and decomposes each query, fans out vector and graph retrieval in parallel, synthesises evidence, conditionally delegates reconciliation over the A2A SDK to an AutoGen `BaseChatAgent`, and applies a bounded critic route. The AutoGen agent invokes the same tenant-scoped Pydantic/MCP reconciliation contract used by the gateway. Audit records are emitted through an adapter, but the reference deployment does not persist them in an immutable `mcp_audit` table.

The A2A server/client tests now exercise a complete LangGraph-to-A2A-to-AutoGen-to-MCP round trip, including artifact return and conditional reconciliation. The reference deployment exposes the AutoGen reconciler as a separate service. Deployed mTLS and a committed live C3 result remain future evidence; implementation alone is not treated as proof that decomposition improves quality.

---

<!-- PDF page 30 -->

**Figure 4.6: MCP Gateway and A2A Bridge Architecture**

<!-- Embedded image from source PDF page 30. -->

### 4.3.5 P5 — SLM Distillation

---

<!-- PDF page 31 -->

Four consecutive activities make up the distillation process. First, teacher trajectories are generated via the staging tool gateway. Next, each tool operation is replayed during trace validation, and any trajectory that includes a failed step is discarded. Phi-4-mini is then trained on the accepted traces using supervised QLoRA, followed by DPO alignment with preference pairs constructed from accepted and rejected variants. This setup directly targets the cascading tool-error problem described in [10] and the intermediate-step validation rationale presented in [11]. Figure 4.7 illustrates trace generation, staging verification, and student optimisation.

The first phase involves generating teacher traces. The pipeline queries a large frontier teacher model—such as GPT-4o or Claude—and asks it to produce structured JSON reasoning traces that cover query classification, planned tool calls, and grounded synthesized responses.

The second phase consists of staging and trace validation. To mitigate the cascading tool-error issue, each generated trace is staged by re-executing the teacher’s proposed Cypher and vector search tool calls against the live system. If any query yields an error or an empty result, the trace is immediately discarded, so that the student model is trained only on valid data.

The third phase is supervised fine-tuning. The model is fine-tuned in a supervised manner using the Hugging Face trl, peft, and transformers libraries. Phi-4-mini (3.8B) is trained with 4-bit QLoRA so it can learn the proper syntax and ordering of the MCP tool-calling framework.

The fourth phase is the DPO alignment step. The student model runs a Direct Preference Optimisation (DPO) pass using DPOTrainer to enhance output faithfulness. Preference pairs are built by using critic-rejected, low-faithfulness drafts as the losing examples and fully grounded drafts as the winning examples. This steers the student model toward faithful answers without requiring an additional reward model.

The fifth phase covers quantisation and publication. The LoRA adapters are merged into the base weights, and the resulting artifact is quantised to 4-bit with AWQ, which retains tool-call accuracy better than GPTQ. A canonical descriptor covering the promoted evidence and every checkpoint file is cryptographically signed with Cosign and published alongside the Git LFS model artifact.

The sixth phase is gated offline evaluation. Before the model is promoted, it is processed through an offline evaluation harness. If any of the seven core gates, G1 through G7, fail or degrade by more than 3 points, publication is blocked and targeted corrective measures—such as doubling the training data or retuning the critic—are initiated.

The final phase is deployment of the inference server. After approval, a fail-closed launcher verifies the Cosign bundle, promotion record, manifest, Git LFS provenance, and every checkpoint hash before loading the model into vLLM. The hardened container uses offline model-loading settings, API authentication, immutable image digests, read-only mounts, and a non-root process. Network-enforced zero egress and an air-gapped registry remain future infrastructure controls and are not claimed by this deployment.

---

<!-- PDF page 32 -->

**Figure 4.7: Reasoning-Trace SLM Distillation and Deployment Pipeline**

<!-- Embedded image from source PDF page 32. -->

---

<!-- PDF page 33 -->

### 4.3.6 P6 — Drift Detection and Re-indexing

Ingestion events use fingerprints to identify altered entities. The drift component calculates cosine change between each entity’s newly computed and stored embeddings and, at the community level, quantifies movement between the updated and previously stored embedding centroids. If either threshold is exceeded, it triggers selective reconstruction of a scope that includes the entity, its community, and neighbouring nodes within a bounded radius. Full rebuilds are performed only when structural schema changes occur. Figure 4.8 summarises the scoring and selective-update workflow.

Phase 1: Ingestion and Event Trigger: The process starts when a developer pushes a code commit. After the Pillar 1 Ingestion Pipeline processes the raw files, it publishes an EntityChanged message to the entity-changed Kafka topic, configured with 12 partitions and a 14-day retention period. This design separates the real-time ingestion pipeline from downstream scoring and indexing activities.

Phase 2: Dual-Layer Drift Scoring Engine: The drift_scorer worker consumes Kafka events and assesses both fine-grained entity drift and wider structural drift.

At the entity level, the worker extracts the updated text, creates a new 1024-dimensional BGE-M3 embedding, and calculates its cosine distance relative to the stored embedding. If the drift is under 0.15, the entity is treated as current and only the last_verified timestamp is updated in Neo4j. Otherwise, the entity is flagged as stale (stale = true, staled_at = datetime()) and a DriftDetected event is published to Kafka.

At the cluster level, a nightly Celery Beat task evaluates domain-level drift across code modules. Active entities are grouped into Louvain communities, computed offline in Pillar 2, and the embedding centroid for each community is determined. If a community centroid moves by more than 0.10, every member entity is marked stale in Neo4j, thereby capturing macro-level architectural drift. In addition, a daily Celery task applies confidence decay to entities that have not been re-verified within 30 days, prioritizing recently updated files during retrieval.

Phase 3: The Decision Gate: When entities become stale, the system gauges the graph’s overall health by calculating the ratio of stale nodes to active nodes. If the ratio signals large-scale changes, such as a framework migration, the system skips Celery queueing and schedules a Full System Re-index. Otherwise, it issues a selective 2-Hop Re-index request to a Redis FIFO queue at redis://redis:6379/0. Phase 4: Asynchronous Re-index Engine: A pool of Celery workers pulls tasks from Redis and runs a five-step extraction pipeline. First, each worker retrieves the two-hop neighborhood of the stale node using a parameterised Neo4j Cypher Breadth-First Search query (MATCH p=(n:CodeEntity {id:$id})-[*1..2]-(m)), which specifies the local impact region. Second, it resolves the associated source file paths. Third, it re-parses those files with Tree-sitter for Python and JavaScript ASTs and SpanBERT for unstructured documents, producing updated entities and relationships.

Next, the system carries out a compensating logical transaction: raw file blobs are stored in the object store, before-images are staged in Postgres, structural metadata is updated in Neo4j using parameterised MERGE statements, and updated embeddings are written through pgvector. After success, the Postgres journal and `entity-changed` outbox are committed together. On failure, scoped compensations restore the before-images; this is not a cross-store ACID commit.

Phase 5: Back-Pressure and Fairness Controls: To manage large ingestion bursts, the scheduling layer uses two safeguards. Redis sorted sets provide a priority queue with per-tenant rate limiting, ensuring that no repository uses more than 30% of the active re-indexing workers. Prometheus monitors the reindex_lag_seconds metric; if a tenant’s p95 re-indexing lag exceeds 5 minutes, the system creates a warning ticket and triggers a critical alert, instructing Site Reliability Engineering (SRE) to scale the Celery workers.

---

<!-- PDF page 34 -->

**Figure 4.8: Dual-Layer Drift Detection and Selective Re-indexing Pipeline**

<!-- Embedded image from source PDF page 34. -->

---

<!-- PDF page 35 -->

## 4.4 Query-Time Control Flow

Figure 4.9 shows the runtime query sequence. After classifying the request, IDRKD carries out graph and vector retrieval, drafts a response, and routes its claims through a bounded critic. When a proposition is not supported, it causes targeted re-retrieval, but only up to the configured maximum, thereby limiting latency.

<!-- Embedded image from source PDF page 35. -->

**Figure 4.9: Bounded Agentic Retrieval-Augmented Generation (RAG) and Critic Loop**

## 4.5 Security and Governance Design

The governance model introduces controls motivated by MCP security studies [5]. Tool permissions are specific to tenants and constrained by JWT scopes; every argument is validated against a JSON Schema; every invocation is recorded in a structured append-only audit trail; retrieved data is kept separate from executable instructions to mitigate prompt injection; and A2A endpoints use mTLS. Throughout the dissertation implementation, proprietary repository content is excluded.

## 4.6 Knowledge Graph Data Model

To make traversal behaviour easier to understand, the Knowledge Graph model is deliberately constrained. Each node is assigned to one of four families and contains a persistent identifier, a deterministic content hash, an embedding pointer, PageRank and community attributes, and temporal fields for first observation and the most recent verification. Edges preserve an explicit type and direction. The primary elements are listed below. Extensions are permitted only when a canonical fingerprinting rule is defined, maintaining idempotent ingestion.

---

<!-- PDF page 36 -->

**Table 4.2: Principal Node and Edge Types of the IDRKD Graph**

| Element | Type | Meaning |
| --- | --- | --- |
| Module / Class / Function | Code node | Program units extracted from the syntax tree, with file path and span |
| Table / Column | Data node | Relational schema elements from ingested schema files |
| Document / Section | Document node | Operational documents and their addressable spans |
| Conflict | Conflict node | A recorded disagreement between two sourced claims about the same subject |
| IMPORTS / CALLS | Code edge | Static dependency and invocation relationships |
| READS / WRITES | Code–data edge | Data access relationships from code to schema elements |
| DESCRIBES / REFERENCES | Document edge | Documentation coverage and cross-references |
| CONFLICTS WITH / RESOLVED BY _ _ | Conflict edge | Links a conflict to its positions and, when applicable, its resolution |

Two design choices are especially important. A conflict is represented as a node rather than as metadata on an edge, which supports independent provenance, discovery time, resolution history, and direct retrieval of unresolved disagreements. Additionally, each node preserves its fingerprint, enabling the P6 drift process to determine exactly which stored assertions have been invalidated by a repository change.

## 4.7 Tools and Platform

**Table 4.3: Technical Specifications of the Prototype Platform**

| Technical parameter | Planned specification |
| --- | --- |
| Programming languages | Python for ingestion, training, evaluation, and agent services; JavaScript/TypeScript optional for User Interface (UI) and integration adapters. |
| Parsing | Tree-sitter grammars for Python and JavaScript in the Minimum Viable Product (MVP) [55]. |
| Graph database | Neo4j Community Edition with Cypher queries, PageRank, and Louvain community analytics. |
| Vector store | Postgres with pgvector and Hierarchical Navigable Small World (HNSW) index for semantic retrieval. |
| Object store | MinIO or Simple Storage Service (S3)-compatible storage for raw artefacts and replay support. |
| Agent orchestration | LangGraph for deterministic state-machine orchestration; AutoGen for the reconciliation agent. |
| Tool protocol | Model Context Protocol (MCP) over JSON-RPC 2.0 with Pydantic/JSON Schema validation. |
| Agent-to-agent protocol | Agent-to-Agent Protocol (A2A) via the official a2a-sdk v1.0 for cross-framework delegation and capability exchange. |
| Local model | Phi-4-mini, 3.8 billion parameters, fine-tuned using QLoRA and aligned using DPO. |
| Evaluation benchmarks | Berkeley Function Calling Leaderboard (BFCL), MCP-TaskBench, multi-hop Knowledge Graph question answering, repository-level QA, faithfulness and latency metrics. |
| Security controls | JWT-scoped tool access, tenant identifiers, structured audit logging, prompt-injection containment; no proprietary repository data. |
| Deployment target | Docker Compose or single-node Kubernetes-style reference stack suitable for dissertation evaluation. |
| Evaluation execution profile | Live student inference on a single NVIDIA L40S; no hosted-teacher comparison is included in the evidence package. |
| Hardware and run manifest | The release records Python, vLLM, PyTorch, CUDA, GPU identity, model manifest, checkpoint Git LFS hashes, and evidence hashes. CPU, driver, container digest, and peak VRAM are not recorded. |

---

<!-- PDF page 37 -->

Specific hardware details are recorded in archived run manifests rather than being imposed as fixed architectural requirements. Because the frontier teacher is hosted while the students run locally, the underlying hardware is neither controlled nor equivalent. Accordingly, Chapter 6 treats the latency numbers as operational observations rather than as a hardware-normalised benchmark.

## 4.8 System Context, Container and Deployment Views

The system design is cross-validated against the accompanying High-Level Design and Low-Level Design (HLD/LLD). The following views clarify the system boundary, deployment units, and separate trust zones for training and inference. They adopt the C4 modelling approach, which separates context from container perspectives so that stakeholders can examine responsibilities and communication at an appropriate level of abstraction [45].

### 4.8.1 System Context and External Boundaries IDRKD mediates between enterprise-style systems of record and the stakeholders who interrogate them. Engineers pose questions in natural language, platform and Site Reliability Engineering (SRE) staff monitor health, and researchers execute MCP-TaskBench benchmarks and ablation experiments. Repository code, database schemas, API definitions, and documents are accepted only through governed ingestion pathways. Access to the frontier teacher is available solely via the isolated training route; the production route depends on the locally deployed Small Language Model and does not require any outbound connection to an external model service.

---

<!-- PDF page 38 -->

<!-- Embedded image from source PDF page 38. -->

**Figure 4.10: IDRKD System Context and External Trust Boundaries**

### 4.8.2 Container View and Service Responsibilities The container-based representation maps each of the six pillars to specific deployable components. Neo4j, Postgres with pgvector, MinIO, Redis, and Kafka constitute the stateful tier. Stateless services include parser workers, the MCP gateway, LangGraph orchestration, the AutoGen reconciliation agent, drift detection, re-index workers, the benchmark harness, and the local inference endpoint. A separate observability plane consolidates metrics, traces, and structured events, allowing a single correlation identifier to link every stage of a query.

---

<!-- PDF page 39 -->

<!-- Embedded image from source PDF page 39. -->

**Figure 4.11: IDRKD Container View and Service Responsibilities**

### 4.8.3 Deployment Topology and Training Isolation The reference system may be deployed either with Docker Compose or within a single Kubernetes-style namespace. Training resources and inference resources run in separate environments. Credentials for the teacher are exposed only within the training zone, and only signed, version-controlled model packages are moved across the boundary through object storage. The production policy permits local inference while blocking outbound language-model communication. This design preserves the privacy objective while maintaining reproducibility in the research process.

---

<!-- PDF page 40 -->

<!-- Embedded image from source PDF page 40. -->

**Figure 4.12: Deployment Topology and Training-Inference Isolation**

## 4.9 Architecture Decisions and Quality Attributes

### 4.9.1 Architecture Decision Record Architecturally significant choices are documented as Architecture Decision Records (ADRs), with each record capturing the problem context, the selected alternative, and the resulting trade-offs [46]. Table 4.4 lists the decisions that most directly determine the system’s technical behaviour.

| ADR | Decision | Rationale | Consequence |
| --- | --- | --- | --- |
| ADR-001 | Neo4j Community for the Knowledge Graph | Typed relationships, Cypher traversal, PageRank, and Louvain analytics are available in one graph platform. | Single-node reference deployment; high availability is a documented migration path. |
| ADR-002 | Postgres with pgvector for vector retrieval | Metadata and vector rows share transactional storage and tenant filters; HNSW supports approximate nearest-neighbour search. | A dedicated vector database becomes a migration option at larger scale. |
| ADR-003 | LangGraph plus AutoGen | LangGraph manages stateful cycles and the critic loop; AutoGen provides a separate reconciliation role for the A2A experiment. | Two framework dependencies are retained because they directly support C3. |
| ADR-004 | Phi-4-mini as the student model | The model is compact enough for local inference and QLoRA-based adaptation under the dissertation compute budget. | The claim is limited to task-specific tool use, not general frontier-model equivalence. |
| ADR-005 | MCP for all tool invocation | A single JSON-RPC and schema-governed envelope provides uniform validation, auditing, and replaceable agents. | A2A is reserved for communication across framework boundaries. |
| ADR-006 | Bounded critic loop | At most two additional retrieval rounds are permitted before returning the best supported answer with unresolved claims marked. | Worst-case latency is bounded and failure behaviour is explicit. |

### 4.9.2 Quality Attributes and Service-Level Objectives The HLD/LLD defines quantitative Service-Level Objectives (SLOs), enabling operational characteristics to be verified empirically rather than being stated only in qualitative terms. Table 4.5 presents the Minimum Viable

---

<!-- PDF page 41 -->

Product (MVP) design targets. Chapter 6 reports experimental measurements tied to the research questions and the selected ablations; targets not included in those runs remain architectural objectives instead of measured claims.

| Quality attribute | Metric | MVP target |
| --- | --- | --- |
| Single-hop query latency | p50 end-to-end answer time | ≤ 3.0 s |
| Query latency with re-retrieval | p95 end-to-end time | ≤ 8.0 s |
| Local inference | Time to first token for Phi-4-mini | ≤ 1.2 s |
| Ingestion freshness | Git webhook to graph update, typical commit | ≤ 5.0 s |
| Drift detection lag | Commit to stale flag on affected entities | ≤ 60 s |
| Hybrid retrieval | Top-10 recall on the selected development set | ≥ 0.75 |
| MCP tool reliability | BFCL F1 for MCP-formatted calls | ≥ 0.82 |
| Answer faithfulness | NLI/RAGAS-aligned faithfulness score | ≥ 0.78 |
| Inference memory | Peak 4-bit inference VRAM | ≤ 6 GB |

## 4.10 Canonical Data and Protocol Contracts

### 4.10.1 Store Ownership and Canonical Identifiers Three canonical stores are assigned mutually exclusive ownership. Neo4j is authoritative for entity identities and typed relationships; Postgres with pgvector owns indexed chunks, vector representations, and MCP audit records; and the object store retains immutable source artifacts, repository versions, and signed model bundles. Redis is limited to caches and asynchronous queues and is not authoritative. Every persisted entity includes a stable identifier scoped to the tenant, an origin location and revision, a deterministic fingerprint, and a timestamp indicating the last verification

| Store | Owned data | Identity and consistency rule |
| --- | --- | --- |
| Neo4j | Code, data, document, conflict, and community nodes; typed relationships | Stable entity identifier and deterministic fingerprint |
| Postgres + pgvector | Chunks, embeddings, metadata filters, MCP audit rows | tenant id + entity id + content hash |
| Object store | Raw files, commit snapshots, training datasets, model artefacts | _ _ _ Content-addressed object key and checksum |
| Kafka | Commit, entity-change, drift, re-index, and repair events | Partition key by repository or entity |
| Redis | Hot-query cache and re-index/A2A work queues | Not a source of truth; entries are rebuildable |

For every disagreement, the conflict node retains both assertions along with their provenance, confidence, source revision, and detection time. Automated resolution begins by comparing logical clocks and applying the configured source-priority policy only for concurrent changes, consistent with Lamport’s event-ordering principles [54]. If temporal ordering is not sufficient, a confidence rule or a human decision may be used. The chosen resolution is appended immutably to the audit history, and the superseded assertion is kept rather than erased.

### 4.10.2 MCP Message and Tool Contract A single, typed parameter definition functions as the source of truth for every MCP tool contract. It produces the published JavaScript Object Notation (JSON) Schema and also validates incoming JSON-RPC 2.0 messages, thereby reducing divergence between documentation and implementation [3]. The request envelope contains jsonrpc, id, method, and params; a response returns either result or a structured error together with metadata such as correlation identifier and execution latency. Each call is scoped to a tenant, validated, and written to the audit store before completion.

---

<!-- PDF page 42 -->

**Table 4.7: MCP Request, Response, and Audit Contract Summary**

| Contract element | Required content | Control objective |
| --- | --- | --- |
| Request envelope | jsonrpc='2.0', id, method, params | Reject malformed messages before dispatch |
| Tool parameters | Pydantic/JSON Schema model per tool | Validate type, range, required fields, and tenant filters |
| Success response | result plus metadata | Include correlation id, tool name, and latency _ |
| Error response | code, message, optional data | Return structured and recoverable errors instead of free-form failure text |
| Audit record | principal, tenant, tool, redacted arguments, status, duration | Append-only record linked to the end-to-end trace |

---

<!-- PDF page 43 -->

# 5. IMPLEMENTATION

This chapter maps the six-pillar design to the research prototype. To avoid conflating architecture with evidence, capabilities are identified as implemented, partially implemented, or design targets. Pseudocode describes the intended complete behavior; where the executable prototype is narrower, that boundary is stated explicitly. Chapter 6 reports only measurements backed by committed machine-readable artifacts.

## 5.1 Implementation Phases

**Table 5.1: Implementation Phases and Delivery Status**

| Phase | Timeline | Work delivered | Status |
| --- | --- | --- | --- |
| 1. Foundation and ingestion MVP | Weeks 1–2 | Docker Compose stack; signed webhook; manual-commit Kafka consumer; immutable MinIO archive; Tree-sitter ingestion; Postgres saga journal/outbox; Neo4j and pgvector writes; compensating rollback, repair worker, and DLQ. Repository checkout automation and production retry tuning remain deployment work. | Implemented; deployment validation pending |
| 2. Graph and retrieval MVP | Weeks 3–4 | Parameterised graph traversal, vector search, Reciprocal Rank Fusion (RRF), reranking, and query–answer flow. | Implemented |
| 3. MCP tool gateway | Weeks 5–6 | JSON-RPC 2.0-style tools with Pydantic schema validation, tenant checks, metrics, and structured errors. Durable audit persistence remains partial. | Partial |
| 4. MCP-TaskBench | Weeks 7–8 | 440 internal cases, case-aligned scoring, deterministic train/holdout split, and live-model conformance execution. Independent authorship and human grading remain future work. | Implemented internally |
| 5. SLM trace generation and fine-tuning | Weeks 9–11 | Curated trace fixtures, schema admission, QLoRA/DPO execution, llm-compressor AWQ, and vLLM release gates. Hosted-teacher provenance is not established by the committed fixtures. | Partial |
| 6. A2A and reconciliation agent | Weeks 12–13 | LangGraph state graph, official A2A SDK bridge, separate AutoGen reconciliation agent, tenant-scoped MCP dispatch, transport configuration, and paired C3 harness. A live C3 artifact and deployed mTLS exchange remain pending. | Implemented; evaluation pending |
| 7. Drift detection and re-indexing | Weeks 14–15 | Entity/centroid drift scoring and Redis-backed two-hop re-index workers. Fair scheduling and full-rebuild policy automation remain design targets. | Partial |
| 8. Evaluation and ablations | Weeks 16–18 | Live holdout, full conformance, RAG faithfulness, streaming, security, and release evidence. Comparative baselines, human annotation, confidence intervals, and ablations remain unexecuted. | Partial |
| 9. Final report and paper draft | Weeks 19–22 | Evidence-aligned dissertation and reproducibility package. | In progress |

## 5.2 P1 — Structural Ingestion with Deterministic Fingerprints

The executable ingestion path accepts HMAC-authenticated commit webhooks, waits for Kafka broker acknowledgement, and consumes typed events with manual offset commits. It stores content-addressed source blobs and an immutable event manifest in MinIO before parsing, stages deterministic IDs and before-images in a Postgres saga journal, and then writes Neo4j and pgvector. A transactional outbox publishes `entity-changed` after commit. Deterministic fingerprints and Neo4j `MERGE` operations provide idempotent identity; duplicate event IDs are not applied again.

Algorithm 1: Idempotent structural ingestion

procedure INGEST_TARGET(change_event): verify_hmac(change_event); archive_manifest ← archive_immutable(change_event.files); staged ← persist_plan_and_before_images(change_event, archive_manifest); try: graph.upsert(staged.entities, staged.relations); vectors.upsert(staged.embeddings); commit_journal_and_outbox(staged, "entity-changed") catch error: compensate_vectors(staged.before_vectors); compensate_graph(staged.before_graph); publish_dlq(error); if compensation_failed: publish_repair(error)

---

<!-- PDF page 44 -->

## 5.3 P2 — Knowledge Graph Construction and Governed Query Access

The knowledge graph is deployed on Neo4j Community. Schema preparation is performed only once during bootstrap: composite uniqueness constraints are defined on the (tenant_id, entity_id) pair for each canonical node label, and supporting indexes are created for anchor lookup by qualified name, by file path, and by fingerprint. Node labels and relationship types are taken from the single canonical vocabulary specified in Section 4.3.2, ensuring that code, documentation, and schema evidence all use the same addressable namespace. All write operations are sent to the graph by the ingestion worker as MERGE actions instead of CREATE, which makes replay after a partially completed logical transaction safe: repeating an event results in the same node and edge set rather than duplicating it. When concurrent writers modify the same entity, their updates are ordered by the Lamport timestamp included in the entity-changed message, so a late-arriving revision is discarded instead of overwriting a newer one. Algorithm 2 formalizes this approach.

Algorithm 2: Idempotent graph upsert with Lamport-ordered conflict resolution

procedure GRAPH_UPSERT(entity e, edges E, lamport ts, tenant t): begin_transaction() cur ← lookup(tenant=t, entity_id=e.id) // parameterised Cypher if cur ≠ ∅ ∧ cur.lamport ≥ ts: abort_transaction(); return STALE // discard late revision MERGE node n on (tenant_id=t, entity_id=e.id) // idempotent write on create: n ← props(e); n.lamport ← ts; n.created ← now() on match: n ← props(e); n.lamport ← ts; n.updated ← now() set_labels(n, canonical_label(e.kind)) for each relationship r in E: MERGE src on (tenant_id=t, entity_id=r.src) MERGE dst on (tenant_id=t, entity_id=r.dst) MERGE edge (src)-[r.type]->(dst); edge.lamport ← ts prune_edges(n, E) // drop edges absent from this revision commit_transaction(); return APPLIED Graph analytics are computed outside the write path, keeping ingestion latency and query latency independent of them. Each night, a job projects the tenant subgraph into the Graph Data Science in-memory catalog, runs PageRank with a damping factor of 0.85 to measure structural importance, and runs Louvain to assign community membership. The outputs are written back as node properties and then used downstream: importance is used for candidate ordering in P3, while the per-community embedding centroids produced by the same job establish the baseline against which P6 evaluates community-centroid drift. Since the job is scheduled rather than transactional, its analytics properties may lag behind the graph by no more than one cycle. Algorithm 3 specifies the refresh sequence.

Algorithm 3: Nightly graph analytics refresh and community centroid baseline

procedure ANALYTICS_REFRESH(tenant t): g ← project_subgraph(t, labels=canonical, rels=structural) pr ← PageRank(g, damping=0.85, tolerance=1e-7, max_iter=20) for each node n in g: write_property(n, importance, pr[n]) com ← Louvain(g, max_levels=10) for each node n in g: write_property(n, community_id, com[n]) for each community C in com: centroid(C) ← mean({ embedding(n): n ∈ C }) store_centroid(t, C, centroid(C)) // P6 drift baseline drop_projection(g); emit(analytics_refreshed, t) Read access is intentionally constrained. At query time, every traversal originates from a fixed library of parameterized Cypher patterns; any query text provided by a user is supplied as a parameter and is never concatenated into a statement, preventing it from being interpreted by the planner as syntax. Each pattern in the library includes a required tenant predicate and an explicit maximum traversal depth, which structurally prevents

---

<!-- PDF page 45 -->

both cross-tenant disclosure and unbounded expansion rather than merely discouraging them. The graph_query, bfs_impact, path_trace, and community_scope tools made available through the MCP gateway are thin typed wrappers over this pattern library, ensuring that the tool interface and the query interface cannot diverge. Each pattern is profiled during the build so that index usage is verified rather than assumed; any pattern that falls back to a label scan fails the contract tests described in Section 5.10.3.

## 5.3 P3 — Hybrid Retrieval with Reciprocal Rank Fusion and Bounded Critic Loop

During retrieval, the executable orchestrator combines graph and vector candidates with RRF and optional MiniLM reranking. A bounded loop repeats retrieval and synthesis when the whole-answer faithfulness score is below threshold. It does not yet perform intent-specific routing, sentence-level claim decomposition, or targeted query rewriting; those operations remain the complete design shown in Algorithms 2 and 3.

Algorithm 2: Hybrid retrieval with Reciprocal Rank Fusion

procedure RETRIEVE(query): anchors ← classify_and_link(query) // intent + entity linking G ← graph_traverse(anchors, max_hops=h(query)) V ← vector_search(embed(query), top_k) for each candidate c in G ∪ V: score(c) ← Σ_lists 1 / (60 + rank_list(c)) // RRF, k = 60 return top_n candidates by score

Algorithm 3: Bounded critic loop for answer verification

procedure ANSWER(query): E ← RETRIEVE(query); draft ← synthesise(query, E) for iter ← 1 to MAX_ITERS: unsupported ← { claim ∈ claims(draft): ¬supported(claim, E) } if unsupported = ∅: return finalise(draft, E) E ← E ∪ RETRIEVE(focus(unsupported)) // targeted re-retrieval draft ← revise(draft, E, unsupported) return finalise(draft, E) with unresolved claims flagged

## 5.4 P4 — MCP Tool Gateway and MCP-TaskBench Scoring

A JSON Schema is provided for every registered tool. Before dispatch, the gateway validates each JSON-RPC request and returns a structured error on failure. The executable TaskBench harness measures case-aligned tool selection, exact arguments, schema validity, protocol execution, required result keys, and semantic availability separately. Human partial-credit and fault-recovery agreement are not yet measured; Algorithm 4 remains the target composite benchmark.

The implemented MCP catalog contains `search_code`, `get_entity`, `graph_bfs`, `graph_path`, `get_community`, `enqueue_reindex`, `schema_diff`, `impact_analysis`, `reconcile`, `resolve_conflict`, `get_centroid_drift`, and `get_conflict`. Each tool has typed inputs, structured errors, and tenant-scoped checks. Internal TaskBench cases carry tool-call oracles; semantic result oracles are available only where fixture or live-repository expectations are explicitly supplied.

**Table 5.2: MCP-TaskBench Scoring Dimensions**

| Dimension | What is measured | Oracle |
| --- | --- | --- |
| Task success | Did the interaction achieve the task goal? | Machine-checkable oracle per task; human grading for partial credit (κ reported) |
| Tool selection | Was the semantically correct tool chosen? | Gold tool set per task |
| JSON-RPC validity | Is the request a well-formed JSON-RPC 2.0 message (id, method, params)? | Protocol validator |

---

<!-- PDF page 46 -->

| Dimension | What is measured | Oracle |
| --- | --- | --- |
| Schema conformance | Do the arguments validate against the advertised tool schema? | JSON Schema validation |
| Error handling | Under injected faults, does the agent surface structured, recoverable errors? | Fault-injection harness |

Algorithm 4: Protocol-conformance scoring for one agent–task pair

procedure SCORE(agent, task): transcript ← run(agent, task, gateway=instrumented) s_task ← oracle(task, transcript.outcome) s_tool ← [chosen_tools(transcript) = gold_tools(task)] s_rpc ← fraction of requests passing JSON-RPC 2.0 validation s_schema ← fraction of requests whose params validate against schema s_err ← graded behaviour under injected faults(task) return weighted_aggregate(s_task, s_tool, s_rpc, s_schema, s_err)

## 5.5 P5 — Verified Trace Distillation into Phi-4-mini

The training code applies schema, tool-use, and faithfulness admission checks before dataset construction. The committed 350-record archive is a curated fixture corpus: all records passed schema replay and 30 were excluded by the 320-record cap, not by observed execution failure. TaskBench SFT/DPO datasets instead use a deterministic 80/20 prompt-group split and generated hard negatives. QLoRA, DPO, adapter merging, AWQ, and vLLM serving were executed, but the exact split-v2 adapter run summaries still need to be published for complete training provenance. Algorithm 5 describes the stronger hosted-teacher workflow that remains to be executed.

Algorithm 5: Staged-trace validation and two-stage training

procedure BUILD_STUDENT(tasks): T ← ∅ // admitted traces for each task in tasks: trace ← teacher.solve(task, gateway=staging) ok ← replay(trace, staging) ∧ oracle(task, trace.outcome) if ok: T ← T ∪ {trace} else: R ← R ∪ {trace} // rejected pool student ← QLoRA_finetune(Phi4mini, supervised_pairs(T)) prefs ← { (t⁺ ∈ T, t⁻ ∈ R) matched by task } student ← DPO_align(student, prefs) return student

## 5.6 P4/P6 — A2A Delegation and Drift-Aware Selective Re-indexing

The prototype implements A2A SDK cards, messages, task execution, HMAC card verification, optional mTLS client configuration, and a LangGraph route that delegates reconciliation to a separate AutoGen service. Drift workers implement entity and community scoring plus bounded re-indexing through Redis queues. Kafka linkage, per-tenant fair scheduling, expiry-based staleness, and automated full rebuilds remain design targets. Algorithm 6 describes the intended integrated flow.

Algorithm 6: Dual-layer drift scoring with selective re-indexing

procedure DRIFT(affected_set): for each entity e in affected_set: d_e ← 1 − cos(embed_new(e), embed_stored(e)) // entity drift C ← community(e) d_c ← ‖centroid_new(C) − centroid_stored(C)‖₂ // community drift if d_e > θ_e ∨ d_c > θ_c: scope ← e ∪ C ∪ neighbours(e, radius=r) reindex(scope) // embeddings, analytics, materialised views else: update_in_place(e) // cheap path

---

<!-- PDF page 47 -->

Values for θ_e, θ_c, and neighbourhood radius r are selected using a held-out history of changes, and Section 6.5 reports the sensitivity analysis. Complete re-indexing is invoked only for structural schema modifications, which were infrequent in the observed distribution of changes.

## 5.7 Deployment and Operational Considerations

The reference Docker Compose environment defines Kafka and topic bootstrap, Neo4j, Postgres with pgvector, Redis, MinIO, observability services, signed ingestion webhook, ingestion and repair workers, the MCP gateway, a separate AutoGen A2A reconciliation service, re-index workers, and optional local model serving. LangGraph runs in the query/evaluation process rather than as a standalone service. The release package captures model/runtime evidence, while complete container digests, a production secret manager, and training-run provenance remain to be added.

Implementation explicitly required observability, bounded execution, and replay. For every tool call, delegation, and re-index decision, the system produces structured records, which proved essential for debugging. The implementation sets limits for critic iterations, traversal depth, per-query tool calls, and A2A timeouts, thereby converting worst-case latency into an explicit system parameter. In addition, keeping raw artefacts, fingerprints, and audit histories enables previously generated answers to be regenerated and examined, supporting both engineering analysis and scientific reproducibility.

An important implementation finding emerged from schema evolution within the tool catalogue: an action that is valid for one tool schema may become invalid under a later revision. Accordingly, the gateway versions all schemas, and each MCP-TaskBench task logs the required version, ensuring that conformance outcomes from separate runs remain directly comparable.

## 5.8 Detailed Runtime and Resilience Flows

### 5.8.1 Event-Driven Ingestion and Logical Transaction The prototype exposes an HMAC-verified Git-commit webhook, Kafka producer, manual-commit consumer, immutable MinIO source archive, Postgres saga journal and outbox, Neo4j and pgvector writers, and dedicated repair and dead-letter topics. Invalid Kafka records are routed to the DLQ so that a poison record cannot indefinitely block a partition. Kafka offsets advance only after the saga result and its pending outbox messages have been durably published [39].

The logical transaction stores raw content first, snapshots affected graph and vector records, and persists the staged plan before either destination is changed. On a later failure, compensation deletes records created by the event and restores overwritten records from their before-images. If compensation itself fails, the journal remains in `repair_required`, a repair notification is emitted, and the repair worker retries from the durable plan. The final journal transition and outbox inserts share one Postgres transaction. This coordinator provides replayable saga semantics; it must not be treated as evidence of cross-store ACID atomicity.

---

<!-- PDF page 48 -->

<!-- Embedded image from source PDF page 48. -->

**Figure 5.1: Event-Driven Ingestion and Logical Transaction Flow**

For documentation and API text, extraction includes span-based Named Entity Recognition (NER), whereas JSON and comma-separated-value inputs undergo schema inference. SpanBERT is suitable for NER because its training objective models and predicts complete contiguous spans rather than treating tokens independently [53]. The outputs from Tree-sitter, NER, and schema inference are mapped to a single canonical entity vocabulary before fingerprints are generated, which allows code, documents, and schema evidence to coexist within the same graph.

### 5.8.2 Query State Machine and Bounded Verification The executable orchestrator maintains a typed `QueryState` and runs a fixed sequence of router marker, vector retrieval, graph traversal, RRF, optional reranking, synthesis, and critique. A maximum of two rounds bounds execution. The router does not yet classify intent, decompose multi-hop requests, run retrieval branches concurrently, or delegate conflicts. The LangGraph node topology described in the design figures is a migration target, not the framework used by the measured release.

The measured release uses `cross-encoder/nli-deberta-v3-large` to score the complete answer against concatenated evidence at threshold 0.78. When the score fails, the same query is retrieved again; sentence-level claims, targeted query rewriting, and unsupported-span annotation remain future improvements. The query execution procedure returns after acceptance or after the bounded second round.

---

<!-- PDF page 49 -->

<!-- Embedded image from source PDF page 49. -->

**Figure 5.2: Target LangGraph State Machine for Query Resolution**

BGE-M3 supplies the initial bi-encoder representations, pgvector HNSW performs approximate nearest-neighbour search, and an MS MARCO MiniLM cross-encoder reorders the merged candidates. The engineering schedule allots approximately 40 ms to query embedding, 60 ms to vector retrieval, 180 ms to reranking up to 40 candidates, 40 ms to an indexed single-hop Cypher operation, and at most 300 ms to a bounded three-hop traversal. These values act as diagnostic targets for detecting regressions and are treated as measurements only when they are observed during benchmark execution.

### 5.8.3 A2A Delegation and Cross-Agent Governance The prototype uses a compiled LangGraph `StateGraph` for classification, decomposition, parallel retrieval, synthesis, conditional delegation, reconciliation, and critic routing. Reconciliation crosses the official A2A SDK boundary to an AutoGen `BaseChatAgent`, which invokes the governed tenant-scoped MCP tool and returns a JSON artifact to LangGraph. Shared-secret card signing and an mTLS client-context builder are also supplied. A short-lived token issuer and deployed mTLS exchange remain absent. The paired C3 harness can execute this topology and the fixed-loop baseline on identical cases, but no live C3 artifact is committed yet [15], [16].

---

<!-- PDF page 50 -->

<!-- Embedded image from source PDF page 50. -->

**Figure 5.3: Target A2A Delegation from LangGraph to an AutoGen Reconciliation Agent**

### 5.8.4 Error Handling, Retry, and Back-Pressure Rules

| Failure condition | Response | Design objective |
| --- | --- | --- |
| Webhook cannot publish to Kafka | Return HTTP 503 and rely on source-control webhook retry | No unlogged ingestion acceptance |
| Tree-sitter contains ERROR nodes | Skip the invalid subtree, continue valid extraction, and emit a warning | Partial progress with visible data-quality status |
| Neo4j write fails | Restore graph/vector before-images; emit repair and DLQ messages if compensation fails | Replayable failure without accepting a partial logical commit |
| Postgres commit fails after Neo4j commit | Compensate graph and vector writes; retain `repair_required` state when restoration fails | Restore graph-vector consistency |
| Object-store write fails | Stop before graph/vector mutation and emit a DLQ record | Raw source remains a hard prerequisite for replay |
| MCP tool timeout | Return a structured timeout error and record duration | Agent may recover without hallucinating a result |
| A2A target unavailable | Retry with jitter, apply circuit breaker, then return controlled delegation failure | Protect the main query path from indefinite waiting |
| Re-index queue pressure | Per-tenant rate limits and fair scheduling | Prevent one noisy repository from monopolising workers |

## 5.9 Security, Observability, and Delivery Controls

### 5.9.1 Threat Model and Prompt-Injection Containment The threat assessment applies STRIDE to spoofing, tampering, repudiation, information disclosure, denial of service, and privilege escalation across ingestion, agent-tool interaction, data-store access, A2A exchange, training, and model publication [52]. Defensive controls are layered: retrieved passages are prevented from entering the system-instruction scope; tool outputs are wrapped in typed non-executable structures; potentially suspect content is quarantined before synthesis; JWT permissions restrict tool access; mTLS authenticates agents; sensitive audit

---

<!-- PDF page 51 -->

fields are redacted; and model signatures are verified before loading. Credential boundaries align with execution zones. Frontier-teacher secrets are available only inside temporary training jobs and are never exposed to the reference inference stack. Database secrets and JWT signing material are rotated, model-signing keys are stored outside the runtime cluster, and LoRA adapters along with quantised checkpoints are fingerprinted prior to registration. As a result, local inference, MCP execution, and publication operate under separate trust domains.

| Asset or boundary | Primary threat | Control |
| --- | --- | --- |
| Ingested code and documents | Prompt injection or data exfiltration | Untrusted-context delimiters, injection screening, no production egress |
| MCP gateway | Unauthorised or malformed tool use | JWT tool scopes, JSON Schema validation, rate limits, structured audit |
| A2A bridge | Agent impersonation or cross-tenant delegation | mTLS, signed capability records, tenant-bound task envelope |
| Knowledge stores | Cross-tenant reads or Cypher injection | Parameterised queries and mandatory tenant predicates |
| Training traces | Poisoned or failing teacher behaviour | Re-execution validation, dataset hashing, sampled manual review |
| Model artefacts | Substituted or corrupted weights/adapters | Signed artefacts, checksum validation, controlled registry |
| Inference service | Long-context denial of service | Token limits, quotas, circuit breaker, bounded agent loops |

### 5.9.2 Observability and SLO-Based Alerting OpenTelemetry helpers and Prometheus collectors instrument selected ingestion, graph, MCP, and drift operations [51]. Correlation fields exist in commit and A2A envelopes, but end-to-end propagation across every store and model request has not been demonstrated. The alert conditions below remain operating targets until exporter and alert-rule evidence is archived.

| Metric | Type | Labels | Operational use |
| --- | --- | --- | --- |
| query latency seconds _ _ | Histogram | query type, path _ | Page when p95 exceeds twice the SLO for five minutes |
| mcp tool calls total | Counter | tool, status, tenant | Alert on sustained protocol or schema errors |
| _ _ _ critic reretrieve rounds | Histogram | outcome | Review when give-up rate exceeds baseline |
| _ _ faithfulness score | Histogram | query type | Ticket when rolling p50 falls below 0.78 |
| _ ingest latency seconds | Histogram | _ parse/write/embed | Page when p95 exceeds twice the ingestion SLO |
| _ _ drift events total | Counter | tenant | Dashboard for anomalous change rate |
| _ _ reindex lag seconds | Histogram | tenant | Ticket when p95 exceeds five minutes |
| _ _ kafka consumer lag | Gauge | topic, partition, group | Alert before freshness objectives are breached |
| _ _ slm tokens per second | Gauge | model | Capacity and regression tracking |

### 5.9.3 Evaluation Harness, CI/CD, and Reproducibility Evaluation is a standalone package that writes machine-readable TaskBench, RAG, streaming, security, and promotion artifacts. Tool calls are case-aligned and semantic backend outcomes are now reported separately from protocol success. Human grading, official BFCL execution, container digests, and protocol-revision fields are not present in the archived release. The three-seed deterministic bundle is explicitly a harness self-test using oracle predictors and is ineligible for empirical model claims. Continuous integration runs unit/evaluation tests, Ruff, and strict mypy; live GPU/database gates remain separately executed release procedures.

| Environment | Purpose | Mandatory gate |
| --- | --- | --- |
| Development | Feature work on synthetic fixtures | Lint, type check, unit tests, contract tests |
| Continuous integration | Pull-request validation | Five-task MCP smoke benchmark and schema compatibility |
| Benchmark | Frozen reproducible evaluation | Full task suite, fixed seeds, archived manifests |

---

<!-- PDF page 52 -->

| Environment | Purpose | Mandatory gate |
| --- | --- | --- |
| Training | Teacher trace generation, QLoRA, and DPO | Isolated egress and signed output artefacts |
| Reference deployment | Demonstration and thesis evaluation | Tagged release with fixed images and configuration |

### 5.9.4 Capacity Planning and Migration Triggers The Minimum Viable Product is designed to support a single repository with roughly 500,000 lines of code, close to 80,000 graph entities, and about 250,000 searchable chunks. Neo4j Community and pgvector remain appropriate at this scale. The documented migration criteria are as follows: a specialized vector database is required once the system exceeds approximately ten million vectors; Neo4j Enterprise is needed to provide high availability via clustering; additional vLLM replicas are added when the serving queues fail to meet latency targets; and multi-region deployment calls for replicated object stores and mirrored event logs.

## 5.10 SLM Serving, Quantisation, and Publication Gates

### 5.10.1 Detailed Distillation and Serving Flow The SLM workflow begins with validated teacher trajectories, proceeds through supervised QLoRA and a single DPO alignment stage, and concludes with publication gates. Retained traces include the planning decision, MCP requests, observed tool responses, references to supporting evidence, and the final grounded output. Retrieval uses BGE-M3 because its multilingual, multifunction, and multigranular representation supports a wide range of technical content [49], whereas the answer critic is implemented with DeBERTaV3-based NLI [50].

After adapter optimization, parameters are merged and quantized for local deployment. Activation-aware Weight Quantization (AWQ) is the primary 4-bit release format since it retains the most influential weight channels while reducing memory usage [48]. The signed model package is served with vLLM; its PagedAttention mechanism limits fragmentation of the key-value cache and supports effective continuous batching [47]. If performance degrades beyond the configured benchmark or memory tolerances, publication is halted.

---

<!-- PDF page 53 -->

<!-- Embedded image from source PDF page 53. -->

**Figure 5.4: Distillation, Quantisation, and Gated Model Publication Pipeline**

The training implementation supports 4-bit QLoRA, LoRA rank 16, alpha 32, dropout 0.05, and Phi-specific fused projection targets. The promoted artifact was produced from a split-v2 DPO adapter and quantised with llm-compressor AWQ using W4A16 asymmetric groups of 128. The committed generic adapter summaries concern a separate 500-record run, so epoch, step, and learning-rate claims for the promoted split-v2 adapter are not reported as reproducible until its exact SFT/DPO summaries and datasets are published.

| Gate | Metric | Threshold | Action if not met |
| --- | --- | --- | --- |
| G1 | BFCL function-call F1 | ≥ 0.82 | Expand validated trace data and repeat supervised training |
| G2 | MCP-TaskBench aggregate | ≥ 0.70 | Inspect category failures and add targeted examples |
| G3 | Faithfulness score | ≥ 0.78 | Retune NLI threshold and review DPO pairs |
| G4 | Teacher-to-student task completion gap | ≤ 8 percentage points | Report a bounded null result and per-category gap |
| G5 | Hallucination rate | ≤ 0.12 | Add evidence-focused examples and stronger critic checks |
| G6 | Time to first token | ≤ 1.2 s | Review serving and quantisation configuration |
| G7 | Peak inference VRAM | ≤ 6 GB | Reduce cache budget or revise quantisation settings |

---

<!-- PDF page 54 -->

## 5.11 Drift Operations and Re-index Scheduling

### 5.11.1 Event Topics and Trigger Conditions The executable workers use entity cosine-drift threshold 0.15, community-centroid threshold 0.10, and bounded two-hop re-index requests. The event topics below, 30-day confidence reduction, and 40% full-rebuild trigger are proposed operating policies; they are not wired to a scheduler or sensitivity-test artifact.

| Topic | Partition key | Purpose | Retention |
| --- | --- | --- | --- |
| commit-events | repo id | Repository revision and changed-file set | 7 days |
| entity-changed | _ entity id | Created, updated, or deleted entity with old/new hashes | 14 days |
| drift-detected | _ entity id | Entity or community drift score and threshold | 30 days |
| reindex-requests | _ entity id | Two-hop selective re-index request and reason | 7 days |
| repair-dlq | _ entity id | Idempotent repair for cross-store partial failure | 30 days |

### 5.11.2 Fairness and Operational Limits Redis queues and graph-depth limits bound individual re-index requests. A 30% per-tenant concurrency share and fair scheduler remain design requirements and are not enforced by the current FIFO queue. Critic rounds and traversal depth are bounded; a production A2A timeout is not exercised because cross-framework delegation is not connected.

---

<!-- PDF page 55 -->

# 6. RESULTS AND DISCUSSION

This chapter reports only results present in the committed release evidence. It distinguishes live-model measurements from deterministic harness self-tests and identifies preregistered questions that remain unanswered. Appendix C lists the available machine-readable artifacts and the evidence still required for the comparative claims.

## 6.1 Experimental Setup

The release artifact was evaluated on an NVIDIA L40S through vLLM 0.27.1 with PyTorch 2.13.0+cu129. The committed evidence contains one 89-case held-out tool-call run, one 440-case no-split conformance run, five live repository RAG cases, 20 streaming samples, and the security suite. A separate three-seed bundle uses deterministic oracle predictors solely to test harness replay; it is not model evidence and is excluded from the results below. No executed frontier-teacher or baseline-agent runs, human annotation set, bootstrap sample, or official BFCL result is committed.

## 6.2 RQ1 — MCP-TaskBench Discriminative Power (C1)

Table 6.1 reports the live-model evidence that is currently reproducible. Because comparative agents and human annotations were not executed, Cohen's d and Cohen's κ cannot be calculated and C1 is not yet established.

**Table 6.1: Available MCP-TaskBench Evidence**

| Run | Cases | Split | Tool F1 | Argument accuracy | Claim boundary |
| --- | ---: | --- | ---: | ---: | --- |
| Quantised Phi-4-mini holdout | 89 | prompt-group holdout, seed 17 | 1.00 | 1.00 | Held-out tool-call conformance |
| Quantised Phi-4-mini full suite | 440 | all, includes training cases | 1.00 | 1.00 | Deployment regression/conformance only; not generalisation |

The archived 440-case artifact was generated before protocol success and semantic outcome were separated. Inspection shows that 120 graph results reported `available=false` and 60 entity/conflict results reported `found=false`, despite the legacy aggregate pass rate of 1.00. Consequently, that artifact demonstrates model call formatting and exact argument reproduction, not end-to-end task completion. The corrected harness records `tool_call_pass_rate` and `semantic_outcome_rate` independently and treats unavailable/not-found outcomes as failed semantic results. C1 remains open until multiple real agents and human annotations are evaluated with this corrected harness.

## 6.3 RQ2 — Verified Distillation into Phi-4-mini (C2)

Table 6.2 summarises verified release evidence for the student. Official BFCL, a teacher gap, and peak GPU memory were not measured, so the complete C2 criterion cannot be claimed.

---

<!-- PDF page 56 -->

**Table 6.2: Verified Student Release Measurements**

| Metric | Measured value | Evidence boundary |
| --- | ---: | --- |
| Held-out internal tool F1 | 1.00 (89 cases) | Internal TaskBench, not official BFCL |
| Full internal tool F1 | 1.00 (440 cases) | No-split conformance; training overlap expected |
| Streaming p95 TTFT | 0.0332 s | 20 live vLLM samples |
| Streaming p95 completion latency | 1.3201 s | 20 live vLLM samples |
| AWQ format | W4A16 asymmetric, group 128 | llm-compressor compressed-tensors checkpoint |
| Peak inference VRAM | Not measured | Criterion unresolved |

The release strongly supports local deployment feasibility and internal structured tool-call competence. It does not establish a DPO improvement because the SFT and DPO probes were equal on the 60-case diagnostic run, and no corrected paired full evaluation is committed. The 320/350 admission ratio was produced by an explicit cap after all 350 fixture traces passed schema replay; it must not be interpreted as rejection of 8.6% invalid teacher behavior. C2 is therefore partially supported only for local serving and internal conformance.

## 6.4 RQ3 — Decomposed Multi-Agent versus Monolithic Baseline (C3)

The preregistered C3 requirement is an improvement of at least 5 percentage points in Exact Match or task completion, with a paired interval that excludes zero. The committed code now contains both the fixed-sequence Python baseline and a connected LangGraph–A2A–AutoGen topology, together with an alternating-order paired evaluator and deterministic 10,000-sample bootstrap. No live, oracle-labelled C3 run is committed, so C3 remains untested empirically.

**Table 6.3: C3 Evidence Status**

| Required artifact | Status |
| --- | --- |
| Fixed multi-hop case set with answer oracle | Not committed |
| Monolithic live-agent outputs | Not executed |
| Connected LangGraph–AutoGen/A2A outputs | Implemented and integration-tested; no live evaluation artifact committed |
| Paired bootstrap interval | Not calculable |

The A2A contract tests demonstrate protocol-level interoperability primitives, not a quality advantage from multi-agent decomposition. C3 remains an explicit future experiment.

---

<!-- PDF page 57 -->

## 6.5 Ablation Studies

The implementation exposes some ablation switches, but no live paired ablation bundle is committed. Table 6.4 records the evidence needed before causal contribution claims can be made.

**Table 6.4: Ablation Evidence Status**

| Ablation | Required comparison | Status |
| --- | --- | --- |
| A1 graph retrieval | Same live RAG cases with graph enabled/disabled | Not executed |
| A2 RRF | Same retrieved candidates with RRF/concatenation and relevance oracle | Not executed |
| A3 trace admission | Independently trained raw/admitted checkpoints | Not executed |
| A4 selective re-indexing | Same change workload under selective/full rebuild | Not executed |

These mechanisms are architecturally motivated and unit-tested, but their independent empirical effects are unknown. No ablation conclusion is drawn.

## 6.6 Qualitative Observations

Three observations are justified. First, the quantised model produced exact structured calls on the internal held-out set. Second, the legacy perfect full-suite pass rate concealed unavailable and not-found backend outcomes, confirming that syntax and task completion must be separate metrics. Third, five live RAG answers exceeded the NLI threshold, but the sample is too small and lacks entity-recall oracles. No comparative DPO, decomposition, or ablation effect is inferred.

## 6.7 Discussion of Discrepancies and Threats to Validity

The strongest validity threat is benchmark dependence: internal TaskBench prompts and schemas were used for training, and the full no-split run intentionally overlaps that data. The 89-case prompt-group holdout reduces direct wording leakage but remains generated from the same templates and tool catalog. The model prompt includes user-provided scope identifiers, as required to form arguments, but no longer contains an explicit phrase-to-tool answer map. External BFCL and independently authored tasks are needed for generalisation claims. The five RAG cases use public repositories and transformer NLI but omit expected entity identifiers, so retrieval recall is unknown. Training provenance is incomplete for the promoted split-v2 adapter, and no human annotation, comparative baseline, confidence interval, VRAM measurement, or ablation result is currently available. These are reported as missing evidence rather than silently replaced with assumed values.

---

<!-- PDF page 58 -->

# 7. CONCLUSIONS AND RECOMMENDATIONS

## 7.1 Conclusions

This dissertation delivered a six-pillar reference design and a substantial prototype covering structural parsing, Neo4j/pgvector persistence, hybrid retrieval, an MCP-style tool gateway, A2A SDK contracts, Small Language Model training and AWQ publication, and drift-sensitive re-indexing. It also provides an internal TaskBench harness and cryptographically bound release workflow. The implemented boundaries and unexecuted experiments are stated explicitly.

The measured release supports a narrower conclusion. The quantised Phi-4-mini checkpoint achieved 1.00 tool F1 and argument accuracy on an 89-case internal prompt-group holdout, p95 time to first token of 0.0332 seconds, and p95 completion latency of 1.3201 seconds. Five live repository answers achieved minimum NLI faithfulness of 0.8744 and mean 0.9577, while the security suites passed. These results support internal tool-call competence and local serving feasibility. C1 discrimination, the complete C2 teacher/BFCL/VRAM criteria, and C3 multi-agent superiority remain unproven because their required comparative evidence was not executed.

## 7.2 Recommendations

- Evaluation should report tool-call conformance, protocol execution, semantic backend outcome, and answer quality separately. Official BFCL and independently authored tasks should be added before claiming generalisation. Teacher traces should retain provider/version/request provenance and actual replay outcomes. Live RAG cases should include stable expected entity identifiers so retrieval recall can be measured.

- The implemented LangGraph–AutoGen path and each ablation should be executed as paired live experiments before recommendations about decomposition, RRF contribution, admission filtering, or selective rebuilding are made. Split-v2 SFT/DPO run summaries, datasets, and adapter hashes should be published with the model provenance.

## 7.3 Limitations and Future Scope of Work

The findings are restricted to project-developed TaskBench cases, five live RAG queries over public repositories, synthetic enterprise-like schemas, and one quantised model release. The LangGraph–AutoGen comparison is runnable but has not produced committed live evidence. Future work should commission independent task design and annotation, run official BFCL and multiple agent baselines, execute paired C3 and ablation experiments, add Java and other enterprise languages, curate retrieval-recall labels, measure peak GPU memory, and exercise the signed release in its hardened serving profile. A paper-oriented manuscript should wait until these evidence gaps are closed.

---

<!-- PDF page 59 -->

# Appendix A: Supplementary Pseudocode

This appendix provides the supplementary procedures referenced in Chapter 5. Consistent with the submission requirements, implementation source code is omitted; the following pseudocode specifies the complete behaviour.

Algorithm A.1: A2A delegation envelope construction

procedure DELEGATE_TARGET(subtask, target_agent): card ← discover(target_agent) // A2A agent card assert capabilities(card) ⊇ needs(subtask) token ← mint_scoped_jwt(subtask.tool_scope, ttl=short) env ← { task: subtask, context: provenance(subtask), credentials: token, reply_to: self.endpoint } send_over_mTLS(card.endpoint, env) // target design; not in measured path return await result with timeout; on timeout → compensate()

Algorithm A.2: Conflict entity creation during reconciliation

procedure RECONCILE(claim_a, claim_b): if semantically_equal(claim_a, claim_b): return merge(claim_a, claim_b) c ← new ConflictEntity(subject(claim_a), positions = {claim_a, claim_b}, provenance = {source(claim_a), source(claim_b)}, detected_at = now()) graph.upsert(c); link(c, subject(claim_a)) return c // surfaced to the user with both positions cited

---

<!-- PDF page 60 -->

# Appendix B: Mcp-Taskbench Task Category Examples

For each category, the entries define the intended task, expected gold-standard tool selection, and failures injected to assess error handling. Concrete cases are instantiated from parameterised templates over the fixed corpus.

**Table B.1: MCP-TaskBench Task Category Examples**

| Category | Example intent | Gold tool(s) | Injected faults |
| --- | --- | --- | --- |
| Dependency lookup | Which modules depend on the customer schema? | graph.dependency search _ | Unknown schema name; missing required parameter |
| Impact analysis | If API X changes, which services are affected? | graph.impact paths _ | Cyclic dependency edge; empty result set |
| Semantic search | Find documentation about the billing retry policy | vector.search | Ambiguous query below similarity floor |
| Conflict inspection | Why do two systems define ‘account status’ differently? | conflict.inspect | Conflicting provenance timestamps |
| Multi-hop reconciliation | Trace the field ‘MSISDN’ from schema to API to consuming module | graph.path query + vector.sear_ ch | One hop missing; requires structured partial answer |
| Protocol stress | Any of the above under malformed-response replay | as above | Gateway returns structured error; agent must recover, not hallucinate |

---

<!-- PDF page 61 -->

# Appendix C: Reproducibility Checklist

- The repository corpus is fixed by commit hash, and each experiment logs the corresponding snapshot hashes.

- The internal split seed and release artifacts are stored. The three-seed oracle replay is marked as a harness self-test and is not model evidence. • The live release preserves per-case model outputs, checkpoint Git LFS hashes, runtime versions, evidence-file hashes, and a canonical promotion-record digest. • A durable append-only MCP audit export, hosted-teacher provider records, container digests, explicit protocol revision per task, human annotations, and repeated live runs remain missing from the reproducibility package. • The official A2A SDK now connects LangGraph and AutoGen in code and integration tests, but no live C3 experiment is archived.

---

<!-- PDF page 62 -->

# Appendix D: HLD/LLD Technical Coverage Matrix

Table D.1 documents the alignment review between the companion HLD/LLD and the dissertation. Its role is to confirm coverage of the complete technical path without duplicating every implementation listing contained in the design artefact.

| HLD/LLD area | Report location | Status |
| --- | --- | --- |
| System context and external actors | 4.8.1, Figure 4.10 | Covered |
| Container view and service boundaries | 4.8.2, Figure 4.11 | Covered |
| Query path and bounded critic loop | 4.4, 5.8.2, Figures 4.9 and 5.2 | Covered |
| Event-driven ingestion and logical transaction | 5.2, 5.8.1, Figure 5.1 | Design covered; consumer, object archive, and repair coordinator pending |
| Dual-layer drift and selective re-indexing | 4.3.6, 5.6, 5.11, Figure 4.8 | Covered |
| Deployment topology and training isolation | 4.8.3, Figure 4.12 | Covered |
| Architecture Decision Records | 4.9.1, Table 4.4 | Covered |
| Quality attributes and SLOs | 4.9.2, Table 4.5 | Covered |
| Canonical graph, vector, object, and event ownership | 4.6, 4.10.1, Table 4.6 | Covered |
| MCP envelopes, schemas, error semantics, and audit | 4.10.2, 5.4, Table 4.7 | Covered |
| Knowledge Graph writes, traversal, analytics, and conflicts | 4.3.2, 4.6, 5.2, Appendix A | Covered at report level |
| Planner routing, hybrid retrieval, reranking, and NLI critic | 4.3.3, 5.3, 5.8.2 | Covered |
| MCP tool gateway and tool catalogue | 5.4, Appendix B | Covered at report level |
| A2A bridge, reconciliation, authentication, and tenancy | 4.3.4, 5.8.3, Figure 5.3 | SDK contracts implemented; cross-framework runtime pending |
| Verified trace generation, QLoRA, DPO, quantisation, vLLM | 4.3.5, 5.5, 5.10 | Covered |
| Kafka topics, thresholds, Celery-style orchestration, fairness | 5.11, Table 5.8 | Threshold workers implemented; topic consumers and fairness pending |
| STRIDE security, injection containment, secrets, signed artefacts | 4.5, 5.9.1, Table 5.4 | Covered |
| OpenTelemetry, metrics, and SLO alerting | 5.9.2, Table 5.5 | Covered |
| Evaluation harness and reproducibility manifest | 3.4-3.6, 5.9.3, Appendix C | Live release harness implemented; comparative evidence pending |
| CI/CD environments and capacity migration paths | 5.9.3-5.9.4, Table 5.6 | Static CI implemented; live capacity validation pending |

---

<!-- PDF page 63 -->

# Appendix E: Design Assumptions and Claim Boundaries

The added technical information does not extend the dissertation’s stated claims. The constraints specified in the HLD/LLD remain in effect:

- The reference system is not described as a production-ready platform supporting multi-region operation and high

availability.

- Evaluation of the distilled Small Language Model is limited to the defined tool-use, reconciliation, and multi-

hop synthesis workloads; no claim is made about equivalence to frontier-model general capability.

- MCP-TaskBench continues to function as a specialised, protocol-aware instrument that complements rather than

substitutes broader benchmarks for function calling and agents.

- Cross-framework A2A interoperability is not yet demonstrated; the prototype currently validates SDK-level cards, messages, and task execution.

- The primary corpus comprises open-source Python and JavaScript projects and synthetic enterprise-style

schemas; no proprietary organisational repository is included.

- Chapter 6 reports measured release outcomes and explicitly marks C1, the complete C2 criteria, C3, and the ablation studies as unresolved. Service-

Level Objectives that are not measured in these experiments remain targets for future design.

| ID | Risk | Response |
| --- | --- | --- |
| R1 | Student remains more than the target gap behind the teacher | Report the null result with per-category analysis and preserve the local-deployment findings |
| R2 | Teacher API cost or availability disrupts trace generation | Cap traces, deduplicate tasks, and preserve a secondary teacher option |
| R3 | MCP or A2A protocol revision changes | Pin versions in each run manifest and isolate protocol adapters |
| R4 | Benchmark task design is insufficiently discriminative | Pilot with reviewers, version the benchmark, and release task-level scores |
| R5 | Prompt-injection control misses a new attack pattern | Layer controls, add adversarial tasks, and document residual risk |
| R6 | Operational claims exceed measured evidence | Move unmeasured properties to design targets or future work |

---

<!-- PDF page 64 -->

# Appendix F: Pillar information

<!-- Embedded image from source PDF page 64. -->

**Figure F.1: IDRKD High-Level System Architecture (Six-Pillar Design)**

---

<!-- PDF page 65 -->

<!-- Embedded image from source PDF page 65. -->

**Figure F.2: Pillar 1 — Structural Ingestion**

---

<!-- PDF page 66 -->

<!-- Embedded image from source PDF page 66. -->

**Figure F.3: Pillar 2 — Knowledge Graph**

---

<!-- PDF page 67 -->

<!-- Embedded image from source PDF page 67. -->

**Figure F.4: Pillar 3 — Agentic RAG Pipeline**

---

<!-- PDF page 68 -->

<!-- Embedded image from source PDF page 68. -->

**Figure F.5: Pillar 4 — MCP and A2A Orchestration**

---

<!-- PDF page 69 -->

<!-- Embedded image from source PDF page 69. -->

**Figure F.6: Pillar 5 — SLM Distillation and Serving**

---

<!-- PDF page 70 -->

<!-- Embedded image from source PDF page 70. -->

**Figure F.7: Pillar 6 — Drift Detection and Re-indexing**

---

<!-- PDF page 71 -->

# References

[1] S. G. Patil et al., “The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models,” Proceedings of Machine Learning Research, vol. 267, pp. 48371–48392, 2025. [Online]. Available: https://proceedings.mlr.press/v267/patil25a.html. [Accessed: 30-Jul-2026].

[2] Y. Qin et al., “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs,” arXiv preprint arXiv:2307.16789, 2023. [Online]. Available: https://arxiv.org/abs/2307.16789. [Accessed: 30-Jul-2026].

[3] Model Context Protocol, “Basic Protocol Specification — Messages,” 26 Mar. 2025. [Online]. Available: https://modelcontextprotocol.io/specification/2025-03-26/basic. [Accessed: 30-Jul-2026].

[4] V. Srinivasan, “Bridging Protocol and Production: Design Patterns for Deploying AI Agents with Model Context Protocol,” arXiv preprint arXiv:2603.13417, 2026. [Online]. Available: https://arxiv.org/abs/2603.13417. [Accessed: 30-Jul-2026].

[5] D. Zhang et al., “MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents,” arXiv preprint arXiv:2510.15994, 2025. [Online]. Available: https://arxiv.org/abs/2510.15994. [Accessed: 30-Jul-2026].

[6] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient Finetuning of Quantized LLMs,” arXiv preprint arXiv:2305.14314, 2023. [Online]. Available: https://arxiv.org/abs/2305.14314. [Accessed: 30-Jul-2026].

[7] R. Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” arXiv preprint arXiv:2305.18290, 2023. [Online]. Available: https://arxiv.org/abs/2305.18290. [Accessed: 30-Jul-2026].

[8] Microsoft Research, “Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs,” arXiv preprint arXiv:2503.01743, 2025. [Online]. Available: https://arxiv.org/abs/2503.01743. [Accessed: 30-Jul-2026].

[9] M. Kang, J. Jeong, S. Lee, J. Cho, and S. J. Hwang, “Distilling LLM Agent into Small Models with Retrieval and Code Tools,” arXiv preprint arXiv:2505.17612, 2025. [Online]. Available: https://arxiv.org/abs/2505.17612. [Accessed: 30-Jul-2026].

[10] Q. Zhong et al., “SOD: Step-wise On-policy Distillation for Small Language Model Agents,” arXiv preprint arXiv:2605.07725, 2026. [Online]. Available: https://arxiv.org/abs/2605.07725. [Accessed: 30-Jul-2026].

[11] Y. Ji, Z. Li, H. Ji, and D. He, “StepGap: A Hybrid NLI-LLM Checker for Step-Level Evidence-Gap Detection in Multi-Hop Question Answering,” arXiv preprint arXiv:2605.24733, 2026. [Online]. Available: https://arxiv.org/abs/2605.24733. [Accessed: 30-Jul-2026].

[12] Y. B. Alebachew, H. Leary, S. Vaishampayan, and C. Brown, “Beyond Code Snippets: Benchmarking LLMs on Repository-Level Question Answering,” arXiv preprint arXiv:2603.26567, 2026. [Online]. Available: https://arxiv.org/abs/2603.26567. [Accessed: 30- Jul-2026].

[13] F. Zhang et al., “EraRAG: Efficient and Incremental Retrieval Augmented Generation for Growing Corpora,” arXiv preprint arXiv:2506.20963, 2025. [Online]. Available: https://arxiv.org/abs/2506.20963. [Accessed: 30-Jul-2026].

[14] K. H. Lau et al., “Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation,” arXiv preprint arXiv:2602.01965, 2026. [Online]. Available: https://arxiv.org/abs/2602.01965. [Accessed: 30-Jul-2026].

[15] A. Ehtesham, A. Singh, G. K. Gupta, and S. Kumar, “A Survey of Agent Interoperability Protocols: Model Context Protocol, Agent Communication Protocol, Agent-to-Agent Protocol, and Agent Network Protocol,” arXiv preprint arXiv:2505.02279, 2025. [Online]. Available: https://arxiv.org/abs/2505.02279. [Accessed: 30-Jul-2026].

[16] The Linux Foundation, “A2A Protocol Surpasses 150 Organizations, Lands in Major Cloud Platforms, and Sees Enterprise Production Use in First Year,” 9 Apr. 2026. [Online]. Available: https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year. [Accessed: 30-Jul-2026].

[17] A. Vaswani et al., “Attention Is All You Need,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 5998–6008. [Online]. Available: https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa- Abstract.html. [Accessed: 30-Jul-2026].

[18] T. B. Brown et al., “Language Models are Few-Shot Learners,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 1877–1901. [Online]. Available: https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a- Abstract.html. [Accessed: 30-Jul-2026].

[19] P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 9459–9474. [Online]. Available: https://papers.nips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html. [Accessed: 30-Jul-2026].

---

<!-- PDF page 72 -->

[20] V. Karpukhin et al., “Dense Passage Retrieval for Open-Domain Question Answering,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 6769–6781. [Online]. Available: https://aclanthology.org/2020.emnlp-main.550/. [Accessed: 30-Jul-2026].

[21] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019, pp. 3982–3992.

[22] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, “Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods,” in Proceedings of the 32nd ACM SIGIR Conference, 2009, pp. 758–759.

[23] Y. A. Malkov and D. A. Yashunin, “Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824–836, 2020.

[24] D. Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” arXiv preprint arXiv:2404.16130, 2024. [Online]. Available: https://arxiv.org/abs/2404.16130. [Accessed: 30-Jul-2026].

[25] Z. Yang et al., “HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018, pp. 2369–2380.

[26] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, “Fast Unfolding of Communities in Large Networks,” Journal of Statistical Mechanics: Theory and Experiment, vol. 2008, no. 10, P10008, 2008.

[27] L. Page, S. Brin, R. Motwani, and T. Winograd, “The PageRank Citation Ranking: Bringing Order to the Web,” Stanford InfoLab Technical Report 1999-66, 1999.

[28] S. Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2023.

[29] T. Schick et al., “Toolformer: Language Models Can Teach Themselves to Use Tools,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[30] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, “Gorilla: Large Language Model Connected with Massive APIs,” arXiv preprint arXiv:2305.15334, 2023. [Online]. Available: https://arxiv.org/abs/2305.15334. [Accessed: 30-Jul-2026].

[31] J. Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022, pp. 24824–24837.

[32] N. Shinn et al., “Reflexion: Language Agents with Verbal Reinforcement Learning,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[33] Q. Wu et al., “AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation,” arXiv preprint arXiv:2308.08155, 2023. [Online]. Available: https://arxiv.org/abs/2308.08155. [Accessed: 30-Jul-2026].

[34] G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv preprint arXiv:1503.02531, 2015. [Online]. Available: https://arxiv.org/abs/1503.02531. [Accessed: 30-Jul-2026].

[35] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[36] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter,” arXiv preprint arXiv:1910.01108, 2019. [Online]. Available: https://arxiv.org/abs/1910.01108. [Accessed: 30-Jul-2026].

[37] Z. Feng et al., “CodeBERT: A Pre-Trained Model for Programming and Natural Languages,” in Findings of the Association for Computational Linguistics: EMNLP 2020, 2020, pp. 1536–1547.

[38] D. Guo et al., “GraphCodeBERT: Pre-training Code Representations with Data Flow,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[39] J. Kreps, N. Narkhede, and J. Rao, “Kafka: A Distributed Messaging System for Log Processing,” in Proceedings of the NetDB Workshop, 2011.

[40] J. Cohen, “A Coefficient of Agreement for Nominal Scales,” Educational and Psychological Measurement, vol. 20, no. 1, pp. 37–46, 1960.

[41] J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988.

[42] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of NAACL-HLT 2019, 2019, pp. 4171–4186.

---

<!-- PDF page 73 -->

[43] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAS: Automated Evaluation of Retrieval Augmented Generation,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL): System Demonstrations, 2024, pp. 150–158.

[44] H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models,” arXiv preprint arXiv:2302.13971, 2023. [Online]. Available: https://arxiv.org/abs/2302.13971. [Accessed: 30-Jul-2026].

[45] S. Brown, “The C4 model for visualising software architecture,” C4 Model. [Online]. Available: https://c4model.com/. [Accessed: 29- Jul-2026].

[46] M. Nygard, “Documenting Architecture Decisions,” Cognitect, 15 Nov. 2011. [Online]. Available: https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions. [Accessed: 29-Jul-2026].

[47] W. Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” arXiv:2309.06180, 2023. [Online]. Available: https://arxiv.org/abs/2309.06180. [Accessed: 30-Jul-2026].

[48] J. Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration,” arXiv:2306.00978, 2023. [Online]. Available: https://arxiv.org/abs/2306.00978. [Accessed: 30-Jul-2026].

[49] J. Chen et al., “M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation,” arXiv:2402.03216, 2024. [Online]. Available: https://arxiv.org/abs/2402.03216. [Accessed: 30-Jul-2026].

[50] P. He, J. Gao, and W. Chen, “DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing,” arXiv:2111.09543, 2021. [Online]. Available: https://arxiv.org/abs/2111.09543. [Accessed: 30-Jul-2026].

[51] OpenTelemetry Authors, “OpenTelemetry Specification,” 2026. [Online]. Available: https://opentelemetry.io/docs/specs/otel/. [Accessed: 29-Jul-2026].

[52] Microsoft, “Microsoft Threat Modeling Tool threats: STRIDE model,” 25 Aug. 2022. [Online]. Available: https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats. [Accessed: 29-Jul-2026].

[53] M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy, “SpanBERT: Improving Pre-training by Representing and Predicting Spans,” Transactions of the Association for Computational Linguistics, vol. 8, pp. 64–77, 2020, doi: 10.1162/tacl_a_00300.

[54] L. Lamport, “Time, Clocks, and the Ordering of Events in a Distributed System,” Communications of the ACM, vol. 21, no. 7, pp. 558– 565, Jul. 1978, doi: 10.1145/359545.359563.

[55] Tree-sitter, "Introduction," [Online]. Available: https://tree-sitter.github.io/. [Accessed: 30-Jul-2026].

[56] Z. Wang et al., "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers," arXiv preprint arXiv:2508.20453, 2025. [Online]. Available: https://arxiv.org/abs/2508.20453. [Accessed: 30-Jul-2026].

[57] C. Bandi et al., "MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers," arXiv preprint arXiv:2602.00933, 2026. [Online]. Available: https://arxiv.org/abs/2602.00933. [Accessed: 30-Jul-2026].

[58] Z. Ma, J. Liu, X. Luo, Z. Huang, Q. Zhu, and W. Che, "Advancing Tool-Augmented Large Language Models via Meta-Verification and Reflection Learning," arXiv preprint arXiv:2506.04625, 2025. [Online]. Available: https://arxiv.org/abs/2506.04625. [Accessed: 30- Jul-2026].

[59] A. Q. Jiang et al., "Mistral 7B," arXiv preprint arXiv:2310.06825, 2023. [Online]. Available: https://arxiv.org/abs/2310.06825. [Accessed: 30-Jul-2026]

---

<!-- PDF page 74 -->

# Glossary

This glossary expands the specialised terminology and abbreviations appearing in the report and identifies the section in which each is first introduced.

**Table G.1: Glossary of Terms and Abbreviations**

| Term | Meaning | First use |
| --- | --- | --- |
| A2A | Agent-to-Agent Protocol — an open protocol for communication and delegation between agents across frameworks | 1.2 |
| ADR | Architecture Decision Record - a short record of an architectural decision, its context, and consequences | 4.9.1 |
| API | Application Programming Interface | 1.1 |
| APScheduler | Python scheduling library used for periodic graph analytics and maintenance jobs | 4.3.2 |
| AWQ | Activation-aware Weight Quantization | 5.10.1 |
| BFCL | Berkeley Function Calling Leaderboard — a benchmark for semantic function-calling accuracy | 2.4 |
| BGE-M3 | Multi-lingual embedding model | 5.10.1 |
| CI | Confidence Interval - a range expressing uncertainty around an estimated metric | 3.4.3 |
| CI/CD | Continuous Integration and Continuous Delivery | 5.9.3 |
| Cohen's d | A standardised effect-size measure for the difference between two means | 3.3 |
| Cohen's κ (kappa) | A chance-corrected coefficient of inter-annotator agreement | 3.3 |
| CPU | Central Processing Unit | 4.7 |
| Critic loop | A bounded verification stage that checks each answer claim against retrieved evidence | 4.3.3 |
| DeBERTaV3 | Transformer encoder used for Natural Language Inference-based claim verification | 4.3.3 |
| DPO | Direct Preference Optimisation — preference alignment without a separate reward model | 1.4 |
| Drift (semantic) | Divergence between stored representations and the current state of the underlying source | 2.3 |
| EM | Exact Match — a strict answer-equality metric | 3.4.3 |
| GPU | Graphics Processing Unit | 2.6 |
| GraphRAG | Graph-based Retrieval-Augmented Generation | 2.3 |
| HNSW | Hierarchical Navigable Small World — an approximate nearest-neighbour index structure | 2.2 |
| IDRKD | Intelligent Data Reconciliation and Knowledge Discovery — the system presented in this report | 1.1 |
| JSON-RPC | JavaScript Object Notation Remote Procedure Call — the message protocol underlying MCP | 1.2 |
| JWT | JSON Web Token — a signed token used for scoped authorisation | 4.3.4 |
| Kafka | Distributed event-streaming platform used for ingestion, drift, re-indexing, and repair events | 2 |
| KG | Knowledge Graph | 1.4 |
| LangGraph | State-machine orchestration framework used to coordinate deterministic agent workflows | 1 |
| LLM | Large Language Model | 1.2 |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning through low-rank weight updates | 2.6 |
| MCP | Model Context Protocol — an open standard connecting agents to tools and data over JSON-RPC 2.0 | 1.2 |
| MCP-TaskBench | The benchmark contributed by this dissertation, scoring semantic tool selection and protocol conformance jointly | 1.4 |
| mTLS | Mutual Transport Layer Security — two-way authenticated encrypted transport | 4.3.4 |

---

<!-- PDF page 75 -->

| Term | Meaning | First use |
| --- | --- | --- |
| MVP | Minimum Viable Product | 1.5 |
| Neo4j | Graph database used to store entities, relationships, conflicts, and graph analytics | 4.1 |
| NER | Named Entity Recognition | 5.8.1 |
| NLI | Natural Language Inference | 5.8.2 |
| OpenTelemetry | Vendor-neutral framework for distributed traces, metrics, and logs | 5.9.2 |
| pgvector | The Postgres vector extension used for semantic retrieval | 2.2 |
| Postgres | Relational database used for vector metadata, embeddings, and MCP audit records | 4.2 |
| QLoRA | Quantised Low-Rank Adaptation — LoRA fine-tuning over quantised base weights | 1.4 |
| RAG | Retrieval-Augmented Generation | 1.1 |
| Redis | In-memory data store used for caches, work queues, and re-index scheduling | 4.3.6 |
| RRF | Reciprocal Rank Fusion — a rank-based method for fusing candidate lists from multiple retrievers | 2.2 |
| SLM | Small Language Model | 1.1 |
| SLO | Service-Level Objective - a measurable target for system reliability or performance | 4.9.2 |
| SpanBERT | Span-based BERT variant for NER | 5.8.1 |
| Static Graph Fallacy | The incorrect assumption that a knowledge graph, once built, remains valid as its sources change | 2.3 |
| Tree-sitter | An incremental parsing framework producing concrete syntax trees for many languages | 1.4 (via P1) |
| UI | User Interface | 4.7 |
| vLLM | High-throughput LLM serving system | 5.10.1 |
| VRAM | Video Random Access Memory — the memory of a GPU | 3.2 |
