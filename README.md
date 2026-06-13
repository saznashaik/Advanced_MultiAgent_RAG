# 🤖 Advanced Multi-Agent RAG System

> **A production-grade Retrieval-Augmented Generation pipeline built on LangGraph — with hybrid retrieval, self-correcting hallucination loops, LLM guardrails, a multi-model gateway, and a full 7-metric evaluation suite.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Pipeline Walkthrough — 5 Nodes](#-pipeline-walkthrough--5-nodes)
- [Advanced Retrieval Engine](#-advanced-retrieval-engine)
- [Latency Optimisation](#-latency-optimisation)
- [LLM Guardrails](#-llm-guardrails)
- [LLM Gateway (LiteLLM)](#-llm-gateway-litelllm)
- [Evaluation Suite](#-evaluation-suite)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Running the System](#-running-the-system)
- [Sample Output](#-sample-output)
- [Design Decisions & Engineering Notes](#-design-decisions--engineering-notes)
- [Known Limitations](#-known-limitations)

---

## 🧠 Overview

Most RAG tutorials stop at "embed → retrieve → answer." This project goes further — it treats the full question-answering pipeline as a **multi-agent graph** where each node is a specialised agent with a single responsibility, and the graph self-corrects when quality drops.

The knowledge base is seeded from three Lilian Weng research blogs covering:
- **LLM-powered autonomous agents** (planning, memory, tool use)
- **Prompt engineering** techniques (few-shot, CoT, zero-shot, etc.)
- **Adversarial attacks on LLMs** (token manipulation, jailbreaks, red-teaming)

When you ask a question, the system doesn't just retrieve and dump text into a prompt. It routes intelligently, grades retrieved documents for relevance, generates a grounded answer, checks that answer for hallucinations, and **retries the entire retrieval loop** if the answer drifts from the source material — all orchestrated by a compiled LangGraph state machine.

---

## 🏗 System Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              USER QUESTION                  │
                         └──────────────────┬──────────────────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │      INPUT GUARDRAILS       │
                              │  • Length check (3–600 ch)  │
                              │  • Prompt injection detect  │
                              │  • Topic restriction filter │
                              └─────────────┬───────────────┘
                                            │
                              ┌─────────────▼───────────────┐
                              │       NODE 1: ROUTER        │
                              │  LLM decides: retrieve or   │
                              │  answer from general know.  │
                              └──────┬──────────────┬────────┘
                                     │              │
                               needs_retrieval   direct answer
                                     │              │
                         ┌───────────▼──────┐       │
                         │  NODE 2: RETRIEVER│       │
                         │  Parallel hybrid  │       │
                         │  BM25 + vector    │       │
                         │  Multi-query exp. │       │
                         └───────────┬───────┘       │
                                     │               │
                         ┌───────────▼───────┐       │
                         │  NODE 3: GRADER   │       │
                         │  Batch LLM grades │       │
                         │  all docs at once │       │
                         │  useful/not_useful│       │
                         └───────────┬───────┘       │
                                     │               │
                         ┌───────────▼───────────────▼──┐
                         │      NODE 4: ANSWER GENERATOR │
                         │  Grounded mode (with docs)    │
                         │  Fallback mode (no docs)      │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │   NODE 5: HALLUCINATION CHECK  │
                         │  Structured LLM judge          │
                         │  grounded=true → END           │
                         │  grounded=false → retry ───────┼──→ NODE 2
                         │  (max 2 retries, then END)     │
                         └───────────────┬────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   OUTPUT GUARDRAILS  │
                              │  • PII filter        │
                              │  • Empty-resp. check │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     FINAL ANSWER     │
                              └─────────────────────┘
```

The graph is compiled with `StateGraph` from LangGraph. Every node reads from and writes to a shared `GraphState` TypedDict — making state transitions explicit, traceable, and easy to debug.

---

## 🔁 Pipeline Walkthrough — 5 Nodes

### Node 1 · Query Router

The router is an LLM-powered decision gate. Given the incoming question, it returns a structured `RouteDecision` with `needs_retrieval: bool`.

```python
class RouteDecision(TypedDict):
    needs_retrieval: Annotated[bool, ..., "True if retrieval is needed"]
```

The system prompt is deliberately biased toward retrieval — *"when in doubt, retrieve"* — because the cost of a missed retrieval (empty answer) is worse than the cost of an unnecessary one (slightly slower). If the LLM call itself errors, the node safely defaults to `needs_retrieval=True`.

Questions clearly outside the knowledge domain (cooking, sports, weather) are routed directly to the answer node, which generates a `[General knowledge]`-labelled response from the model's parametric memory.

---

### Node 2 · Retriever

The retriever runs the full `advanced_retrieve` pipeline (described in detail below). On the **first attempt**, it retrieves for the original query alone. On **subsequent retries** (triggered by hallucination), it automatically expands to three parallel queries:

```python
queries = [q] if retry == 0 else [q, f"detailed explanation: {q}", f"examples of: {q}"]
```

This retry escalation is a key design decision: being aggressive with query expansion only when you need to, rather than paying that cost on every request.

Results are parallelised via `ThreadPoolExecutor` and deduplicated by MD5 hash of the first 150 characters of each chunk.

---

### Node 3 · Document Grader

The grader evaluates all retrieved documents **in a single batched LLM call** rather than one call per document. This was an explicit engineering choice — N individual calls would add N × latency to every request.

The prompt is intentionally generous: `grade='useful'` if **any** document is relevant; `grade='not_useful'` only if all are completely unrelated. Partial relevance counts. Over-filtering is a worse failure mode than under-filtering here, because the hallucination checker downstream acts as a second quality gate.

```python
class GradeDecision(TypedDict):
    grade: Annotated[Literal["useful", "not_useful"], ..., "Document relevance grade"]
```

---

### Node 4 · Answer Generator

The answer node operates in two modes depending on document availability:

**Grounded mode** (documents retrieved): The LLM is instructed to synthesise across documents and stick closely to the source material. The instruction was deliberately relaxed from "ONLY from documents" to "primarily from documents" after discovering that the strict version caused the hallucination checker to flag legitimately good synthesised answers.

**Fallback mode** (no documents): Rather than returning a useless "No documents retrieved" message, the node generates a helpful, concise answer from the LLM's parametric knowledge, clearly prefixed with `[General knowledge]` so the user understands the source.

Both modes route through the `gateway_invoke` function, ensuring the same fallback model logic applies.

---

### Node 5 · Hallucination Checker

The hallucination checker is the system's self-healing mechanism. It receives the retrieved facts and the generated answer, and returns a structured verdict:

```python
class HallucinationDecision(TypedDict):
    grounded: Annotated[bool, ..., "True if answer is grounded in documents"]
```

**Critical implementation detail:** `retry_count` is incremented **inside this node**, not in the conditional edge function. LangGraph conditional edge functions are read-only routing logic — any state mutations written there are silently dropped by the framework. This was a real bug in the original implementation that caused an infinite retry loop.

The routing logic after this node:
- `grounded=True` → `END` (return the answer)
- `grounded=False` and `retry_count <= MAX_RETRIES` → back to `NODE 2` (re-retrieve)
- `grounded=False` and retries exhausted → `END` (return best available answer)

**Edge case handled:** When `documents=[]` (fallback general-knowledge answers), the hallucination checker will always return `grounded=False` because it's comparing a non-empty answer against empty facts. The `rag_bot` function detects this case and overrides the hallucination status to `"grounded (general knowledge)"`, preventing wasteful retries.

---

## 🔍 Advanced Retrieval Engine

The retrieval pipeline is a three-layer system designed to maximise recall while keeping precision high.

### Layer 1 — Hybrid BM25 + Vector Search

Neither keyword search nor semantic search alone is sufficient:

- **BM25** excels at exact-match recall — if the user asks about "ReAct" or "chain-of-thought," BM25 will surface chunks containing those exact tokens even if the embedding similarity is mediocre.
- **Vector search** (OpenAI `text-embedding-3-small`, via Astra DB) handles paraphrastic and semantic queries — "how does an LLM plan ahead" surfaces agent-planning content even without the word "planning."

Results are **interleaved** (alternating between the two ranked lists) rather than concatenated. Interleaving ensures neither source dominates, and MD5-based deduplication removes any chunks appearing in both lists.

```python
def hybrid_retrieve(query: str, k: int = 6) -> List[Document]:
    vector_docs = base_retriever.invoke(query)
    bm25_docs   = bm25_retrieve(query, k=k)
    merged = []
    for pair in zip(vector_docs, bm25_docs):
        merged.extend(pair)
    # append remainder from the longer list
    ...
    return _dedup(merged)[:k]
```

### Layer 2 — Multi-Query Expansion

A single query phrasing may not surface the best chunks. The expansion chain uses GPT-4o-mini to generate **3 alternative phrasings** of the original question:

```
Original: "What are adversarial attacks on LLMs?"
Variant 1: "How are large language models attacked or manipulated?"
Variant 2: "Techniques used to exploit vulnerabilities in LLMs"
Variant 3: "Methods for bypassing safety in language models"
```

All four queries (original + 3 variants) are run through hybrid retrieval and the results are merged and deduplicated. This meaningfully improves recall for paraphrase-sensitive topics.

### Layer 3 — Contextual Compression: A Deliberate Omission

Contextual compression (via `LLMChainExtractor`) was evaluated and **intentionally removed**. In practice, `LLMChainExtractor` frequently returns empty strings for factual, definition-heavy chunks — exactly the kind of content this knowledge base contains. The document grader node handles relevance filtering instead, and it's a much more reliable signal.

---

## ⚡ Latency Optimisation

Three caching layers prevent redundant work:

| Layer | What's cached | Key |
|---|---|---|
| Embedding cache | OpenAI embedding vectors | MD5 of input text |
| Query-result cache | Full retrieval results | MD5 of `query:k` |
| Async retrieval | Non-blocking thread pool | — |

```python
# Cached call is near-instant on repeat questions
First call:  ~2.3s
Cached call: ~0.001s
```

Async retrieval wraps `advanced_retrieve` in `loop.run_in_executor`, ensuring the retrieval I/O doesn't block the event loop. The `parallel_retrieve` function in the graph node takes this further — multiple query variants are submitted concurrently to the thread pool via `ThreadPoolExecutor(max_workers=4)`.

---

## 🛡 LLM Guardrails

### Input Guardrails

Three checks run in sequence before the question reaches the graph — all regex-based, zero-latency, no LLM calls:

**1. Length validation**
```
Min: 3 characters  |  Max: 600 characters
```

**2. Prompt injection detection**

Regex patterns catch common injection attempts:
```
"ignore all previous instructions"
"pretend to be"
"you are now"
"DAN mode"
"override safety guidelines"
```

**3. Topic restriction — Two-tier matching**

Rather than a simple keyword list, topic matching uses two tiers:

- **Tier 1 (exact):** Substring match against a broad keyword list covering agents, prompting, adversarial attacks, LLMs, RAG, fine-tuning, and related research vocabulary.
- **Tier 2 (stem):** Prefix matching against stems like `"agent"`, `"prompt"`, `"retriev"`, `"hallucinat"` — this catches morphological variants like `"agentic"`, `"prompting"`, `"retrieval"` that wouldn't match the exact list.

This two-tier approach was added specifically to fix queries like *"tell about agentic system"* that were previously being blocked by an overly rigid exact-match check.

### Output Guardrails

Two post-generation checks:

- **PII filter:** Regex patterns catch SSNs (`\d{3}-\d{2}-\d{4}`), 16-digit card numbers, and email addresses.
- **Empty response check:** Answers shorter than 5 characters are flagged as invalid.

---

## 🌐 LLM Gateway (LiteLLM)

All LLM calls in the pipeline route through a centralised `gateway_invoke` function rather than calling OpenAI directly. This provides:

### Automatic Model Fallback

```python
GATEWAY_MODELS = [
    "gpt-4o-mini",    # primary — fast and cheap
    "gpt-3.5-turbo",  # first fallback
    "gpt-4o",         # premium fallback — if all else fails
]
```

The gateway tries models in order. A `RateLimitError` triggers an exponential backoff retry (up to 2× per model). An `APIConnectionError` immediately moves to the next model.

### Token Usage Tracking

Every non-streaming call logs prompt tokens, completion tokens, and total tokens:

```python
usage_log.append({
    "model":      model,
    "prompt_tok": usage.prompt_tokens,
    "compl_tok":  usage.completion_tokens,
    "total_tok":  usage.total_tokens,
})
```

Call `print_usage_summary()` at any point to see a full cost accounting across the session.

### Streaming Support

Pass `stream=True` to `gateway_invoke` to receive token-by-token output. The function collects and returns the full text while printing each delta in real time.

---

## 📊 Evaluation Suite

The evaluation suite is designed to answer one question most teams skip: *"How do I know if my RAG system is actually good?"*

### LLM-as-Judge Metrics (4 dimensions)

All four judges use `with_structured_output` with TypedDict schemas to enforce parseable, boolean verdicts with mandatory step-by-step reasoning explanations.

| Metric | What it measures | Judge model |
|---|---|---|
| **Correctness** | Factual accuracy vs. ground truth | GPT-4o-mini |
| **Relevance** | Does the answer address the question? | GPT-4o |
| **Groundedness** | Is the answer supported by retrieved docs? | GPT-4o |
| **Retrieval Relevance** | Are retrieved docs related to the question? | GPT-4o |

### Traditional String Metrics (3 dimensions)

| Metric | What it measures |
|---|---|
| **BLEU** | N-gram overlap with reference answer |
| **ROUGE-L** | Longest common subsequence with reference |
| **Perplexity** | GPT-2 language model score — fluency proxy |

### Running the Evaluation

The full matrix runs via **LangSmith's `evaluate()`** API against a curated 3-example dataset (`"Production RAG"`):

```python
experiment_results = ls_client.evaluate(
    target_runner,
    data="Production RAG",
    evaluators=[correctness, groundedness, relevance, retrieval_relevance, traditional_string_metrics],
    experiment_prefix="advanced-multiagent-rag-eval",
)
df = experiment_results.to_pandas()
```

Results are printed as a pandas DataFrame with all 7 metric columns.

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| **Orchestration** | LangGraph (StateGraph) |
| **LLM** | OpenAI GPT-4o-mini (primary), GPT-3.5-turbo, GPT-4o (fallbacks) |
| **LLM Gateway** | LiteLLM |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | Astra DB (DataStax) |
| **Keyword Search** | BM25Okapi (rank-bm25) |
| **LLM Framework** | LangChain |
| **Tracing & Eval** | LangSmith |
| **Traditional Metrics** | HuggingFace `evaluate` (BLEU, ROUGE-L), GPT-2 perplexity |
| **Data Loading** | LangChain WebBaseLoader |
| **Text Splitting** | RecursiveCharacterTextSplitter (tiktoken encoder) |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
Advanced_MultiAgent_RAG/
│
├── Advanced_MultiAgent_RAG.ipynb   # Main notebook — all cells in execution order
│
├── .env                            # API keys (never commit this)
├── requirements.txt                # All dependencies
└── README.md                       # This file
```

The entire system lives in the notebook, structured in numbered sections:

| Section | Content |
|---|---|
| **Cell 1** | Package installation |
| **Cell 2** | Environment & credentials |
| **Cell 3** | Astra DB connection test |
| **Cell 4** | Document ingestion (3 URLs → Astra DB) |
| **Cell 5** | Hybrid retriever (BM25 + vector + multi-query) |
| **Cell 6** | Latency layer (async, embedding cache, query cache) |
| **Cell 7** | LLM Guardrails (input + output validation) |
| **Cell 8** | LLM Gateway via LiteLLM |
| **Cell 9** | LangGraph — 5-node state machine |
| **Cell 10** | `rag_bot` — full pipeline entry point |
| **Cell 11** | Interactive Q&A loop |
| **Cells 12–15** | Full evaluation suite |
| **Cell 16** | Token usage summary |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [DataStax Astra DB](https://astra.datastax.com/) account (free tier works)
- A [LangSmith](https://smith.langchain.com/) account (free tier works)

### Install dependencies

```bash
pip install langchain langchain-openai langchain-community langchain-astradb \
            langgraph rank-bm25 litellm guardrails-ai cassio astrapy \
            datasets evaluate nltk rouge-score langsmith \
            transformers torch python-dotenv tiktoken cohere
```

Or install from the notebook's first cell directly by running it.

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true

ASTRA_DB_API_ENDPOINT=https://<your-db-id>-<region>.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_KEYSPACE=default_keyspace
```

> ⚠️ **Never commit real credentials to version control.** The notebook contains hardcoded keys for demonstration — replace them with environment variable reads via `os.getenv()` before sharing.

---

## 🚀 Running the System

### 1. Ingest documents (run once)

Run **Cells 1–4** to install packages, load environment variables, connect to Astra DB, and ingest the three knowledge base URLs. Once ingested, the vector store persists — you don't need to re-run Cell 4 on subsequent sessions.

### 2. Build the pipeline (run each session)

Run **Cells 5–10** in order to initialise all components: retriever, latency cache, guardrails, gateway, LangGraph, and `rag_bot`.

### 3. Interactive mode

Run **Cell 11** to start the interactive Q&A loop:

```
--- Multi-Agent RAG ready (type 'exit' to stop) ---
Topics: AI agents, prompting, adversarial attacks, LLM research

Ask a question: What are five types of adversarial attacks on LLMs?
```

### 4. Run evaluation

Run **Cells 12–15** to execute the full evaluation matrix against the LangSmith dataset. Results appear as a DataFrame.

---

## 🖥 Sample Output

```
Ask a question: tell about prompt engineering

  [rag_bot] 7.37s | hallucination=grounded | retries=0 | docs=5

[ANSWER]       : Prompt Engineering, or In-Context Prompting, involves methods to communicate
                 with LLMs to achieve desired outcomes without altering model weights. It is
                 an empirical science requiring experimentation, as effectiveness varies
                 significantly across models. The primary focus is on alignment and model
                 steerability for autoregressive language models.
[DOCS USED]    : 5
[SOURCE]       : RAG documents
[HALLUCINATION]: grounded
[RETRIES]      : 0
```

```
Ask a question: what are the types of machine learning?

  [rag_bot] 1.9s | hallucination=grounded (general knowledge) | retries=0 | docs=0

[ANSWER]       : [General knowledge] Machine learning is broadly categorised into three types:
                 supervised learning (labelled data, prediction tasks), unsupervised learning
                 (unlabelled data, pattern discovery), and reinforcement learning (agent learns
                 via reward signals). Semi-supervised and self-supervised learning are additional
                 paradigms gaining traction in modern deep learning.
[DOCS USED]    : 0
[SOURCE]       : General knowledge (no docs retrieved)
[HALLUCINATION]: grounded (general knowledge)
[RETRIES]      : 0
```

---

## 🧩 Design Decisions & Engineering Notes

**Why LangGraph over a simple chain?**
LangGraph makes the retry loop explicit and inspectable. A plain `while` loop achieves the same logic but hides state transitions. With LangGraph you can visualise the graph, trace each node's input/output in LangSmith, and swap nodes independently.

**Why interleave BM25 and vector results instead of re-ranking?**
Cross-encoder re-ranking (e.g. Cohere Rerank) gives the highest precision but adds a full network round-trip. Interleaving is a zero-latency approximation that performs well in practice when your queries are moderately specific. Re-ranking is the correct upgrade when you need to push precision further.

**Why skip contextual compression?**
`LLMChainExtractor` was evaluated and removed. On this specific corpus (long-form research blog posts), it frequently returned empty strings for chunks that were actually highly relevant — because the extractor looks for "the answer" rather than "supporting context." The document grader is a better fit here.

**Why is the hallucination check prompt strict?**
The hallucination judge uses `grounded=true ONLY if EVERY factual claim is directly supported`. This is intentionally strict. A lenient judge that approves partially-supported answers undermines the entire self-correction loop. Strict grading + limited retries (MAX_RETRIES=2) strikes the right balance.

**Why store `retry_count` in state rather than a mutable variable?**
LangGraph state is immutable between nodes — you return a new state dict on every node call. Any mutation in a conditional edge function is silently dropped. Keeping `retry_count` in state ensures it persists correctly across graph cycles.

---

## ⚠️ Known Limitations

- **Knowledge base is static.** Documents are ingested once. Adding new content requires re-running the ingestion cell.
- **BM25 index is in-memory.** The BM25 index is rebuilt from `doc_splits` each session. For very large corpora this adds startup time.
- **Hallucination checker is binary.** The current implementation classifies answers as fully grounded or not. A scored version (e.g. 0.0–1.0) would allow softer retry thresholds.
- **Query cache doesn't expire.** The in-memory query cache has no TTL. In a long-running server context, stale results could accumulate. For production use, consider Redis with TTL-based eviction.
- **Evaluation dataset is small.** The LangSmith evaluation runs on 3 hand-crafted examples. A statistically meaningful eval requires 50–100+ examples with diverse phrasings and difficulty levels.

---

## 🙋 Author

**Sazna Shaik** — AI Engineer specialising in GenAI, RAG pipelines, and LLM systems on AWS.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shaik-sazna-692376255/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/saznashaik)

---

*Built with LangGraph · LangChain · LiteLLM · Astra DB · OpenAI · LangSmith*
