
# Hybrid RAG over Cybersecurity Internal Documents

An end-to-end Retrieval-Augmented Generation (RAG) application designed to help cybersecurity teams retrieve information from internal security documentation using a locally hosted Large Language Model.

The project implements semantic search, keyword search, hybrid retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking and grounded LLM generation.

## Project Overview

Organisations maintain large collections of cybersecurity documents, including:

- Information security policies
- Incident response procedures
- Vulnerability assessment reports
- Risk registers
- Security control documentation
- Compliance standards
- Threat intelligence reports
- Asset inventories

Searching these documents manually can be time-consuming.

This project aims to build an AI-powered internal knowledge assistant that allows employees to ask natural-language questions and receive answers grounded in the organisation's documents.

The application will retrieve relevant information and provide source references to support generated answers.

## Project Objectives

- Build a multi-format document ingestion pipeline.
- Extract and preprocess information from PDF, DOCX, TXT, CSV and XLSX files.
- Implement document chunking and metadata management.
- Generate text embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Implement semantic retrieval using vector similarity.
- Implement keyword retrieval using BM25.
- Combine retrieval results using Reciprocal Rank Fusion.
- Improve retrieval relevance using cross-encoder reranking.
- Integrate a locally hosted LLM through Ollama.
- Generate grounded answers with source citations.
- Evaluate retrieval quality and application performance.
- Build an interactive application using Streamlit.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.13 |
| Package management | uv |
| LLM runtime | Ollama |
| Initial generation model | Qwen3 8B |
| Document extraction | PyMuPDF, python-docx |
| Spreadsheet processing | openpyxl |
| Embedding model | Sentence Transformers |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Hybrid retrieval | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| User interface | Streamlit |
| Testing | pytest |
| Code quality | Ruff |
| Version control | Git and GitHub |

## Architecture

```text
User
 |
 v
Streamlit Interface
 |
 v
Document Ingestion
 |
 +-- PDF
 +-- DOCX
 +-- TXT
 +-- CSV
 +-- XLSX
 |
 v
Document Processing
 |
 +-- Text Extraction
 +-- Cleaning
 +-- Metadata
 +-- Chunking
 |
 v
Embedding Generation
 |
 v
Retrieval
 |
 +--------------------+
 |                    |
 v                    v
Vector Search       BM25 Search
 |                    |
 +---------+----------+
           |
           v
      Hybrid Retrieval
           |
           v
           RRF
           |
           v
    Cross-Encoder
       Reranking
           |
           v
    Context Construction
           |
           v
       Ollama LLM
           |
           v
   Grounded Answer
           |
           v
      Citations
```

## Project Structure

```text
hybrid-rag-cybersecurity/
|
+-- data/
|   +-- raw/
|   |   +-- pdf/
|   |   +-- docx/
|   |   +-- csv/
|   |   +-- xlsx/
|   |   +-- txt/
|   |
|   +-- processed/
|   +-- chroma_db/
|
+-- src/
|   +-- hybrid_rag_cybersecurity/
|       +-- __init__.py
|       +-- main.py
|       +-- ingestion/
|       +-- processing/
|       +-- embeddings/
|       +-- retrieval/
|       +-- reranking/
|       +-- generation/
|       +-- evaluation/
|       +-- utils/
|
+-- tests/
+-- configs/
+-- app/
+-- notebooks/
+-- scripts/
|
+-- .env.example
+-- .gitignore
+-- .python-version
+-- pyproject.toml
+-- uv.lock
+-- README.md
```

## Installation and Setup

### Prerequisites

- macOS, Linux or Windows
- Python 3.13
- uv
- Git
- Ollama

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd hybrid-rag-cybersecurity
```

### 2. Synchronise dependencies

```bash
uv sync
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com/download

### 5. Download the initial LLM

```bash
ollama pull qwen3:8b
```

### 6. Verify Ollama

```bash
ollama list
```

## Run the Streamlit application

The app reads the existing `data/chroma_db` collection and source files under
`data/raw` to prepare keyword search. It does not build or update the vector
database. Build the collection separately with the following command only
when you intend to ingest or refresh the indexed documents:

```bash
uv run python scripts/build_vector_store.py
```

Start the app with:

```bash
uv sync
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3:8b
uv run streamlit run app/streamlit_app.py
```

The cross-encoder is downloaded by Sentence Transformers on first use. The
app displays retrieved source names and page, sheet, or row locations when
available. Its citation checks are structural warnings and do not prove that
an answer is semantically supported.

## Automated checks

Run the fast unit suite and lint checks with:

```bash
uv run pytest
uv run ruff check src tests app scripts
uv run ruff format --check src tests app scripts
```

These unit tests mock the model and vector-store boundaries; they do not call
Ollama or write to the local Chroma database.

## Document Ingestion

The application supports the following initial document formats:

| Format | Extraction library | Extraction strategy |
|---|---|---|
| PDF | PyMuPDF | One record per non-empty page |
| DOCX | python-docx | Paragraphs and tables |
| TXT | pathlib | One record per file |
| CSV | csv | One record per data row |
| XLSX | openpyxl | One record per worksheet data row |

Each extracted record contains:

- Text content
- Source filename
- Source path
- File type
- Relevant page, row or worksheet metadata

### Sample cybersecurity data sources

- NIST Cybersecurity Framework: https://www.nist.gov/publications/nist-cybersecurity-framework-csf-20
- OWASP ASVS: https://asvs.owasp.org/downloads/
- CISA KEV: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
- MITRE ATT&CK: https://attack.mitre.org/resources/working-with-attack/

Synthetic internal cybersecurity documents will also be used for testing.

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Environment, configuration, and repository safety | Implemented |
| 2 | Multi-format document ingestion and processing | Implemented |
| 3 | Embeddings, vector storage, and hybrid retrieval | Implemented |
| 4 | Reranking, grounded generation, and evaluation tools | Implemented; live model run pending |
| 5 | Streamlit application | Implemented; local runtime check pending |
| 6 | Basic security review | Reviewed; unresolved findings are tracked below |

## Evaluation

The completed application will be evaluated using:

- Retrieval Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Retrieval latency
- Answer relevance
- Citation correctness
- Groundedness and hallucination behaviour

## Security Review and Findings

**Review date:** 2026-11-16  
**Scope:** static review of Python source and configuration, secret-pattern search,
Git ignore/status review, and a vulnerability scan of the locked Python
requirements. The existing raw documents and Chroma database were not opened or
modified as part of this review.

### Checks completed

- Searched project source for obvious embedded credential/private-key patterns,
  shell execution, `eval`, and `exec`. No obvious credential patterns or
  application-level shell execution were found. This is a targeted source scan,
  not a guarantee that secrets are absent.
- Confirmed `.env` and local database ignore rules; `data/raw/` and
  `data/chroma_db/` are ignored by Git. The repository currently has no commits,
  so this review could not scan prior Git history for secrets.
- Reviewed the app's user input limit (2,000 characters), default safe Markdown
  rendering, document-as-untrusted-data prompt rules, and exception handling.
  The app now shows generic errors in the browser and keeps diagnostic details in
  local logs.
- Reviewed the query path: the app does not call the vector-store indexing,
  update, or delete methods. Index construction remains a separate manual script.
- Exported all locked dependency groups and ran `pip-audit` against OSV. The scan
  reported four distinct advisories for `chromadb==1.5.9` (the tool emitted eight
  records, including duplicates). No other package was reported by that scan.

### Findings and follow-up register

| ID | Severity | Finding | Current handling / next step |
|---|---|---|---|
| SEC-01 | Critical / high, conditional exposure | The lock resolves `chromadb==1.5.9`, which is in the affected ranges for CVE-2026-45829, CVE-2026-45833, CVE-2026-45830, and CVE-2026-45831. The first two advisories describe code-injection paths in Chroma's server; the latter two concern authorization scope in server deployments. The reviewed advisories list no patched release. | This project uses Chroma through its local persistent client and does not start a Chroma HTTP server. Keep it that way: do not run or expose a Chroma server to untrusted clients. Recheck the advisories before deployment and upgrade when a fixed release is published. [CVE-2026-45829](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c), [CVE-2026-45833](https://github.com/advisories/GHSA-36p7-vc44-83pf), [CVE-2026-45830](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr), [CVE-2026-45831](https://github.com/advisories/GHSA-xph7-9rjv-w5fr). |
| SEC-02 | Medium before network exposure | The Streamlit interface has no user authentication or document-level authorization. Anyone who can reach a shared instance can query the indexed corpus and consume local model resources. | Run for a trusted local user only. Before sharing it on a network, add authentication, authorization, request limits, and a controlled deployment boundary. |
| SEC-03 | Medium | Prompt instructions and structural citation checks cannot guarantee resistance to malicious instructions in retrieved documents or semantic support for generated claims. | Treat generated responses as advisory. Add adversarial prompt-injection tests and independent semantic evaluation before operational use. |
| SEC-04 | Low / privacy | Ingestion metadata includes source paths. Sharing or exporting the Chroma database could reveal local directory information. | Keep the database private; consider storing relative paths or only source names in any shareable index. |
| SEC-05 | Low / review limitation | The secret scan was pattern-based and there is no commit history to inspect. | Run a dedicated secret scanner when establishing version control, and repeat it before publishing the repository. |

The Chroma findings are dependency advisories, not proof that the local embedded
workflow reaches the vulnerable HTTP endpoints. This review did not launch a
Chroma server, test exploit payloads, or perform a penetration test. Advisory
status can change; recheck it before any deployment.

## Current Implementation Status

The ingestion, processing, hybrid retrieval, reranking, answer-generation,
evaluation, and Streamlit modules are present. Unit tests use mocks around
expensive model and vector-store boundaries. A live end-to-end run still needs
to be performed in the local environment with Ollama, the required models, and
the existing Chroma collection.

## Project Limitations

The initial implementation will focus on text-based document extraction. Scanned PDF OCR, complex table extraction, advanced document parsing and production-grade access control are future improvements unless added during development.

## License and Attribution

External cybersecurity standards and datasets remain subject to their respective licenses and terms of use.

Synthetic company documents are created exclusively for demonstration and testing.

## Future Improvements

- OCR for scanned PDFs
- Advanced table extraction
- Metadata-based retrieval filtering
- Document version management
- Multi-agent security assistant
- AI-powered SOC alert investigation
- Integration with security monitoring platforms
