# Production-Grade RAG Architecture for Large-Scale File-Based Knowledge Systems

## 1. Executive Summary

This document describes a **production-grade Retrieval-Augmented Generation (RAG) system** designed for a company managing **10+ million structured and unstructured files** (PDF, CSV, XLSX) organized by **project / material / content**.

The goal is to enable a **GDPR-compliant, Germany-hosted LLM (Mistral AI)** to answer questions **strictly grounded in internal files**, with **traceability, scoring, and source attribution**.

This document covers:

* Business requirements
* Data and file structure assumptions
* End-to-end system architecture
* Technology stack (POC vs Production)
* Indexing, retrieval, scoring, and answering logic
* Scalability, security, and compliance considerations

---

## 2. Business Problem

The company maintains a very large file repository (>10M files) containing:

* Pricing data
* Technical specifications
* Materials documentation
* Project-specific datasets

### Example File Structure

```
projectname/
  material/
    content/

apexindustries/
  metal/
    metal.csv
  wood/
    wood.csv
    wood.pdf
```

### User Expectations

Users want to ask **natural language questions**, such as:

> "What is the price of wood for the ApexIndustries project?"

And receive:

* An answer **only from internal files**
* A relevance/confidence score
* Clear source attribution (file path, file type)

---

## 3. Core Requirements

### Functional Requirements

* Track file system changes (create / update / delete)
* Extract text and tabular data from PDF, CSV, XLSX
* Preserve file hierarchy as metadata
* Semantic + structured retrieval
* Answer generation using **Mistral AI only**
* Relevance scoring and source citation

### Non-Functional Requirements

* GDPR-compliant (Germany / EU hosting)
* Production-grade scalability (10M+ files)
* Incremental indexing (no full reindex on every change)
* High availability and fault tolerance
* Auditability and explainability

---

## 4. High-Level Architecture

```
[File System]
     ↓
[Change Detection / File Watcher]
     ↓
[Document Ingestion & Parsing]
     ↓
[Chunking + Metadata Enrichment]
     ↓
[Embedding Generation (Mistral-compatible)]
     ↓
[Vector Database + Metadata Index]
     ↓
[Hybrid Retrieval Engine]
     ↓
[Re-ranking + Scoring]
     ↓
[Mistral LLM]
     ↓
[Answer + Sources + Confidence]
```

---

## 5. File Ingestion & Tracking

### Change Detection Strategy

For 10M+ files, OS-level file watchers are insufficient.

**Recommended approach:**

* Periodic directory scans with hash + timestamp comparison
* Event-based tracking if backed by object storage (S3-compatible)

### Metadata Captured per File

* project_name (e.g. apexindustries)
* material (e.g. wood, metal)
* file_path
* file_type
* last_modified
* version_hash

---

## 6. Content Extraction Layer

### File-Type Handling

| Type | Extraction Strategy            |
| ---- | ------------------------------ |
| CSV  | Row/column parsing             |
| XLSX | Sheet + table parsing          |
| PDF  | Text extraction + OCR fallback |

Each file is converted into **normalized text blocks**.

---

## 7. Chunking & Semantic Units

Files are split into **semantic chunks**:

* CSV/XLSX: per logical table or row group
* PDF: per section or paragraph

Each chunk includes metadata:

* project_name
* material
* file_path
* chunk_index
* semantic_type (price, spec, description)

This ensures **precise retrieval and traceability**.

---

## 8. Embedding & Vector Storage (Production)

### Embedding Strategy

* Embeddings generated using **EU-hosted Mistral-compatible embedding models**
* Batched and asynchronous processing

### Recommended Vector Store (Production)

**Primary Recommendation: Qdrant (Self-Hosted)**

Reasons:

* High-performance at scale
* Strong metadata filtering
* GDPR-friendly self-hosting
* Mature production usage

Alternative (if hybrid keyword search is critical):

* Weaviate (self-hosted, EU deployment)

---

## 9. Retrieval Strategy

### Hybrid Retrieval

1. Semantic vector search
2. Metadata filtering:

   * project_name = "apexindustries"
   * material = "wood"
3. Optional keyword (BM25) fallback

### Scoring Components

* Vector similarity score
* Metadata match score
* Recency score (optional)

Final relevance score is a weighted combination.

---

## 10. Re-ranking Layer

Before sending context to the LLM:

* Top-N chunks are re-ranked using a cross-encoder or LLM-based scorer
* Reduces hallucination risk
* Improves precision for numerical queries (prices, quantities)

---

## 11. Answer Generation (Mistral AI)

### Prompt Strategy

The LLM is instructed to:

* Answer **only using provided context**
* Return "Not found" if data is missing
* Cite file paths and chunk IDs

### Example Prompt Skeleton

```
You are an enterprise assistant.
Answer ONLY using the provided sources.
If the answer is not present, say "Data not available".

Sources:
{{retrieved_chunks}}

Question:
{{user_question}}
```

---

## 12. Output Format

The system returns:

* Answer text
* Confidence score
* Source list

Example:

```
Answer: The price of wood for ApexIndustries is 120 EUR per unit.
Confidence: 0.87
Sources:
- apexindustries/wood/wood.csv (row 14)
- apexindustries/wood/wood.pdf (section 2.1)
```

---

## 13. Security & GDPR Compliance

* All data stored and processed in EU (Germany)
* No external data leakage
* Role-based access control
* Full audit logs for queries
* Encryption at rest and in transit

---

## 14. Demo POC vs Production Stack

### Current POC

* ChromaDB
* Tavily (external search)
* Mistral API

### Production Upgrade Path

| Layer     | POC      | Production          |
| --------- | -------- | ------------------- |
| Vector DB | Chroma   | Qdrant / Weaviate   |
| Ingestion | Script   | Distributed workers |
| Storage   | Local FS | Object storage      |
| Retrieval | Basic    | Hybrid + rerank     |
| Security  | none     | RBAC + audit        |

---
ssh montassar@4.210.218.116
MontassarDrag&1
## 15. Scalability Considerations

* Horizontal ingestion workers
* Sharded vector collections by project/material
* Background re-indexing
* Caching of frequent queries

Designed to scale from **POC → enterprise production** without re-architecture.

---

## 16. Conclusion

This architecture provides:

* Strict grounding on internal data
* High accuracy with explainability
* GDPR-compliant Mistral-based LLM usage
* Clear upgrade path from demo to production

It is suitable both as a **sales POC** and as a **long-term enterprise solution**.

---

**Prepared for:** Enterprise Knowledge & AI Systems
**Purpose:** Demo validation and production planning
