# Research & Design Comparison: Credit Foundation Models & Temporal Databases

This document explores how our **PostgreSQL-optimized temporal credit model design** aligns with state-of-the-art academic literature and industrial implementations. It details similar research, theoretical baselines, and architectural justifications.

---

## 1. Industry Precursors & Similar Models

Our overall design (treating loan histories as event sequences for representation learning) is directly aligned with recent advances in **Transaction Foundation Models (TFMs)** and **Tabular Foundation Models (LTMs)**.

### A. Revolut's PRAGMA Model
*   **Domain**: Payment and transaction sequence modeling.
*   **Methodology**: PRAGMA is a Large Transaction Model (LTM) developed by Revolut. Instead of using static borrower snapshots (e.g., credit bureau scores), PRAGMA models a customer’s raw transaction logs over time as text-like tokens. A temporal Transformer learns customer state representations.
*   **Similarity to Our Design**: Our architecture tokenizes monthly loan panel dynamics (Performing, DPD levels, Cures, Defaults, Prepayments) as event sequences, mirroring how PRAGMA tokenizes payment event streams.

### B. NVIDIA Transaction Foundation Model (TFM) Blueprint
*   **Domain**: NeMo-based sequence modeling on tabular datasets.
*   **Methodology**: NVIDIA's TFM blueprint provides standard guidelines for building foundation models on credit card transactions, retail purchases, and bank accounts. The blueprint focuses on:
    *   **Vectorized sequence building**: Aggregating raw event logs into time-aligned sequence vectors.
    *   **Downstream fine-tuning**: Adapting the pretrained sequence model to default, attrition, or fraud targets.
*   **Similarity to Our Design**: The hackathon project deck explicitly references the NVIDIA TFM blueprint as the target architecture.

### C. TabPFN (Tabular Prior-Data Fitted Network)
*   **Domain**: Tabular Foundation Models.
*   **Methodology**: Developed by PriorLabs/University of Freiburg, TabPFN is a Transformer model trained on millions of synthetic datasets. It achieves strong "zero-shot" and "few-shot" predictions on tabular benchmarks without requiring training-time gradient updates or hyperparameter optimization.
*   **Relevance to Our Design**: TabPFN shows that synthetically generated datasets (like our Dutch RMBS dataset) can successfully proxy real-world portfolios to bootstrap large-scale foundation networks.

---

## 2. Database-Mediated Feature Engineering & Sequence Building

Our design shifts sequence tokenization and temporal feature engineering into **PostgreSQL** using window functions, composite indexing, and database views. This aligns with academic research on **Predictive Query Languages (PQLs)** and **Feature Stores**.

```
Raw Data Ingestion
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ PostgreSQL 16                                          │
│  ├── static_loans (1:1 static attributes)             │
│  │                                                     │
│  └── dynamic_performance (1:Many monthly cutoffs)     │
│        └── Composite Index (loan_id, reporting_date)   │
└──────┬─────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ SQL Views & Materialized Views                         │
│  ├── gold_features (LEAD/LAG window features)          │
│  └── event_sequence (JSONB aggregations)               │
└──────┬─────────────────────────────────────────────────┘
       │ (High-speed, memory-efficient query)
       ▼
┌────────────────────────────────────────────────────────┐
│ Python ML Pipeline                                     │
│  ├── Baseline Model (XGBoost OOT train/test)           │
│  └── Credit Foundation Model (Temporal embeddings)     │
└────────────────────────────────────────────────────────┘
```

### Theoretical Context: Predictive Query Languages (PQL)
Academic research (e.g., *“A Predictive Query Language for Time-Series Relational Data”*) highlights that processing time-series data for ML inside databases is significantly more efficient than doing so in memory using Python:

1.  **Out-of-Core Processing**: A Python Pandas script requires loading all 11.2 million rows (~1.2 GB raw, but up to 6 GB in-memory) into RAM to compute `LEAD()` and `LAG()` states for temporal default labels. PostgreSQL processes this out-of-core, writing temporary results to disk when memory limits are reached, ensuring scalability.
2.  **Point-in-Time Joins (Data Leakage Prevention)**: The most common failure in temporal modeling is **data leakage** (e.g., using information from $t+1$ to predict at $t$). By structuring the feature engineering inside a Postgres View using deterministic SQL, we guarantee that features only reference past states relative to the `reporting_date` index.
3.  **JSONB for Token Sequences**: Traditional ML pipelines require custom Python tokenizers. In PostgreSQL, we can use `jsonb_agg` to build chronological sequences of JSON event tokens directly on the database engine. This provides clean, model-ready JSON sequences to deep learning pipelines (PyTorch/NeMo) without additional processing overhead.

---

## 3. Comparison of Design Implementations

| Metric | Flat Parquet Files (Baseline Design) | PostgreSQL 16 (Our Improved Design) |
| :--- | :--- | :--- |
| **Storage foot-print** | High (static data duplicated across all 24 cutoffs). | Low (3NF normalization via static/dynamic tables). |
| **Data Drift Risk** | Present (static traits can diverge over cutoffs due to bugs). | **Zero** (structural guarantee via foreign keys). |
| **Temporal Labeling** | Python Pandas `shift()` (memory intensive). | SQL view with `LEAD()` window function. |
| **Sequence Building** | Manual Python grouping & sorting. | `jsonb_agg` ordered sequence queries. |
| **Inference Serving** | Reload flat files. | Real-time SQL feature retrieval for live loans. |
