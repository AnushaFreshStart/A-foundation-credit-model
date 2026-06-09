# Credit Foundation Model Application

This folder contains the consolidated **Credit Foundation Model (CFM)** hackathon application. It provides a single, unified FastAPI application and dashboard where you can sequentially execute and visualize the entire pipeline from end-to-end.

## Folder Structure

- **`app.py`**: The single FastAPI server and entry point. It serves the unified dashboard and exposes the REST endpoints to run all pipelines.
- **`ui/`**: Shared static web dashboard code (`index.html`, `style.css`, `app.js`).
- **`pipelines/`**: Core data processing pipelines shared by the entire flow:
  - `db_schema.sql`: DuckDB schema definition.
  - `ingest_bronze.py`: Ingests synthetic Parquet files.
  - `validate_gold.py`: Runs the 18-point gold validation suite.
  - `baseline_model.py`: Trains the XGBoost baseline default prediction model.
- **Sequence construction (Step 2)**:
  - `tokenizer.py`: Loan Event Tokenizer converting event timelines to token sequences.
  - `build_sequences.py`: Sequence builder pipeline producing loan histories.
  - `sequence_dataset.py`: PyTorch dataset wrappers for downstream models.
- **Artifacts**:
  - `sequences.parquet`: The constructed token sequences dataset.
  - `tokenizer.json` & `tokenizer_summary.json`: Stored tokenizer files.
  - `sequence_stats.json`: Statistics report for the constructed sequences.

## Running the Application

You only need to run **one** server to access the entire system.

1. Activate your virtual environment.
2. Navigate to this directory:
   ```bash
   cd src/credit-foundation-model
   ```
3. Run the FastAPI application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to **`http://localhost:8000`**.

Using the dashboard, you can trigger Step 1 (Ingest, Validate, Train Baseline) and Step 2 (Tokenize Sequences) in sequence and observe the results in real-time.
