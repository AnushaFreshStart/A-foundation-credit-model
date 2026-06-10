"# A-foundation-credit-model" 

Step 1 — Generate Synthetic Data (if out_10k doesn't exist)

cd A-foundation-credit-model\deep-loans\synthetic-data-designer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install data-designer numpy pandas pyarrow duckdb

python run.py --num-records 10000 --out-dir .\out_10k --deal-year 2024 --seed 42


cd A-foundation-credit-model\src\credit-foundation-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Step 2 — Start the FastAPI App

cd A-foundation-credit-model\src\credit-foundation-model
python app.py
Open http://localhost:8000 and use the dashboard to trigger:

Step 1: Ingest → Validate → Train Baseline
Step 2: Tokenize Sequences
Step 3: Foundation model
Step 4: Downstreams

