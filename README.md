# Stage 1: NLP Vulnerability Knowledge Extraction Pipeline

Turns unstructured CVE and advisory text into a structured, queryable Vulnerability
Knowledge Base. This knowledge base is the grounding layer that Stage 2's agent can
retrieve from using RAG. See `project_plan.md` for full project context. This README
covers only the Stage 1 pipeline.

**v1 scope:** SQL Injection (CWE-89) and Command Injection (CWE-78) only.

## Pipeline stages

```text
nvd_client.py          → pull CVE data from NVD, filtered to target CWEs
exploitdb_client.py    → cross-reference matched CVEs against known public exploits
gpt_annotator.py       → use GPT to create NER silver labels from CVE descriptions
train_ner.py           → fine-tune a transformer NER model on the GPT-labeled data
attack_classifier.py   → train a multi-label classifier: free text → MITRE ATT&CK technique(s)
knowledge_graph.py     → combine CVE data, ATT&CK predictions, exploits, and FAISS index
config.py              → shared settings, paths, label schema, API config
preprocessing.py       → shared text-cleaning helpers
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

## Environment variables

Create a `.env` file in the project root, or export the variables in your terminal.

Required for GPT-based NER annotation:

```bash
OPENAI_API_KEY=your-openai-key-here
```

Optional but recommended for a higher NVD rate limit:

```bash
NVD_API_KEY=your-nvd-key-here
```

The NVD API key increases the rate limit from the public unauthenticated limit to the authenticated limit.

## Running Stage 1

Run the scripts in this order.

### 1. Collect CVE data from NVD

```bash
python nvd_client.py
```

For a quick test run, use the script's max-record option if available in your local version.
For example:

```bash
python nvd_client.py --max-cves 20
```

This creates cleaned CVE files such as:

```text
data/raw/clean_cwe89.json
data/raw/clean_cwe78.json
```

### 2. Cross-reference Exploit-DB

```bash
python exploitdb_client.py
```

This creates:

```text
data/raw/exploitdb_matches.json
```

### 3. Generate GPT-assisted NER silver labels

For a quick test:

```bash
python gpt_annotator.py --max-records 50 --spot-check 10
```

For a full run:

```bash
python gpt_annotator.py
```

This creates:

```text
data/processed/ner_silver_dataset.jsonl
```

The `--spot-check` option prints random labeled examples so you can visually inspect whether the GPT labels look correct.

### 4. Train the NER model

After reviewing the GPT-labeled silver data:

```bash
python train_ner.py
```

This creates:

```text
models/ner_model/
```

### 5. Train the ATT&CK classifier

```bash
python attack_classifier.py
```

This creates:

```text
models/attack_classifier.joblib
models/attack_label_binarizer.joblib
```

### 6. Build the final knowledge base and FAISS index

```bash
python knowledge_graph.py
```

This creates:

```text
data/processed/vulnerability_knowledge_base.json
models/vuln_kb.faiss
models/vuln_kb_meta.json
```

## Outputs

- `data/raw/clean_cwe89.json`, `data/raw/clean_cwe78.json` — cleaned CVE records per CWE
- `data/raw/exploitdb_matches.json` — known public exploits cross-referenced by CVE
- `data/processed/ner_silver_dataset.jsonl` — GPT-labeled NER training data
- `models/ner_model/` — fine-tuned NER model after running `train_ner.py`
- `models/attack_classifier.joblib` — trained ATT&CK classifier
- `models/attack_label_binarizer.joblib` — label converter for ATT&CK technique IDs
- `data/processed/vulnerability_knowledge_base.json` — final structured vulnerability KB
- `models/vuln_kb.faiss` — FAISS vector index for semantic retrieval
- `models/vuln_kb_meta.json` — metadata that maps FAISS rows back to CVE IDs


## Known limitations

1. **GPT-assisted NER labels are still silver labels, not ground truth.**
   `gpt_annotator.py` usually produces better labels than regex-only weak supervision, but it can still make mistakes. Review a sample of `data/processed/ner_silver_dataset.jsonl` before training the NER model.

2. **NER training quality depends on annotation quality.**
   If the GPT labels contain incorrect spans or inconsistent labels, `train_ner.py` can learn those mistakes. For stronger results, manually review and correct a subset of examples before fine-tuning.

3. **ATT&CK classifier labels are weak labels.**
   `attack_classifier.py` uses the static CWE-to-ATT&CK mapping in `config.py` to build training labels. This is useful for bootstrapping, but it is not the same as expert-confirmed ATT&CK mapping for every CVE.

4. **Exploit-DB source verification may be needed.**
   If the Exploit-DB raw CSV URL changes, `exploitdb_client.py` may need to be updated. Run a small test first and confirm that `data/raw/exploitdb_matches.json` is created correctly.

5. **NVD API behavior may change.**
   The NVD client follows the NVD API structure used during development. Test with a small run before collecting a large dataset.

6. **Large generated files are intentionally not tracked.**
   Raw CVE data, processed datasets, trained models, and FAISS indexes should be regenerated locally rather than committed to GitHub.

## Suggested project structure

```text
stage1_nlp_pipeline/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
│
├── config.py
├── preprocessing.py
├── nvd_client.py
├── exploitdb_client.py
├── gpt_annotator.py
├── train_ner.py
├── attack_classifier.py
├── knowledge_graph.py
│
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
│
└── models/
    └── .gitkeep
```

## Next steps after Stage 1

Once `vulnerability_knowledge_base.json` and the FAISS index are built, Stage 2 can query the index for grounding during its planning loop.
