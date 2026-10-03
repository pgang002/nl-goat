"""
Central configuration for the Stage 1 NLP extraction pipeline.

Everything that other modules need to agree on (paths, scope, API
settings, label schemas) lives here so it's defined exactly once.
"""
from pathlib import Path

# --- Project paths -------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- v1 locked-in scope (see project_plan.md section 1a) -----------------
# CWE-89 = SQL Injection, CWE-78 = OS Command Injection
TARGET_CWE_IDS = ["CWE-89", "CWE-78"]

# --- NVD API ---------------------------------------------------------------
# Docs: https://nvd.nist.gov/developers/vulnerabilities
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = None  # set via env var NVD_API_KEY instead of hardcoding (raises rate limit from 5 to 50 req/30s)
NVD_RESULTS_PER_PAGE = 2000  # NVD API max page size
NVD_RATE_LIMIT_DELAY_NO_KEY = 6.5   # seconds between requests, safely under 5 req/30s
NVD_RATE_LIMIT_DELAY_WITH_KEY = 0.7  # seconds between requests, safely under 50 req/30s

# --- exploit-db ------------------------------------------------------------
# NOTE: Offensive Security migrated Exploit-DB from GitHub to GitLab in
# 2022; the GitHub repo (offensive-security/exploitdb) is now an archived
# legacy mirror with stale data. Use the GitLab source below.
# Repo: https://gitlab.com/exploit-database/exploitdb
EXPLOITDB_CSV_URL = (
    "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
)

# --- ATT&CK mapping ----------------------------------------------------------
# Minimal static seed table for the two in-scope CWEs. This seeds the
# weak-label training data for attack_classifier.py — expand/refine as
# real labeled data comes in.
CWE_TO_ATTACK_TECHNIQUE = {
    "CWE-89": [
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application"},
        {"technique_id": "T1505.003", "technique_name": "Server Software Component: Web Shell"},
    ],
    "CWE-78": [
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application"},
        {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter"},
    ],
}

# --- NER entity schema -------------------------------------------------------
NER_LABELS = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-VERSION", "I-VERSION",
    "B-ATTACK_VECTOR", "I-ATTACK_VECTOR",
    "B-COMPONENT", "I-COMPONENT",
]

# --- Logging ------------------------------------------------------------------
LOG_LEVEL = "INFO"
