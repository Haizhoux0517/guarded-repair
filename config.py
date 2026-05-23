import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

FLASH_MODEL = "deepseek-v4-flash"
PRO_MODEL = "deepseek-v4-pro"

NUM_SAMPLES = 1000
DATASET_SEED = 42
DATASET_SPLIT = "validation"
LOCAL_DATASET_PATH = "data/asdiv_numeric_random_1000_seed42.jsonl"

OUTPUT_DIR = "outputs"
RAW_RESULTS_PATH = f"{OUTPUT_DIR}/raw_results.jsonl"
FINAL_REPORT_PATH = f"{OUTPUT_DIR}/final_report.json"
ERROR_SET_PATH = f"{OUTPUT_DIR}/error_set.jsonl"
ERROR_ANALYSIS_PATH = f"{OUTPUT_DIR}/error_analysis.jsonl"

INITIAL_CACHE_PATH = f"{OUTPUT_DIR}/cache/initial_reasoning_cache.jsonl"
GUARDED_OUTPUT_DIR = f"{OUTPUT_DIR}/from_cache_guarded"
GUARDED_RAW_RESULTS_PATH = f"{GUARDED_OUTPUT_DIR}/raw_results.jsonl"
GUARDED_FINAL_REPORT_PATH = f"{GUARDED_OUTPUT_DIR}/final_report.json"
GUARDED_ERROR_SET_PATH = f"{GUARDED_OUTPUT_DIR}/error_set.jsonl"
GUARDED_ERROR_ANALYSIS_PATH = f"{GUARDED_OUTPUT_DIR}/error_analysis.jsonl"
ENABLE_ORACLE_ERROR_ANALYSIS = False

# guarded graph checker settings
ENABLE_GRAPH_GUARD = True
GRAPH_GUARD_TRIGGER_REPAIR = True
GRAPH_MIN_SCORE_DROP = 0.05
GRAPH_GUARD_ACCEPT_THRESHOLD = 0.75
REPAIR_MODEL = PRO_MODEL
META_STEP_THRESHOLD = 0.45
META_GLOBAL_THRESHOLD = 0.50
# repair acceptance gate
REPAIR_MIN_IMPROVEMENT = 0.08