from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('pip install --quiet "numpy==2.0.2" "pandas==2.2.3" "scikit-learn==1.5.2" "lightgbm==4.5.0" "mlflow==2.17.2"')

import os, shutil, sys, pathlib, subprocess, time, requests

SRC_ROOT = "/content/drive/MyDrive/Assignment_SreedharK/Assignment_SreedharK/Assignment_SreedharK"
assert os.path.isdir(SRC_ROOT), f"Not found: {SRC_ROOT}. Adjust SRC_ROOT to where you uploaded."

DP_SRC = f"{SRC_ROOT}/Lead_scoring_data_pipeline"
TP_SRC = f"{SRC_ROOT}/Lead_scoring_training_pipeline"
IP_SRC = f"{SRC_ROOT}/Lead_scoring_inference_pipeline"

DP_WORK = "/content/data_pipeline"
TP_WORK = "/content/training_pipeline"
IP_WORK = "/content/inference_pipeline"

for src, dst in [(DP_SRC, DP_WORK), (TP_SRC, TP_WORK), (IP_SRC, IP_WORK)]:
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)

DP_SCRIPTS = f"{DP_WORK}/AssignmentFolderFiles/scripts"
TP_SCRIPTS = f"{TP_WORK}/AssignmentFolderFiles/Scripts"
IP_SCRIPTS = f"{IP_WORK}/AssignmentFolderFiles/scripts"

print("Copied:")
print("  data       ->", DP_WORK)
print("  training   ->", TP_WORK)
print("  inference  ->", IP_WORK)

# Data pipeline constants — point at local data + start with labelled CSV
dp_c = pathlib.Path(f"{DP_SCRIPTS}/constants.py")
dp_c.write_text(f"""DB_PATH = '{DP_WORK}/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db'
DATA_DIRECTORY = '{DP_WORK}/data/'
INTERACTION_MAPPING = '{DP_WORK}/mapping/'
INDEX_COLUMNS = ['created_date','city_tier','first_platform_c','first_utm_medium_c','first_utm_source_c','total_leads_droppped','referred_lead','app_complete_flag']
LEAD_SCORING_FILE='leadscoring'
""")

# Training pipeline constants — point DB_PATH at data pipeline output
tp_c = pathlib.Path(f"{TP_SCRIPTS}/constants.py")
tp_txt = tp_c.read_text()
tp_txt = tp_txt.replace("/home/airflow/dags/Lead_scoring_data_pipeline/", f"{DP_WORK}/")
tp_txt = tp_txt.replace("'/home/database/'", f"'{DP_WORK}/'")
tp_c.write_text(tp_txt)

# Inference pipeline constants — point at all 3 local dirs
ip_out = f"{IP_WORK}/output/"
os.makedirs(ip_out, exist_ok=True)
ip_c = pathlib.Path(f"{IP_SCRIPTS}/constants.py")
txt = ip_c.read_text()
txt = txt.replace("/home/airflow/dags/Lead_scoring_data_pipeline/",     f"{DP_WORK}/")
txt = txt.replace("/home/airflow/dags/Lead_scoring_inference_pipeline/", f"{IP_SCRIPTS}/")
txt = txt.replace("/home/airflow/dags/Lead_scoring_training_pipeline/",  f"{TP_WORK}/")
txt = txt.replace("'/home/database/'", f"'{DP_WORK}/'")
txt = txt.replace("'/home/Assignment/03_inference_pipeline/scripts/'", f"'{ip_out}'")
txt = txt.replace("STAGE = 'production'", "STAGE = 'Production'")
ip_c.write_text(txt)

print("--- DP constants ---");  print(dp_c.read_text())
print("--- TP constants ---");  print(tp_c.read_text())
print("--- IP constants ---");  print(ip_c.read_text())

os.system("pkill -f 'mlflow server' 2>/dev/null; sleep 2")

proc = subprocess.Popen(
    ["mlflow", "server",
     "--backend-store-uri", "sqlite:///Lead_scoring_mlflow_production.db",
     "--host", "0.0.0.0", "--port", "6006"],
    cwd=TP_WORK,
    stdout=open("/tmp/mlflow.log", "w"),
    stderr=subprocess.STDOUT,
)

for _ in range(20):
    try:
        if requests.get("http://0.0.0.0:6006/").status_code == 200:
            print(f"MLflow up on http://0.0.0.0:6006 (pid {proc.pid})"); break
    except Exception:
        pass
    time.sleep(2)
else:
    print("MLflow failed — see /tmp/mlflow.log")
    get_ipython().system('tail -30 /tmp/mlflow.log')

import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def step(name, fn, *a, **kw):
    t = time.time()
    print(f">>> {name}")
    out = fn(*a, **kw)
    print(f"    done in {time.time()-t:.1f}s -> {out}\n")
    return out

print("helpers ready")

dp_utils  = load("dp_utils",  f"{DP_WORK}/utils.py")
dp_checks = load("dp_checks", f"{DP_WORK}/data_validation_checks.py")
dp_const  = load("dp_const",  f"{DP_SCRIPTS}/constants.py")
dp_schema = load("dp_schema", f"{DP_WORK}/schema.py")
dp_city   = load("dp_city",   f"{DP_WORK}/mapping/city_tier_mapping.py")
dp_sig    = load("dp_sig",    f"{DP_WORK}/mapping/significant_categorical_level.py")

step("build_dbs",                dp_utils.build_dbs,                dp_const.DB_PATH, dp_const.DB_FILE_NAME)
step("raw_data_schema_check",    dp_checks.raw_data_schema_check,   dp_const.DATA_DIRECTORY, dp_schema.raw_data_schema)
step("load_data_into_db",        dp_utils.load_data_into_db,        dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_const.DATA_DIRECTORY, dp_const.LEAD_SCORING_FILE)
step("map_city_tier",            dp_utils.map_city_tier,            dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_city.city_tier_mapping_dict)
step("map_categorical_vars",     dp_utils.map_categorical_vars,     dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_sig.list_platform, dp_sig.list_medium, dp_sig.list_source)
step("interactions_mapping",     dp_utils.interactions_mapping,     dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_const.INTERACTION_MAPPING, dp_const.INDEX_COLUMNS)
step("model_input_schema_check", dp_checks.model_input_schema_check, dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_schema.model_input_schema)

import sqlite3, pandas as pd
cnx = sqlite3.connect(dp_const.DB_PATH + dp_const.DB_FILE_NAME)
print("Tables in DB:")
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", cnx))
cnx.close()

tp_utils  = load("tp_utils",  f"{TP_SCRIPTS}/utils.py")
tp_const  = load("tp_const",  f"{TP_SCRIPTS}/constants.py")

step("encode_features",   tp_utils.encode_features,
     tp_const.DB_PATH, tp_const.DB_FILE_NAME, tp_const.ONE_HOT_ENCODED_FEATURES, tp_const.FEATURES_TO_ENCODE)

step("get_trained_model", tp_utils.get_trained_model,
     tp_const.DB_PATH, tp_const.DB_FILE_NAME, tp_const.MODEL_CONFIG, tp_const.EXPERIMENT, tp_const.TRACKING_URI)

from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=tp_const.TRACKING_URI)
versions = client.search_model_versions("name='LightGBM'")
latest = max(versions, key=lambda v: int(v.version))

client.transition_model_version_stage(
    name="LightGBM",
    version=latest.version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Promoted LightGBM v{latest.version} -> Production")

# Switch data pipeline to the inference CSV
dp_c.write_text(dp_c.read_text().replace("'leadscoring'", "'leadscoring_inference_final_v2'"))
if os.path.exists(dp_const.DB_PATH + dp_const.DB_FILE_NAME):
    os.remove(dp_const.DB_PATH + dp_const.DB_FILE_NAME)
    print("Removed old DB")

# Reload constants with new value
dp_const = load("dp_const", f"{DP_SCRIPTS}/constants.py")
print("LEAD_SCORING_FILE =", dp_const.LEAD_SCORING_FILE)

# Rebuild the data tables (skip schema checks — they only apply to labelled data)
step("build_dbs",            dp_utils.build_dbs,            dp_const.DB_PATH, dp_const.DB_FILE_NAME)
step("load_data_into_db",    dp_utils.load_data_into_db,    dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_const.DATA_DIRECTORY, dp_const.LEAD_SCORING_FILE)
step("map_city_tier",        dp_utils.map_city_tier,        dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_city.city_tier_mapping_dict)

step("map_categorical_vars", dp_utils.map_categorical_vars, dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_sig.list_platform, dp_sig.list_medium, dp_sig.list_source)
inference_index_columns = [c for c in dp_const.INDEX_COLUMNS if c != 'app_complete_flag']
step("interactions_mapping", dp_utils.interactions_mapping, dp_const.DB_PATH, dp_const.DB_FILE_NAME, dp_const.INTERACTION_MAPPING, inference_index_columns)
print("Inference data ready.")

ip_utils = load("ip_utils", f"{IP_SCRIPTS}/utils.py")
ip_const = load("ip_const", f"{IP_SCRIPTS}/constants.py")

step("encode_features",       ip_utils.encode_features,
     ip_const.DB_PATH, ip_const.DB_FILE_NAME, ip_const.ONE_HOT_ENCODED_FEATURES, ip_const.FEATURES_TO_ENCODE)

step("input_features_check",  ip_utils.input_features_check,
     ip_const.DB_PATH, ip_const.DB_FILE_NAME, ip_const.ONE_HOT_ENCODED_FEATURES)

step("get_models_prediction", ip_utils.get_models_prediction,
     ip_const.DB_PATH, ip_const.DB_FILE_NAME, ip_const.MODEL_NAME, ip_const.STAGE, ip_const.TRACKING_URI)

step("prediction_ratio_check", ip_utils.prediction_ratio_check,
     ip_const.DB_PATH, ip_const.DB_FILE_NAME, ip_const.SCRIPTS_OUTPUT)

import glob

cnx = sqlite3.connect(ip_const.DB_PATH + ip_const.DB_FILE_NAME)
preds    = pd.read_sql("SELECT * FROM predicted_output", cnx)
features = pd.read_sql("SELECT * FROM features", cnx)
cnx.close()

print("Total predictions:", len(preds))
print(preds["predicted_output"].value_counts(normalize=True).rename("share"))

print("\nLatest prediction distribution files:")
for f in sorted(glob.glob(ip_const.SCRIPTS_OUTPUT + "prediction_distribution_*.txt"))[-3:]:
    print("-", f)
    print(open(f).read())

# === Consolidated results CSV (features + prediction per lead) ===
results_path = f"{IP_WORK}/output/lead_scoring_predictions.csv"
results = pd.concat([features.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
results.to_csv(results_path, index=False)
print(f"\nWrote {len(results)} rows -> {results_path}")
print(results.head())

from google.colab import files
files.download(results_path)

import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


