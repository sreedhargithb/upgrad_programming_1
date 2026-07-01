# Skipped Scripts

This document lists Python scripts in this repository that are **excluded from the CI "Run All Scripts" workflow** (`.github/workflows/trigger-all.yml`, `SKIP_LIST` array), together with the reason each script is skipped.

Every entry here corresponds one-to-one with an item in that `SKIP_LIST`. Keep the two in sync when adding or removing scripts.

## Skip criteria

A script is skipped if **any** of the following apply:

- **Interactive input** — the script calls `input()`, `getpass()`, or otherwise expects a human at the keyboard. CI runs are non-interactive and would hang until the 5-minute timeout.
- **Cloud-only runtime** — the script uses APIs available only inside Google Colab / Kaggle notebooks (`google.colab`, kernel-injected secrets, `%%bigquery`, etc.) that cannot be reproduced on a plain Ubuntu runner.
- **Non-public dataset** — the script needs a dataset hosted only behind Kaggle auth (or another auth wall) and no equivalent public mirror was found. Adding Kaggle credentials to the runner is a maintenance burden and each Kaggle download would require a separate secret + workflow permission.

Scripts that used to be here but now have a working public-mirror rewrite (e.g. Telco Churn, Flowers, Chest X-ray) have been removed from `SKIP_LIST` and are executed normally.

## Current skip list

| # | Script | Reason | Category |
|---|--------|--------|----------|
| 1 | `Coding_questions.py` | Contains 8 `input()` calls (interactive coding-challenge harness). Would hang the runner. | Interactive input |
| 2 | `Lead_Scoring_ALL_IN_ONE_COLAB.py` | Imports `google.colab` and uses Colab-only mount / secret injection. Not reproducible on `ubuntu-latest`. | Cloud-only runtime |
| 3 | `Gesture_Recognition_Project.py` | Archived earlier version of the SmartTV gesture assignment. Uses `kagglehub.dataset_download('imsparsh/gesture-recognition')` which needs Kaggle auth. | Non-public dataset |
| 4 | `gesture_control_for_smarttv_model_1_2_3.py` | SmartTV gesture assignment (models 1–3). Same `imsparsh/gesture-recognition` Kaggle-auth dependency; no public mirror for the UpGrad 30-frame video subset. | Non-public dataset |
| 5 | `gesture_control_for_smarttv_model_4_5.py` | SmartTV gesture assignment (models 4–5). Same dataset dependency as above. | Non-public dataset |
| 6 | `gesture_control_for_smarttv_model_6_7_8.py` | SmartTV gesture assignment (models 6–8). Same dataset dependency as above. | Non-public dataset |
| 7 | `gesture_control_for_smarttv_model_9_10.py` | SmartTV gesture assignment (models 9–10). Same dataset dependency as above. | Non-public dataset |
| 8 | `skin_cancer_detection_melanoma.py` | Uses `kagglehub.dataset_download("jaiahuja/skin-cancer-detection")` (2 357-image ISIC subset). No verified no-auth mirror exists for this specific 9-class layout. Candidate ISIC-challenge S3 mirrors (2018 / 2019) have different class structure and are 2.7–9.7 GB, making them impractical to swap in without a full re-write of the training script. | Non-public dataset |

## How to unskip a script

1. Provide the missing prerequisite: either
   - replace interactive input with hard-coded / env-var values, **or**
   - remove the Colab-only import path (guard it behind `try/except ImportError`), **or**
   - find a public no-auth mirror and rewrite the data-load block to use `urllib.request` / `pandas.read_csv(URL)`.
2. Remove the script's basename from `SKIP_LIST` in [`.github/workflows/trigger-all.yml`](.github/workflows/trigger-all.yml).
3. Delete the corresponding row in the table above.
4. Push and confirm the script exits `0` in the next CI run.
