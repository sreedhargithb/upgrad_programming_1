# gcolab CLI runner repository

This repository is a transformation of `upgrad_programming` where every
Jupyter notebook is appended with two extra cells so it is runnable
headlessly via the Google Colab CLI from GitHub Actions:

1. List files in `outputs/`
2. Zip `outputs/` -> `outputs.zip` (downloaded as the build artifact)

## How to run

1. Add a GitHub Actions secret `GCP_ADC_JSON` containing your Google
   Application Default Credentials JSON.
2. Open the **Actions** tab and pick one of the `gcolab - *` workflows.
3. Click **Run workflow**. The job will provision a free Colab T4 GPU
   session, execute every notebook in that section sequentially, and
   upload the per-notebook `outputs.zip` plus the executed `.ipynb` as
   build artifacts.

## Workflow files

- `.github/workflows/gcolab_root.yml` - Root (2 notebooks)
- `.github/workflows/gcolab_prep.yml` - Prep Sessions (6 notebooks)
- `.github/workflows/gcolab_exam1.yml` - Exam 1 (Stats + ML) (16 notebooks)
- `.github/workflows/gcolab_exam2.yml` - Exam 2 (ML2 + Deep Learning) (42 notebooks)
- `.github/workflows/gcolab_nlp.yml` - NLP (51 notebooks)
- `.github/workflows/gcolab_mlops.yml` - MLOps (1 notebooks)
- `.github/workflows/gcolab_capstone.yml` - Capstone (4 notebooks)
