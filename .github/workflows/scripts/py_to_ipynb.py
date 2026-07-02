"""Wrap each .py listed below in a single-cell .ipynb next to it.

Used to shift the heavy CIFAR-10 / MNIST CNN training scripts (chunks 9 and
10 of the CI matrix) out of the CPU-only GitHub runner and onto Colab T4
GPU via the `gcolab-cnn-chunks-9-10.yml` workflow. Each generated notebook
contains ONE code cell whose source is the .py file byte-for-byte, so the
original get_ipython().system(...) / run_line_magic(...) calls just work
under Jupyter/Colab (they were %-magics before nbconvert produced the .py).

Re-run this script any time a source .py is edited to refresh its .ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Files in chunks 9 and 10 of the trigger-all.yml matrix. Paths are
# relative to the repo root. Every entry gets a sibling .ipynb.
SOURCES = [
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_2_Intro_to_CNN/Reading_Digital_Image.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/0_Building+a+CNN+_+MNIST_Copy1.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/1_+Cifar_10_with_dropout_without_Batch_Normalization_(BN).py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/2_+Cifar_10_Notebook_with_BN_without_dropout.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/3_+Cifar_10_notebook.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/4_+Cifar10_l2_notebook.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/5_+Cifar10_l2_dropout_notebook.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/6_+Cifar10_morelayer_notebook.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_3_Building_CNNs_with_Python_and_Keras/7_+Cifar10_feature_map_updated.py",
    "2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_4_CNN_Architectures_and_Transfer_Learning/_0_readme.py",
]


def wrap(py_path: Path) -> dict:
    text = py_path.read_text(encoding="utf-8")
    # nbformat expects `source` to be a list of lines each ending with \n
    # (except possibly the last). splitlines(keepends=True) preserves that.
    source = text.splitlines(keepends=True)
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {
                "name": "python",
                "file_extension": ".py",
                "mimetype": "text/x-python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    written = 0
    for rel in SOURCES:
        py = REPO_ROOT / rel
        if not py.is_file():
            print(f"MISSING: {rel}")
            continue
        nb_path = py.with_suffix(".ipynb")
        nb_path.write_text(
            json.dumps(wrap(py), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {nb_path.relative_to(REPO_ROOT)}")
        written += 1
    print(f"\ndone: {written}/{len(SOURCES)} notebooks written")


if __name__ == "__main__":
    main()
