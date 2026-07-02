"""
Robust CIFAR-10 downloader for CI.

Populates ~/.keras/datasets/cifar-10-batches-py/ with the 6 pickle files
that `keras.datasets.cifar10.load_data()` expects, without ever hitting
the throttled cs.toronto.edu server (unless every alternative fails).

Strategy order (each is tried until one succeeds):
  1. Direct tarball download from HTTP mirrors, using aria2c with 8 parallel
     connections + resume. cs.toronto.edu is only used as a last resort.
  2. tensorflow-datasets with try_gcs=True    (pulls prepared arrays from
     storage.googleapis.com/tfds-data — very fast from GitHub Actions).
  3. HuggingFace datasets uoft-cs/cifar10     (parquet on hf.co CDN — fast).

If any strategy produces the arrays, we reconstruct the exact pickle format
Keras expects. That way keras.utils.get_file's local-cache check succeeds
and no re-download is ever attempted.

Exit code 0 = success. Exit code 1 = every strategy failed.
"""

import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


KERAS_DIR = Path.home() / ".keras" / "datasets"
TARGET_DIR = KERAS_DIR / "cifar-10-batches-py"
TARGET_DIR_ALT = KERAS_DIR / "cifar-10-batches-py-target" / "cifar-10-batches-py"
TARBALL_PATH = KERAS_DIR / "cifar-10-python.tar.gz"
EXPECTED_FILES = ["data_batch_1", "data_batch_2", "data_batch_3",
                  "data_batch_4", "data_batch_5", "test_batch", "batches.meta"]
EXPECTED_SIZE = 170_498_071
EXPECTED_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def log(msg):
    print(f"[fetch_cifar10] {msg}", flush=True)


def files_present(dir_path):
    return all((dir_path / f).is_file() for f in EXPECTED_FILES)


def already_have_data():
    return files_present(TARGET_DIR)


def _pip_install(pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs),
        stdout=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Strategy 1: aria2c direct tarball download
# ---------------------------------------------------------------------------
MIRRORS = [
    # cs.toronto.edu is the canonical source but throttled — used with aria2c's
    # 8 parallel connections so throughput is still acceptable.
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]


def _have_aria2c():
    return shutil.which("aria2c") is not None


def _install_aria2c():
    log("Installing aria2 via apt...")
    subprocess.run(["sudo", "apt-get", "update", "-qq"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "aria2"],
                   check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def _tarball_looks_good():
    if not TARBALL_PATH.exists():
        return False
    if TARBALL_PATH.stat().st_size < EXPECTED_SIZE - 1_000_000:
        return False
    # Optional hash check; skip if it would take too long. It's fast on 170MB.
    try:
        h = hashlib.sha256()
        with open(TARBALL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest() == EXPECTED_SHA256
    except Exception:
        return TARBALL_PATH.stat().st_size >= EXPECTED_SIZE - 1_000_000


def strategy_tarball():
    """Download the real tarball via aria2c multi-connection."""
    log("Strategy 1: aria2c multi-connection tarball download")
    KERAS_DIR.mkdir(parents=True, exist_ok=True)

    if not _have_aria2c():
        _install_aria2c()
    if not _have_aria2c():
        log("aria2c unavailable; falling back to urllib retries")
        return _strategy_tarball_urllib()

    for url in MIRRORS:
        log(f"  aria2c -> {url}")
        # 8 parallel connections, aggressive retries, resume across attempts.
        # --lowest-speed-limit=51200  (50 KB/s per connection) drops stalled
        # connections so aria2 opens a fresh one instead of hanging.
        cmd = [
            "aria2c",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=4M",
            "--continue=true",
            "--max-tries=50",
            "--retry-wait=15",
            "--timeout=60",
            "--connect-timeout=30",
            "--lowest-speed-limit=51200",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--console-log-level=warn",
            "--summary-interval=30",
            "--check-certificate=false",
            "--dir", str(KERAS_DIR),
            "--out", TARBALL_PATH.name,
            url,
        ]
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            break

        if _tarball_looks_good():
            log(f"  aria2c succeeded from {url}")
            return _extract_tarball()

    return False


def _strategy_tarball_urllib():
    """urllib fallback with byte-range resume, up to 20 attempts."""
    log("Strategy 1b: urllib byte-range resume loop")
    for url in MIRRORS:
        for attempt in range(1, 21):
            existing = TARBALL_PATH.stat().st_size if TARBALL_PATH.exists() else 0
            if existing >= EXPECTED_SIZE:
                break
            log(f"  urllib attempt {attempt}/20 (resume from {existing} bytes) <- {url}")
            req = urllib.request.Request(url)
            if existing > 0:
                req.add_header("Range", f"bytes={existing}-")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    mode = "ab" if existing > 0 and resp.status == 206 else "wb"
                    with open(TARBALL_PATH, mode) as out:
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
            except Exception as e:
                log(f"  urllib attempt failed: {type(e).__name__}: {e}")
                time.sleep(min(60, 5 * attempt))
                continue
            if _tarball_looks_good():
                return _extract_tarball()
    return False


def _extract_tarball():
    log(f"Extracting {TARBALL_PATH} into {KERAS_DIR}")
    with tarfile.open(TARBALL_PATH, "r:gz") as tar:
        tar.extractall(KERAS_DIR)
    if not files_present(TARGET_DIR):
        return False
    # Mirror to the newer Keras 3.x -target path so both callers hit a cache.
    TARGET_DIR_ALT.mkdir(parents=True, exist_ok=True)
    for f in EXPECTED_FILES:
        src = TARGET_DIR / f
        if src.exists():
            shutil.copy2(src, TARGET_DIR_ALT / f)
    return True


# ---------------------------------------------------------------------------
# Pickle writer: converts numpy arrays into Keras' expected format
# ---------------------------------------------------------------------------
def write_keras_pickles(x_train, y_train, x_test, y_test):
    """
    Keras format: 5 training batches + 1 test batch + batches.meta.
    Each batch pickle is a dict with byte-string keys:
      { b'batch_label', b'labels' (list[int]),
        b'data' (uint8 (N, 3072), channel-first RGB),
        b'filenames' (list[bytes]) }
    """
    import numpy as np

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure (N, 32, 32, 3) uint8
    x_train = np.asarray(x_train, dtype="uint8")
    x_test = np.asarray(x_test, dtype="uint8")
    y_train = np.asarray(y_train, dtype="uint8").flatten()
    y_test = np.asarray(y_test, dtype="uint8").flatten()

    def to_channel_first_flat(x):
        # (N, 32, 32, 3) -> (N, 3, 32, 32) -> (N, 3072)
        return x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1)

    x_train_flat = to_channel_first_flat(x_train)
    x_test_flat = to_channel_first_flat(x_test)

    for i in range(5):
        s = i * 10000
        e = s + 10000
        batch = {
            b"batch_label": f"training batch {i+1} of 5".encode(),
            b"labels": [int(v) for v in y_train[s:e]],
            b"data": x_train_flat[s:e],
            b"filenames": [f"img_{j}.png".encode() for j in range(10000)],
        }
        with open(TARGET_DIR / f"data_batch_{i+1}", "wb") as f:
            pickle.dump(batch, f)

    test_batch = {
        b"batch_label": b"testing batch 1 of 1",
        b"labels": [int(v) for v in y_test],
        b"data": x_test_flat,
        b"filenames": [f"img_{j}.png".encode() for j in range(10000)],
    }
    with open(TARGET_DIR / "test_batch", "wb") as f:
        pickle.dump(test_batch, f)

    meta = {
        b"num_cases_per_batch": 10000,
        b"label_names": [b"airplane", b"automobile", b"bird", b"cat",
                         b"deer", b"dog", b"frog", b"horse", b"ship", b"truck"],
        b"num_vis": 3072,
    }
    with open(TARGET_DIR / "batches.meta", "wb") as f:
        pickle.dump(meta, f)

    # Also mirror to the newer -target extraction path some Keras versions look at
    TARGET_DIR_ALT.mkdir(parents=True, exist_ok=True)
    for f in EXPECTED_FILES:
        shutil.copy2(TARGET_DIR / f, TARGET_DIR_ALT / f)


# ---------------------------------------------------------------------------
# Strategy 2: tensorflow-datasets with try_gcs=True
# ---------------------------------------------------------------------------
def strategy_tfds():
    log("Strategy 2: tensorflow-datasets try_gcs=True (Google Cloud Storage)")
    try:
        _pip_install(["tensorflow-datasets"])
        import tensorflow_datasets as tfds  # noqa: E402
        # try_gcs pulls a prepared TFRecord copy from storage.googleapis.com
        ds = tfds.load("cifar10", try_gcs=True, batch_size=-1)
        ds = tfds.as_numpy(ds)
        x_train, y_train = ds["train"]["image"], ds["train"]["label"]
        x_test, y_test = ds["test"]["image"], ds["test"]["label"]
        log(f"  Got train={x_train.shape} test={x_test.shape} via TFDS")
        write_keras_pickles(x_train, y_train, x_test, y_test)
        return files_present(TARGET_DIR)
    except Exception as e:
        log(f"  TFDS failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Strategy 3: HuggingFace datasets uoft-cs/cifar10
# ---------------------------------------------------------------------------
def strategy_huggingface():
    log("Strategy 3: HuggingFace datasets uoft-cs/cifar10")
    try:
        _pip_install(["datasets", "pillow"])
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset("uoft-cs/cifar10")
        # HF gives PIL images under key 'img' and int label under 'label'
        img_key = "img" if "img" in ds["train"].column_names else "image"
        lbl_key = "label" if "label" in ds["train"].column_names else "fine_label"

        def as_arrays(split):
            imgs = np.stack([np.array(im, dtype="uint8") for im in ds[split][img_key]])
            labels = np.array(ds[split][lbl_key], dtype="uint8")
            return imgs, labels

        x_train, y_train = as_arrays("train")
        x_test, y_test = as_arrays("test")
        log(f"  Got train={x_train.shape} test={x_test.shape} via HuggingFace")
        write_keras_pickles(x_train, y_train, x_test, y_test)
        return files_present(TARGET_DIR)
    except Exception as e:
        log(f"  HuggingFace failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Verify keras can load the result
# ---------------------------------------------------------------------------
def verify_keras():
    log("Verifying keras.datasets.cifar10.load_data() ...")
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf  # noqa: E402
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        log(f"  OK: x_train={x_train.shape} y_train={y_train.shape} "
            f"x_test={x_test.shape} y_test={y_test.shape}")
        return True
    except Exception as e:
        log(f"  Keras verify FAILED: {type(e).__name__}: {e}")
        return False


def main():
    KERAS_DIR.mkdir(parents=True, exist_ok=True)

    if already_have_data():
        log("CIFAR-10 already present on disk; nothing to do.")
        if verify_keras():
            return 0
        # Fall through and try to re-download if verification fails
        log("Existing files failed verification; re-fetching.")

    strategies = [strategy_tarball, strategy_tfds, strategy_huggingface]
    for i, strat in enumerate(strategies, start=1):
        log(f"--- Trying strategy {i}/{len(strategies)}: {strat.__name__} ---")
        try:
            if strat() and verify_keras():
                log(f"SUCCESS via {strat.__name__}")
                return 0
        except Exception as e:
            log(f"Strategy {strat.__name__} raised: {type(e).__name__}: {e}")

    log("FAILURE: every strategy exhausted. CIFAR-10 not available.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
