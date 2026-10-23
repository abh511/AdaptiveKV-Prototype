"""
benchmarks/download_data.py
----------------------------
Download WikiText-2 test split and save as plain text.
Records the source URL and SHA256 of the raw file.
"""

import hashlib
import os
import sys

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_FILE  = os.path.join(DATA_DIR, "wikitext2_test.txt")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Use HuggingFace datasets library — the standard source for WikiText-2
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    print("Downloading WikiText-2 test split via HuggingFace datasets...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    print(f"  Loaded {len(ds)} rows")

    # Join all text rows, keep only non-empty lines, separate by newline
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    size = os.path.getsize(OUT_FILE)
    digest = sha256_file(OUT_FILE)
    print(f"  Saved to: {OUT_FILE}")
    print(f"  File size: {size:,} bytes ({size / 1024:.1f} KB)")
    print(f"  SHA256:    {digest}")
    print(f"  Characters: {len(text):,}")
    return OUT_FILE


if __name__ == "__main__":
    download()
