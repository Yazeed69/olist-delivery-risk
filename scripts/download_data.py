"""Download the public Olist dataset and verify every required source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw_checksums.json"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "olistbr/brazilian-ecommerce"
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checksums() -> dict[str, str]:
    """Load the committed source-data integrity manifest."""
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)["sha256"]


def verify_raw_data(raw_dir: Path = RAW_DIR) -> None:
    """Fail if a required raw file is missing or has unexpected contents."""
    errors: list[str] = []
    for filename, expected in load_checksums().items():
        path = raw_dir / filename
        if not path.is_file():
            errors.append(f"missing: {filename}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"checksum mismatch: {filename} ({actual})")
    if errors:
        raise RuntimeError("Raw-data verification failed:\n- " + "\n- ".join(errors))


def download_raw_data(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    """Download and atomically copy the seven required CSV files."""
    checksums = load_checksums()
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in checksums if (raw_dir / name).exists()]
    if existing and not force:
        verify_raw_data(raw_dir)
        print("Raw data already exists and all checksums match.")
        return

    with tempfile.TemporaryDirectory(prefix="olist-download-") as temp_name:
        temp_dir = Path(temp_name)
        archive = temp_dir / "olist.zip"
        print(f"Downloading {DOWNLOAD_URL}")
        urllib.request.urlretrieve(DOWNLOAD_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            for filename in checksums:
                member = next(
                    (item for item in zipped.namelist() if Path(item).name == filename),
                    None,
                )
                if member is None:
                    raise RuntimeError(f"Downloaded archive is missing {filename}")
                extracted = Path(zipped.extract(member, temp_dir))
                if sha256(extracted) != checksums[filename]:
                    raise RuntimeError(f"Checksum mismatch in download: {filename}")
                shutil.copy2(extracted, raw_dir / filename)

    verify_raw_data(raw_dir)
    print(f"Verified {len(checksums)} files in {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing raw files after verifying the download",
    )
    args = parser.parse_args()
    if args.verify_only:
        verify_raw_data()
        print(f"Verified {len(load_checksums())} files in {RAW_DIR}")
    else:
        download_raw_data(force=args.force)


if __name__ == "__main__":
    main()
