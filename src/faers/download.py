"""Stage 1: fetch quarterly FAERS/LAERS archives and record cryptographic checksums.

FDA republishes quarterly extracts as database *snapshots*, not immutable artifacts: cases can be
revised, merged or withdrawn between releases, and the bytes at a given URL can change. A pinned
``dvc.lock`` hash alone therefore does not guarantee a future re-run sees the same input. We record
a SHA-256 per archive in a committed manifest so that a drift is detected loudly instead of
silently changing every downstream number.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .sources import Quarter

CHUNK = 1 << 20
USER_AGENT = "faers-pipeline/2.0 (+https://github.com/neksa/openfda-faers)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(q: Quarter, dest_dir: Path, retries: int = 4, timeout: int = 120) -> Path:
    """Download one quarterly archive, atomically. Returns the archive path.

    Skips the transfer when a valid archive is already present, which keeps ``dvc repro`` cheap
    after a partial failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{q.label}.zip"
    if dest.exists() and _is_valid_zip(dest):
        return dest

    tmp = dest.with_suffix(".zip.part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(
                q.url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
            ) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
            if not _is_valid_zip(tmp):
                raise OSError(f"{q.label}: downloaded bytes are not a readable ZIP")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {q.label} from {q.url}") from last


def _is_valid_zip(path: Path) -> bool:
    """A truncated download is the common failure; testzip catches CRC damage too."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download_all(
    quarters: list[Quarter], dest_dir: Path, manifest_path: Path, workers: int = 4
) -> dict:
    """Fetch every quarter and write the checksum manifest."""
    dest_dir = Path(dest_dir)
    records: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, q, dest_dir): q for q in quarters}
        for fut in as_completed(futures):
            q = futures[fut]
            path = fut.result()
            records[q.label] = {
                "url": q.url,
                "era": q.era.value,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            print(f"  {q.label:8s} {records[q.label]['bytes']:>12,} B  {path.name}", flush=True)

    manifest = {
        "n_quarters": len(records),
        "total_bytes": sum(r["bytes"] for r in records.values()),
        "archives": dict(sorted(records.items())),
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_against_manifest(manifest_path: Path, dest_dir: Path) -> list[str]:
    """Return the labels whose on-disk bytes no longer match the committed manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    drifted = []
    for label, rec in manifest["archives"].items():
        path = Path(dest_dir) / f"{label}.zip"
        if not path.exists() or sha256(path) != rec["sha256"]:
            drifted.append(label)
    return drifted
