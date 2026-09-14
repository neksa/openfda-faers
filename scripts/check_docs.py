"""Fail if a number quoted in the docs no longer matches the pipeline's own output.

Headline figures get hardcoded into README prose and then go stale the next time a stage re-runs.
That happened: the duplicate-cluster count sat at 287k in the README after the salt merge moved it
to 292k. The report does not have this problem because it computes its numbers from the result
files at render time; the README cannot, so it is checked instead.

Skips silently when results/ is absent, so CI can run this without a 3.5 GB pipeline run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path("results")


def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    ded, sig, dup = (
        load("dedup_stats.json"),
        load("signals/summary.json"),
        load("duplicates_summary.json"),
    )
    if not all((ded, sig, dup)):
        print("results/ incomplete - skipping doc consistency check")
        return 0

    readme = Path("README.md").read_text()
    expected = {
        "dedup redundancy": f'{ded["duplicate_fraction_of_raw"]:.1%}',
        "scored pairs": f'{sig["n_pairs"] / 1e6:.2f}M pairs scored',
        "eb05 signals": f'{sig["signals"]["signal_eb05"]:,}',
        "duplicate clusters": f'{dup["clusters"] / 1000:.0f}k clusters',
        "duplicate rate": f'{dup["redundant_fraction_of_eligible"]:.1%} of linkable',
    }

    stale = {k: v for k, v in expected.items() if v not in readme}
    for key, value in expected.items():
        print(f"  {'ok ' if key not in stale else 'STALE'}  {key}: {value}")

    if stale:
        print("\nREADME quotes figures that no longer match results/:")
        for key, value in stale.items():
            print(f"  {key}: expected to find {value!r}")
        return 1

    # The report must not carry hand-written numbers either; every figure there is computed.
    report = Path("report/index.qmd").read_text()
    if re.search(r"^\s*\|\s*[\d,]{4,}\s*\|", report, re.M):
        print("\nreport/index.qmd contains a hardcoded numeric table row; compute it instead")
        return 1

    print("\ndocs agree with results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
