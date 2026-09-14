"""FAERS/LAERS quarterly source index.

The FDA publishes the Adverse Event Reporting System as quarterly ZIP archives of ``$``-delimited
ASCII tables. There are two eras with materially different schemas:

``LAERS``  2004Q1-2012Q3   URL stem ``aers_ascii_``   keyed by ``ISR``, case grouped by ``CASE``
``FAERS``  2012Q4-present  URL stem ``faers_ascii_``  keyed by
           ``primaryid``/``caseid``/``caseversion``

Everything downstream keys off :func:`quarters`, so extending coverage is a params.yaml edit.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

BASE_URL = "https://fis.fda.gov/content/Exports"

#: The final quarter published in the legacy LAERS format.
LAERS_LAST = (2012, 3)

#: Core tables. STAT (LAERS-only) and the PDF documentation members are ignored.
TABLES = ("DEMO", "DRUG", "REAC", "OUTC", "INDI", "THER", "RPSR")


class Era(StrEnum):
    LAERS = "laers"
    FAERS = "faers"


@dataclass(frozen=True, slots=True)
class Quarter:
    year: int
    quarter: int

    @property
    def era(self) -> Era:
        return Era.LAERS if (self.year, self.quarter) <= LAERS_LAST else Era.FAERS

    @property
    def label(self) -> str:
        """Canonical identifier used in filenames and partition columns, e.g. ``2026Q1``."""
        return f"{self.year}Q{self.quarter}"

    @property
    def yy(self) -> str:
        """Two-digit year as it appears inside archive member names."""
        return f"{self.year % 100:02d}"

    @property
    def url(self) -> str:
        stem = "aers_ascii" if self.era is Era.LAERS else "faers_ascii"
        return f"{BASE_URL}/{stem}_{self.year}q{self.quarter}.zip"

    @property
    def index(self) -> int:
        """Monotonic quarter counter, used for ordering across the era boundary."""
        return self.year * 4 + (self.quarter - 1)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


def parse_quarter(text: str) -> Quarter:
    """Parse ``2026Q1`` / ``2026q1`` into a :class:`Quarter`."""
    m = re.fullmatch(r"(\d{4})[Qq]([1-4])", text.strip())
    if not m:
        raise ValueError(f"not a quarter label: {text!r}")
    return Quarter(int(m.group(1)), int(m.group(2)))


def quarters(start: str, end: str) -> list[Quarter]:
    """Inclusive list of quarters between two labels."""
    a, b = parse_quarter(start), parse_quarter(end)
    if a.index > b.index:
        raise ValueError(f"start {a} is after end {b}")
    out: list[Quarter] = []
    y, q = a.year, a.quarter
    while (y, q) <= (b.year, b.quarter):
        out.append(Quarter(y, q))
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def find_member(zf: zipfile.ZipFile, table: str, q: Quarter) -> str | None:
    """Locate a table inside a quarterly archive.

    Member naming is inconsistent across 22 years: the directory is ``ascii/`` or ``ASCII/`` and the
    extension ``.TXT`` or ``.txt``, with occasional stray members (``Thumbs.db``, PDFs, and in a few
    quarters a duplicate copy under a nested folder). Match on the basename case-insensitively and
    prefer the shallowest path so a stray nested duplicate never wins.
    """
    want = f"{table}{q.yy}Q{q.quarter}.txt".lower()
    hits = [
        n
        for n in zf.namelist()
        if not n.endswith("/") and Path(n).name.lower() == want
    ]
    if not hits:
        return None
    return min(hits, key=lambda n: (n.count("/"), len(n)))


def find_delete_member(zf: zipfile.ZipFile, q: Quarter) -> str | None:
    """Locate the FDA case-retraction list, present in the FAERS era only.

    These files name cases FDA has withdrawn; they must be removed from the corpus. Naming drifts
    across quarters (``DELETE26Q1.txt``, ``ADR12Q4.TXT``), so match on the prefix rather than an
    exact name.
    """
    hits = [
        n
        for n in zf.namelist()
        if not n.endswith("/")
        and Path(n).name.lower().startswith("delete")
        and Path(n).suffix.lower() == ".txt"
    ]
    if not hits:
        return None
    return min(hits, key=lambda n: (n.count("/"), len(n)))
