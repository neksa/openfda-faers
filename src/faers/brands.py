"""Brand and product name resolution, from free public sources only.

FAERS reports what the reporter wrote, which is very often a product rather than a substance:
``HUMIRA . PEN``, ``DURAGESIC-100``, ``PARAGARD 380A``, ``ADVAIR DISKUS 100/50``. FDA's own
``prod_ai`` field covers the FAERS era but leaves a large minority unmapped, and those survive into
the results under their product name. That is damaging in two directions at once -- the substance's
marginal count is understated because reports filed under the brand never reach it, and the brand
fragment earns an inflated disproportionality because it is compared against the whole database on
a small marginal.

Two free sources close most of the gap:

* **FDA NDC directory** (openFDA bulk, public domain). 137k marketed products with active
  ingredients. Matching is not straightforward: NDC brand strings are typically *longer* than the
  FAERS string (``ParaGard T 380A`` against ``PARAGARD 380A``), so neither exact direction works.
  A first-token index does, with a guard described below.
* **RxNav** (NLM, no API key). Covers brands the NDC directory misses, notably biologics whose NDC
  entries carry no ``active_ingredients``.

Neither covers discontinued brands -- ``DURAGESIC`` is in neither -- so a residue always remains.
That residue is *labelled*, never silently presented as an ingredient.

The unambiguity guard is what makes first-token matching safe. ``SODIUM CHLORIDE`` shares its first
token with 904 NDC records spanning many different formulations; accepting the most common one
would be a fabrication. A token resolves only when one ingredient set dominates its records, so
``SODIUM`` and ``NEXIUM`` are declined while ``PARAGARD``, ``LIPITOR`` and ``OXYCONTIN`` resolve.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

NDC_URL = "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip"
RXNAV = "https://rxnav.nlm.nih.gov/REST"

#: A token resolves only if one ingredient set covers at least this share of its NDC records.
#: Tuned against known-ambiguous cases: SODIUM (29%) and NEXIUM (73%) are declined, while ASPIRIN
#: (97%), PARAGARD, LIPITOR and OXYCONTIN (100%) resolve.
DOMINANCE_THRESHOLD = 0.80

#: Tokens too generic to carry brand identity, regardless of dominance.
STOPWORD_TOKENS = frozenset(
    {
        "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "WATER", "DEXTROSE", "GLUCOSE",
        "VITAMIN", "ACID", "OIL", "ALCOHOL", "SALINE", "STERILE", "COMPOUND", "SOLUTION",
        "GENERIC", "MEDICAL", "HEALTH", "CARE", "PHARMA", "LABORATORIES", "THE", "AND",
    }
)


def _tokens(text: str | None) -> list[str]:
    return re.sub(r"[^A-Z0-9 ]", " ", (text or "").upper()).split()


def download_ndc(dest: Path) -> Path:
    """Fetch the NDC directory archive, atomically."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            with zipfile.ZipFile(dest) as zf:
                if zf.testzip() is None:
                    return dest
        except (zipfile.BadZipFile, OSError):
            dest.unlink(missing_ok=True)

    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(NDC_URL, headers={"User-Agent": "faers-pipeline/2.0"})
    with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def build_ndc_index(ndc_zip: Path) -> pl.DataFrame:
    """First-token -> dominant active ingredient set, with the evidence behind each decision."""
    with zipfile.ZipFile(ndc_zip) as zf:
        payload = json.loads(zf.read(zf.namelist()[0]))

    counts: dict[str, dict[str, int]] = {}
    for rec in payload["results"]:
        ingredients = sorted(
            {
                a["name"].strip().upper()
                for a in (rec.get("active_ingredients") or [])
                if a.get("name")
            }
        )
        if not ingredients:
            continue  # biologics frequently have none; RxNav picks these up
        key = "\\".join(ingredients)
        # Index both the marketed brand and its base form; they differ often enough to matter.
        for field in (rec.get("brand_name"), rec.get("brand_name_base")):
            toks = _tokens(field)
            if toks:
                counts.setdefault(toks[0], {}).setdefault(key, 0)
                counts[toks[0]][key] += 1

    rows = []
    for token, by_key in counts.items():
        total = sum(by_key.values())
        best_key, best_n = max(by_key.items(), key=lambda kv: (kv[1], kv[0]))
        rows.append(
            {
                "token": token,
                "ingredients": best_key,
                "support": best_n,
                "records": total,
                "dominance": best_n / total,
            }
        )

    df = pl.DataFrame(rows).sort("token")
    return df.with_columns(
        (
            (pl.col("dominance") >= DOMINANCE_THRESHOLD)
            & ~pl.col("token").is_in(list(STOPWORD_TOKENS))
            & (pl.col("token").str.len_chars() >= 3)
        ).alias("usable")
    )


def _rxnav_json(path: str, **params) -> dict | None:
    url = f"{RXNAV}/{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - a lookup failure must not abort the stage
        return None


def rxnav_lookup(name: str) -> dict | None:
    """Resolve one brand name to its active ingredients and their RxNorm identifiers.

    Two hops: the brand to an RxNorm concept, then that concept to its ``IN``/``MIN`` ingredients.
    Returns the backslash-joined ingredient set (matching FDA's ``prod_ai`` convention) alongside
    the corresponding RxCUIs.

    The RxCUIs are retained rather than discarded because they turn the published drug column into
    a joinable identifier instead of free text -- the convention OFFSIDES/TWOSIDES follow with
    ``drug_rxnorm_id`` beside ``drug_concept_name``. The lookup happens either way, so keeping the
    identifier is free.
    """
    drugs = _rxnav_json("drugs.json", name=name)
    rxcui = None
    for group in ((drugs or {}).get("drugGroup") or {}).get("conceptGroup") or []:
        for prop in group.get("conceptProperties") or []:
            rxcui = prop.get("rxcui")
            break
        if rxcui:
            break
    if not rxcui:
        return None

    # Space-separated, not "IN+MIN": urlencode escapes a literal "+" to %2B, which RxNav reads as
    # one nonexistent term type rather than two. A space encodes to "+" on the wire, which is what
    # the API actually wants. This silently returned zero ingredients for every brand.
    related = _rxnav_json(f"rxcui/{rxcui}/related.json", tty="IN MIN")
    props = [
        p
        for g in ((related or {}).get("relatedGroup") or {}).get("conceptGroup") or []
        for p in g.get("conceptProperties") or []
        if p.get("name")
    ]
    if not props:
        return None

    by_name: dict[str, str] = {}
    for p in props:
        by_name.setdefault(p["name"].strip().upper(), str(p.get("rxcui") or ""))
    names = sorted(by_name)
    return {"ingredients": "\\".join(names), "rxcuis": "\\".join(by_name[n] for n in names)}


def rxnav_ingredients(name: str) -> str | None:
    """Ingredient string only, for callers that do not need the identifiers."""
    found = rxnav_lookup(name)
    return found["ingredients"] if found else None


def _migrate_cache_entry(value):
    """Accept both cache shapes.

    Earlier runs stored a bare ingredient string; entries now carry RxCUIs too. Migrating on read
    preserves several thousand cached lookups that would otherwise be re-queried from NLM one name
    at a time.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return {"ingredients": value, "rxcuis": ""}
    return value


def rxcui_for_ingredient(name: str) -> dict | None:
    """RxNorm identifier and canonical base ingredient for an ingredient name.

    Distinct from :func:`rxnav_lookup`, which starts from a *brand*. Most ingredients here arrive
    via FDA's ``prod_ai`` and never touch a brand lookup, so without this pass the identifier column
    covers only the handful resolved from brands.

    The walk to ``tty=IN`` is what collapses salt forms. FAERS records both ``ATORVASTATIN`` and
    ``ATORVASTATIN CALCIUM``, and treating them as two drugs splits one substance almost evenly in
    half -- 275k reports against 269k -- understating its marginal and inflating the
    disproportionality of both halves. RxNorm resolves the salt to its active moiety, and does so
    without a heuristic: substances where the salt *is* the drug (``SODIUM CHLORIDE``,
    ``CALCIUM CARBONATE``, ``LITHIUM CARBONATE``) come back unchanged, which a suffix-stripping rule
    would have mangled into ``SODIUM`` and ``CALCIUM``.
    """
    got = _rxnav_json("rxcui.json", name=name, search=1)
    ids = ((got or {}).get("idGroup") or {}).get("rxnormId") or []
    if not ids:
        return None
    rxcui = str(ids[0])

    related = _rxnav_json(f"rxcui/{rxcui}/related.json", tty="IN")
    bases = [
        p
        for g in ((related or {}).get("relatedGroup") or {}).get("conceptGroup") or []
        for p in g.get("conceptProperties") or []
        if p.get("name")
    ]
    if not bases:
        # Already a base ingredient, or has no IN relation; keep the name as reported.
        return {"rxcui": rxcui, "base": name, "base_rxcui": rxcui}

    # A single IN is the normal case. Several means a combination product, which must not be
    # collapsed onto one of its components.
    if len(bases) > 1:
        return {"rxcui": rxcui, "base": name, "base_rxcui": rxcui}

    return {
        "rxcui": rxcui,
        "base": bases[0]["name"].strip().upper(),
        "base_rxcui": str(bases[0].get("rxcui") or rxcui),
    }


def _migrate_rxcui_entry(value):
    """Earlier caches stored a bare RxCUI string with no base-ingredient resolution."""
    if value is None:
        return None
    if isinstance(value, str):
        return None  # force a re-query: the base ingredient was never fetched
    return value


def resolve_ingredient_rxcuis(
    names: list[str], cache_path: Path, pause: float = 0.06
) -> dict[str, dict]:
    """Resolve ingredient names to RxCUIs and canonical base ingredients."""
    cache_path = Path(cache_path)
    cache: dict[str, dict | None] = {}
    if cache_path.exists():
        cache = {
            k: _migrate_rxcui_entry(v)
            for k, v in json.loads(cache_path.read_text()).items()
        }

    def flush():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))

    pending = [n for n in names if cache.get(n) is None]
    for i, name in enumerate(pending, 1):
        cache[name] = rxcui_for_ingredient(name)
        time.sleep(pause)
        if i % 500 == 0:
            flush()
            print(f"    rxcui {i}/{len(pending)}", flush=True)

    flush()
    return {k: v for k, v in cache.items() if v}


def resolve_via_rxnav(
    tokens: list[str], cache_path: Path, pause: float = 0.06
) -> dict[str, dict]:
    """Resolve tokens RxNav can map, caching every answer including the misses.

    Caching negatives matters as much as positives: without it, every re-run re-queries thousands
    of names NLM has already told us it does not know.
    """
    cache_path = Path(cache_path)
    cache: dict[str, dict | None] = {}
    if cache_path.exists():
        cache = {k: _migrate_cache_entry(v) for k, v in json.loads(cache_path.read_text()).items()}

    def flush():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))

    # Tokens whose cached entry predates RxCUI capture are re-queried so the identifiers get
    # filled in; genuine negatives (None) are left alone.
    pending = [
        t
        for t in tokens
        if t not in cache or (cache[t] is not None and not cache[t].get("rxcuis"))
    ]
    for i, token in enumerate(pending, 1):
        cache[token] = rxnav_lookup(token)
        time.sleep(pause)  # NLM asks for <= 20 requests/second; this stays well under
        if i % 250 == 0:
            flush()
            print(f"    rxnav {i}/{len(pending)}", flush=True)

    flush()
    return {k: v for k, v in cache.items() if v}
