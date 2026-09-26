"""Field-level normalization (plan §3.1–3.3).

Input : raw TSV rows (train / test × s1 / s2 / s3).
Output: per (split, source, country) parquet under artifacts/normalized/ with
        raw fields + all derived fields.

The pipeline runs cleanly with only anyascii; indic-transliteration is used
where available for higher-quality Indic romanization. `unidecode` is
deliberately not imported (GPL — see plan §0).
"""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from . import abbreviations as abbrev_mod
from .io_utils import read_source_tsv, write_parquet
from .stage0_checks import _script_of

try:
    from anyascii import anyascii  # type: ignore
except Exception:  # pragma: no cover
    def anyascii(x: str) -> str:  # fallback: strip diacritics only
        return "".join(c for c in unicodedata.normalize("NFKD", x)
                       if not unicodedata.combining(c))

try:
    from indic_transliteration import sanscript  # type: ignore
    from indic_transliteration.sanscript import transliterate  # type: ignore
    _INDIC = {
        "devanagari": sanscript.DEVANAGARI,
        "tamil": sanscript.TAMIL,
        "bengali": sanscript.BENGALI,
        "telugu": sanscript.TELUGU,
        "gujarati": sanscript.GUJARATI,
        "kannada": sanscript.KANNADA,
        "malayalam": sanscript.MALAYALAM,
        "gurmukhi": sanscript.GURMUKHI,
    }
    _HAS_INDIC = True
except Exception:  # pragma: no cover
    _HAS_INDIC = False


# ---------------------------------------------------------------------------
# Regex battery
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?([A-Za-z0-9\-]+\.[A-Za-z]{2,})(?:/\S*)?",
                    re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b([A-Za-z0-9\-]+\.(?:com|net|org|in|fr|co|io|us))\b",
                       re.IGNORECASE)
PIPE_TRAIL_RE = re.compile(r"\s*\|.*$")
JUNK_PREFIX_RE = re.compile(r"^[\W_]+")
JUNK_SUFFIX_RE = re.compile(r"[\W_]+$")
MULTIWS_RE = re.compile(r"\s+")
PUNCT_KEEP = re.compile(r"[^\w\s/\-.,&@]")
POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
POSTAL_SPACED_RE = re.compile(r"(?<!\d)(\d{3}\s+\d{3})(?!\d)")
HOUSE_LABEL_RE = re.compile(
    r"\b(?:h\.?\s*no|hno|plot\s*no|shp\s*no|shop\s*no|kh\.?\s*no|s\.?\s*no|"
    r"door|flat)\b\.?\s*[:.-]?\s*([\dA-Za-z][\dA-Za-z/\-]*)",
    re.IGNORECASE,
)
LEADING_NUMBER_RE = re.compile(r"^\s*([\dA-Za-z][\dA-Za-z/\-]*)\s+(?=[A-Za-z])")
LANDMARK_RE = re.compile(
    r"\b(?:near|opposite|opp\.?|behind|beside|next\s+to|in\s+front\s+of|landmark)\b"
    r"[^,]*",
    re.IGNORECASE,
)
NUMBER_TOK_RE = re.compile(r"\b\d[\dA-Za-z/\-]*\b")
US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}
INDIAN_STATE_TOKENS = {
    "maharashtra","gujarat","karnataka","tamilnadu","telangana","andhra",
    "kerala","punjab","haryana","rajasthan","uttar","pradesh","madhya","bihar",
    "west","bengal","odisha","assam","jharkhand","chhattisgarh","goa","delhi",
    "hyderabad","mumbai","pune","bangalore","chennai","kolkata",
}

LEGAL_SUFFIX_TOKENS = {
    # EN / US / IN
    "llc","inc","incorporated","corp","corporation","company","co","ltd",
    "limited","private","pvt","llp","lp","plc","holdings","enterprises",
    # FR
    "sarl","sas","sasu","sa","sci","eurl","snc","societe","cie",
}
GENERIC_ADDRESS_WORDS = {
    "road","street","avenue","boulevard","drive","lane","highway","floor",
    "building","apartment","plot","house","number","near","opposite","sector",
    "block","phase","chowk","market","complex","tower","enclave","colony",
}


# ---------------------------------------------------------------------------
# Field-level cleanup (plan §3.1)
# ---------------------------------------------------------------------------
def _nfkc(s) -> str:
    if not isinstance(s, str) or not s:
        return ""
    return unicodedata.normalize("NFKC", s)


def _romanize(s: str, script: str) -> tuple[str, bool]:
    """Return (romanized_str, success_flag). Latin passes through."""
    if not s:
        return "", True
    if script == "latin":
        return s, True
    if _HAS_INDIC and script in _INDIC:
        try:
            return transliterate(s, _INDIC[script], sanscript.IAST), True
        except Exception:
            pass
    try:
        return anyascii(s), True
    except Exception:
        return s, False


def _strip_accents_lower(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def _strip_junk(s: str) -> str:
    if not s:
        return ""
    s = PIPE_TRAIL_RE.sub("", s)
    s = JUNK_PREFIX_RE.sub("", s)
    s = JUNK_SUFFIX_RE.sub("", s)
    return s


def _basic_clean(s: str) -> str:
    if not s:
        return ""
    s = _nfkc(s)
    s = _strip_junk(s)
    s = s.replace("&", " and ").replace("@", " at ")
    s = PUNCT_KEEP.sub(" ", s)
    s = MULTIWS_RE.sub(" ", s).strip()
    return s


def _dedup_repeated_tokens(s: str) -> tuple[str, bool]:
    if not s:
        return "", False
    toks = s.split()
    out: list[str] = []
    changed = False
    for t in toks:
        if out and out[-1] == t:
            changed = True
            continue
        out.append(t)
    return " ".join(out), changed


# ---------------------------------------------------------------------------
# Name processing (plan §3.2)
# ---------------------------------------------------------------------------
def _extract_domain(name: str) -> tuple[str, str, str]:
    """Return (name_without_domain, domain_stub, from_domain_token)."""
    if not name:
        return "", "", ""
    m = DOMAIN_RE.search(name)
    if not m:
        return name, "", ""
    dom = m.group(1).lower()
    stub = re.sub(r"^www\.", "", dom)
    stub = stub.rsplit(".", 1)[0]
    without = (name[: m.start()] + " " + name[m.end():]).strip()
    from_dom = stub if not without else ""
    return without or stub, stub, from_dom


def _expand(text: str, mapping: dict[str, str]) -> str:
    if not text or not mapping:
        return text
    toks = text.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        matched = False
        # try 2-token bigrams (for 'h no', 's no', 'shop no', 'kh no', 'plot no')
        if i + 1 < len(toks):
            bg = f"{toks[i]} {toks[i+1]}"
            if bg in mapping:
                out.append(mapping[bg])
                i += 2
                matched = True
        if not matched:
            out.append(mapping.get(toks[i], toks[i]))
            i += 1
    return " ".join(out)


def _split_legal(name_expanded: str) -> tuple[str, str]:
    """Peel trailing legal-form tokens off the right of the name."""
    if not name_expanded:
        return "", ""
    toks = name_expanded.split()
    tail: list[str] = []
    while toks and toks[-1] in LEGAL_SUFFIX_TOKENS:
        tail.insert(0, toks.pop())
    return " ".join(toks), " ".join(sorted(set(tail)))


# ---------------------------------------------------------------------------
# Address processing (plan §3.3)
# ---------------------------------------------------------------------------
def _extract_landmarks(addr: str) -> tuple[str, str]:
    if not addr:
        return "", ""
    landmarks = [m.group(0).strip() for m in LANDMARK_RE.finditer(addr)]
    stripped = LANDMARK_RE.sub(" ", addr)
    stripped = MULTIWS_RE.sub(" ", stripped).strip(" ,")
    return stripped, " | ".join(landmarks)


def _extract_postal(addr: str) -> tuple[str, str, str]:
    if not addr:
        return "", "", ""
    m2 = POSTAL_SPACED_RE.findall(addr)
    if m2:
        code = m2[-1].replace(" ", "")
        return code, code[:3], addr.replace(m2[-1], " ")
    m = POSTAL_RE.findall(addr)
    if m:
        code = m[-1]
        return code, code[:3], POSTAL_RE.sub(" ", addr, count=len(m) - 0)
    return "", "", addr


def _extract_house(addr: str) -> tuple[str, str]:
    if not addr:
        return "", ""
    lab = HOUSE_LABEL_RE.search(addr)
    if lab:
        return _norm_house(lab.group(1)), HOUSE_LABEL_RE.sub(" ", addr, count=1)
    lead = LEADING_NUMBER_RE.search(addr)
    if lead:
        return _norm_house(lead.group(1)), LEADING_NUMBER_RE.sub("", addr, count=1)
    return "", addr


def _norm_house(x: str) -> str:
    return x.replace(" ", "").upper()


def _extract_locality_and_state(addr: str) -> tuple[list[str], str]:
    if not addr:
        return [], ""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    tail = parts[-3:] if len(parts) >= 3 else parts
    locality_tokens: list[str] = []
    state_code = ""
    for seg in tail:
        for tok in seg.split():
            t = tok.lower().strip(".")
            if t.isalpha() and t not in GENERIC_ADDRESS_WORDS:
                locality_tokens.append(t)
        # state code detection
        for tok in seg.split():
            u = tok.upper().strip(".")
            if u in US_STATE_CODES:
                state_code = u
                break
        if not state_code:
            for tok in seg.split():
                if tok.lower() in INDIAN_STATE_TOKENS:
                    state_code = tok.lower()
                    break
    return locality_tokens, state_code


def _street_tokens(addr: str, house: str, postal: str,
                   locality: list[str]) -> list[str]:
    if not addr:
        return []
    toks = [t for t in re.split(r"[\s,]+", addr) if t]
    drop = set(locality) | {house.lower()} | {postal}
    out = []
    for t in toks:
        tl = t.lower().strip(".")
        if not tl or tl in drop or tl in GENERIC_ADDRESS_WORDS:
            continue
        if tl.isdigit():
            continue
        out.append(tl)
    return out


def _all_numbers(addr: str) -> list[str]:
    if not addr:
        return []
    return [m.group(0) for m in NUMBER_TOK_RE.finditer(addr)]


# ---------------------------------------------------------------------------
# Row-level normalization
# ---------------------------------------------------------------------------
def _normalize_row(name: str, address: str, name_map: dict[str, str],
                   addr_map: dict[str, str]) -> dict:
    name = name if isinstance(name, str) else ""
    address = address if isinstance(address, str) else ""

    # 1. NFKC + script detect
    name_nfkc = _nfkc(name)
    addr_nfkc = _nfkc(address)
    name_script = _script_of(name_nfkc)
    addr_script = _script_of(addr_nfkc)

    # 2. Romanize where needed
    name_roman, name_rom_ok = _romanize(name_nfkc, name_script)
    addr_roman, addr_rom_ok = _romanize(addr_nfkc, addr_script)

    # 3. Basic cleanup (accents, junk, punctuation, ws)
    name_clean = _strip_accents_lower(_basic_clean(name_roman))
    addr_clean = _strip_accents_lower(_basic_clean(addr_roman))

    # 4. De-dup repeated tokens
    name_clean, name_had_repeat = _dedup_repeated_tokens(name_clean)
    addr_clean, _ = _dedup_repeated_tokens(addr_clean)

    # Name-specific --------------------------------------------------------
    name_no_dom, name_domain, from_dom_tok = _extract_domain(name_clean)
    name_expanded = _expand(name_no_dom, name_map)
    core_name, legal_suffix = _split_legal(name_expanded)
    expansion_changed = int(name_expanded != name_no_dom)

    # keys
    core_toks = core_name.split()
    acronym = "".join(t[0] for t in core_toks
                      if t not in {"of", "the", "and", "for", "in"})
    name_sorted = " ".join(sorted(core_toks))

    # Address-specific -----------------------------------------------------
    addr_no_lm, landmark_text = _extract_landmarks(addr_clean)
    addr_expanded = _expand(addr_no_lm, addr_map)
    postal_code, postal_prefix3, addr_no_postal = _extract_postal(addr_expanded)
    house_number, addr_no_house = _extract_house(addr_no_postal)
    locality_tokens, state_code = _extract_locality_and_state(addr_no_house)
    street_toks = _street_tokens(addr_no_house, house_number, postal_code,
                                 locality_tokens)
    all_nums = _all_numbers(addr_expanded)

    return {
        "name_script": name_script,
        "address_script": addr_script,
        "name_roman": name_roman,
        "address_roman": addr_roman,
        "name_clean": name_clean,
        "name_expanded": name_expanded,
        "core_name": core_name,
        "legal_suffix": legal_suffix,
        "name_domain": name_domain,
        "name_from_domain": from_dom_tok,
        "acronym": acronym,
        "name_sorted": name_sorted,
        "name_had_repeat": int(name_had_repeat),
        "expansion_changed_name": expansion_changed,
        "romanization_ok": int(name_rom_ok and addr_rom_ok),

        "address_expanded": addr_expanded,
        "address_missing": int(not addr_clean),
        "landmark_text": landmark_text,
        "postal_code": postal_code,
        "postal_prefix3": postal_prefix3,
        "postal_missing": int(not postal_code),
        "house_number": house_number,
        "house_number_missing": int(not house_number),
        "all_numbers": " ".join(all_nums),
        "locality_tokens": " ".join(locality_tokens),
        "locality_missing": int(not locality_tokens),
        "state_code": state_code,
        "street_tokens": " ".join(street_toks),
    }


# ---------------------------------------------------------------------------
# File driver
# ---------------------------------------------------------------------------
def normalize_source(split: str, source: str) -> None:
    """Normalize one source file, write a parquet per country partition."""
    C.ensure_dirs()
    ab = abbrev_mod.load()
    name_map, addr_map = ab.get("name", {}), ab.get("address", {})
    path = C.TRAIN_SOURCE[source] if split == "train" else C.TEST_SOURCE[source]

    print(f"[normalize] {split}/{source} ← {path}")
    df = read_source_tsv(path)
    if C.NORMALIZE_ROW_CAP:
        df = df.iloc[:C.NORMALIZE_ROW_CAP].copy()
        print(f"[normalize] capped to {C.NORMALIZE_ROW_CAP:,} rows (NORMALIZE_ROW_CAP)")
    total = len(df)
    print(f"[normalize] {total:,} rows")

    # Parallel row-level normalization across CPU cores.
    from joblib import Parallel, delayed
    def _chunk(names, addrs):
        return [_normalize_row(n, a, name_map, addr_map) for n, a in zip(names, addrs)]

    chunks = []
    for start in range(0, total, C.CHUNK_SIZE):
        chunk = df.iloc[start:start + C.CHUNK_SIZE]
        chunks.append((chunk["business_name"].tolist(),
                       chunk["business_address"].tolist()))
    print(f"[normalize] {split}/{source}: {len(chunks)} chunks × {C.CHUNK_SIZE:,} rows on {C.N_JOBS} cores")
    parts = Parallel(n_jobs=C.N_JOBS, backend="loky", verbose=5)(
        delayed(_chunk)(n, a) for n, a in chunks
    )
    results: list[dict] = [r for part in parts for r in part]

    derived = pd.DataFrame(results)
    full = pd.concat(
        [df.reset_index(drop=True), derived.reset_index(drop=True)], axis=1
    )
    for country, g in full.groupby("country", sort=False):
        out = C.NORMALIZED_DIR / f"{split}__{source}__{country}.parquet"
        write_parquet(g, out)
        print(f"[normalize] wrote {out}  ({len(g):,} rows)")


def normalize_all() -> None:
    for split in ("train", "test"):
        for src in ("s1", "s2", "s3"):
            normalize_source(split, src)


# ---------------------------------------------------------------------------
# IDF (needed for later features; cheap to compute here per country+source pool)
# ---------------------------------------------------------------------------
def compute_and_save_idf() -> None:
    """Per-country token IDF over S1+S2+S3 core_name; written to artifacts/idf/."""
    C.ensure_dirs()
    # discover countries from S1
    for split in ("train", "test"):
        merged: dict[str, list[list[str]]] = {}
        for src in ("s1", "s2", "s3"):
            for p in C.NORMALIZED_DIR.glob(f"{split}__{src}__*.parquet"):
                country = p.stem.split("__")[-1]
                df = pd.read_parquet(p, columns=["core_name"])
                merged.setdefault(country, []).extend(
                    [str(x).split() for x in df["core_name"].fillna("")]
                )
        for country, docs in merged.items():
            N = len(docs) or 1
            df_counts: Counter = Counter()
            for toks in docs:
                for t in set(toks):
                    df_counts[t] += 1
            idf = {t: float(np.log(1.0 + N / n)) for t, n in df_counts.items()}
            out = C.IDF_DIR / f"{split}__{country}.idf.parquet"
            pd.DataFrame({"token": list(idf.keys()), "idf": list(idf.values())}
                         ).to_parquet(out, index=False)
            print(f"[idf] wrote {out}  ({len(idf):,} tokens)")


def load_idf(split: str, country: str) -> dict[str, float]:
    path = C.IDF_DIR / f"{split}__{country}.idf.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["token"], df["idf"]))


if __name__ == "__main__":
    normalize_all()
    compute_and_save_idf()
