"""Abbreviation map — seed + mined from training pairs + unsupervised (plan §3.4).

Layers:
  1. Seed list (hand-curated common EN / IN / FR abbreviations).
  2. Mined from R ∪ G positive pairs: token alignments where a short token
     (2–5 chars) is a prefix or a subsequence of a longer token in the paired
     name/address. Kept if count ≥ 5 and the short→long mapping is dominant
     (≥70% of occurrences).
  3. Unsupervised, per country, over S1+S2+S3 vocab: short prefixes/subsequences
     of longer tokens appearing in the same *slot* (first/last) with a shared
     rare token in surrounding names. This layer needs a light manual review
     for French tokens (top-100).

Conflict rule: mined entries override seed entries; a short form mapped to
multiple long forms is dropped (ambiguous).

Output: artifacts/abbreviations.json  {"name": {...}, "address": {...}}
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import config as C
from . import splits as splits_mod
from .io_utils import explode_ground_truth, read_ground_truth, read_source_tsv


# --- Seed ------------------------------------------------------------------
SEED_NAME = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "intl": "international",
    "mfg": "manufacturing", "assn": "association", "govt": "government",
    "dept": "department", "natl": "national", "cie": "compagnie",
    "ste": "societe", "ets": "etablissements", "svc": "service",
    "svcs": "services", "grp": "group", "hosp": "hospital",
    "univ": "university", "sch": "school",
}
SEED_ADDR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "opp": "opposite",
    "nr": "near", "bd": "boulevard", "av": "avenue", "ch": "chemin",
    "pl": "place", "hn": "house number", "hno": "house number",
    "h no": "house number", "s no": "survey number", "kh no": "khasra number",
    "plot no": "plot number", "shp no": "shop number", "shop no": "shop number",
    "sec": "sector", "flr": "floor", "bldg": "building", "apt": "apartment",
    "flat": "flat", "door": "door",
}


# --- Helpers ---------------------------------------------------------------
def _tokens(s) -> list[str]:
    if not isinstance(s, str) or not s:
        return []
    return [t for t in s.lower().split() if t]


def _is_prefix(short: str, long: str) -> bool:
    return len(short) < len(long) and long.startswith(short)


def _is_subsequence(short: str, long: str) -> bool:
    if len(short) >= len(long):
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def _accept_mining(candidates: Counter, min_count: int = 5,
                   dominance: float = 0.70) -> dict[str, str]:
    """Reduce a (short, long) counter to a clean {short: long} map.

    dominance: fraction of that short's total mappings that the top long form
    must represent.
    """
    by_short: dict[str, Counter] = defaultdict(Counter)
    for (s, l), n in candidates.items():
        by_short[s][l] += n
    out: dict[str, str] = {}
    for short, longs in by_short.items():
        total = sum(longs.values())
        if total < min_count:
            continue
        top_long, top_n = longs.most_common(1)[0]
        if top_n / total >= dominance:
            out[short] = top_long
    return out


# --- Layer 2: mine from true pairs ----------------------------------------
def _mine_from_pairs(text_map_a: dict[str, str], text_map_b: dict[str, str],
                     pair_ids: pd.DataFrame) -> Counter:
    """For every (a_id, b_id) true pair, align tokens by matching a short
    token to a longer token that it prefixes or is a subsequence of, in the
    other side of the pair. Return a Counter over (short, long) pairs.
    """
    cnt: Counter = Counter()
    for a, b in zip(pair_ids["a"], pair_ids["b"]):
        toks_a = _tokens(text_map_a.get(a, ""))
        toks_b = _tokens(text_map_b.get(b, ""))
        for ta in toks_a:
            for tb in toks_b:
                if ta == tb:
                    continue
                if 2 <= len(ta) <= 5 and len(tb) > len(ta):
                    if _is_prefix(ta, tb) or _is_subsequence(ta, tb):
                        cnt[(ta, tb)] += 1
                if 2 <= len(tb) <= 5 and len(ta) > len(tb):
                    if _is_prefix(tb, ta) or _is_subsequence(tb, ta):
                        cnt[(tb, ta)] += 1
    return cnt


# --- Layer 3: unsupervised per country ------------------------------------
def _mine_unsupervised(names_by_country: dict[str, list[str]],
                       min_freq: int = 10) -> dict[str, dict[str, str]]:
    """Very light unsupervised pass: within a country, tokens of length 2–5
    that are a prefix of a much more common token in the same first-or-last
    slot and share vocabulary neighbours.
    """
    out: dict[str, dict[str, str]] = {}
    for country, names in names_by_country.items():
        first: Counter = Counter()
        last: Counter = Counter()
        for n in names:
            t = _tokens(n)
            if not t:
                continue
            first[t[0]] += 1
            last[t[-1]] += 1
        candidates: Counter = Counter()
        for slot in (first, last):
            common = {tok for tok, n in slot.items()
                      if n >= min_freq and len(tok) > 5}
            shorts = {tok for tok, n in slot.items()
                      if n >= min_freq and 2 <= len(tok) <= 5}
            for s in shorts:
                for l in common:
                    if _is_prefix(s, l) or _is_subsequence(s, l):
                        candidates[(s, l)] += 1
        # Accept only unique mappings — ambiguous ones will be dropped by
        # _accept_mining anyway (dominance rule).
        out[country] = _accept_mining(candidates, min_count=1, dominance=0.70)
    return out


# --- Orchestration ---------------------------------------------------------
def build() -> dict:
    C.ensure_dirs()
    cap = C.NORMALIZE_ROW_CAP or 0
    def _load(path):
        df = read_source_tsv(path)
        return df.iloc[:cap].copy() if cap else df
    if cap:
        print(f"[abbrev] loading sources (capped to {cap:,} rows/source) + splits + GT…")
    else:
        print("[abbrev] loading sources + splits + GT…")
    s1 = _load(C.TRAIN_SOURCE["s1"])
    s2 = _load(C.TRAIN_SOURCE["s2"])
    s3 = _load(C.TRAIN_SOURCE["s3"])
    ts1 = _load(C.TEST_SOURCE["s1"])
    ts2 = _load(C.TEST_SOURCE["s2"])
    ts3 = _load(C.TEST_SOURCE["s3"])
    gt = read_ground_truth(C.TRAIN_GT)
    gt_long = explode_ground_truth(gt)
    split = splits_mod.load()
    rg_entities = set(split.loc[split["group"].isin(["R", "G"]), "entity_id"])
    pairs = gt_long[gt_long["source1_entity_id"].isin(rg_entities)].copy()
    pairs.columns = ["a", "b"]

    name_map = pd.concat([s1, s2, s3]).set_index("entity_id")["business_name"].fillna("").to_dict()
    addr_map = pd.concat([s1, s2, s3]).set_index("entity_id")["business_address"].fillna("").to_dict()

    # Only keep pairs where BOTH ids are in our (possibly capped) name_map;
    # otherwise mining iterates over empty tokens millions of times.
    if cap:
        _keys = set(name_map.keys())
        before = len(pairs)
        pairs = pairs[pairs["a"].isin(_keys) & pairs["b"].isin(_keys)].copy()
        print(f"[abbrev] filtered pairs to those in capped subset: {before:,} -> {len(pairs):,}")

    print(f"[abbrev] mining name pairs from {len(pairs):,} R∪G positives…")
    name_pairs = _mine_from_pairs(name_map, name_map, pairs)
    print(f"[abbrev] mining address pairs…")
    addr_pairs = _mine_from_pairs(addr_map, addr_map, pairs)

    name_mined = _accept_mining(name_pairs)
    addr_mined = _accept_mining(addr_pairs)

    print("[abbrev] unsupervised pass per country…")
    names_by_country_all: dict[str, list[str]] = defaultdict(list)
    for df in (s1, s2, s3, ts1, ts2, ts3):
        for c, n in zip(df["country"], df["business_name"].fillna("")):
            names_by_country_all[str(c)].append(n)
    unsup = _mine_unsupervised(names_by_country_all)

    # Merge: seed → unsupervised → mined  (later layers override earlier)
    def _merge(*layers: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for l in layers:
            for k, v in l.items():
                if k in out and out[k] != v:
                    # ambiguous: drop
                    out.pop(k, None)
                else:
                    out[k] = v
        return out

    unsup_flat: dict[str, str] = {}
    for c, m in unsup.items():
        for k, v in m.items():
            if k in unsup_flat and unsup_flat[k] != v:
                unsup_flat.pop(k, None)
            else:
                unsup_flat[k] = v

    name_final = _merge(SEED_NAME, unsup_flat, name_mined)
    addr_final = _merge(SEED_ADDR, addr_mined)

    result = {
        "name": name_final,
        "address": addr_final,
        "meta": {
            "n_seed_name": len(SEED_NAME),
            "n_mined_name": len(name_mined),
            "n_unsup_name": len(unsup_flat),
            "n_seed_addr": len(SEED_ADDR),
            "n_mined_addr": len(addr_mined),
            "per_country_unsup_name": {c: len(m) for c, m in unsup.items()},
        },
    }
    Path(C.ABBREV_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(C.ABBREV_PATH).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[abbrev] wrote {C.ABBREV_PATH}  meta={result['meta']}")
    return result


def load() -> dict:
    if not Path(C.ABBREV_PATH).exists():
        return {"name": dict(SEED_NAME), "address": dict(SEED_ADDR)}
    return json.loads(Path(C.ABBREV_PATH).read_text(encoding="utf-8"))


if __name__ == "__main__":
    build()
