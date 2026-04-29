"""Phase 4: surface linguistic features on comm_round rows."""

from __future__ import annotations

import re
import pandas as pd

# Minimal recipe-adjacent lexicon (extend per order in downstream work)
ENTITY_WORDS = {
    "egg",
    "pot",
    "counter",
    "onion",
    "soup",
    "dish",
    "plate",
    "oven",
    "dispenser",
    "ingredient",
    "cook",
    "pick",
    "place",
}


def _imperative_hits(text: str) -> int:
    t = (text or "").lower()
    return len(re.findall(r"\b(put|place|pick|pickup|deliver|go|move|take|hold|wait)\b", t))


def _question_hits(text: str) -> int:
    return (text or "").count("?")


def _please_hits(text: str) -> int:
    return len(re.findall(r"\bplease\b", (text or "").lower()))


def _hedge_hits(text: str) -> int:
    t = (text or "").lower()
    return len(re.findall(r"\b(i think|maybe|perhaps|not sure|might)\b", t))


def _entity_hits(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for w in ENTITY_WORDS if w in t)


def add_linguistic_columns(comm_df: pd.DataFrame) -> pd.DataFrame:
    if comm_df.empty:
        return comm_df
    out = comm_df.copy()
    for col, fn in [
        ("say_char_len", lambda s: s.astype(str).str.len()),
        ("say_word_count", lambda s: s.astype(str).str.split().str.len()),
        ("say_sentence_count", lambda s: s.astype(str).str.count(r"\.") + s.astype(str).str.count(r"\?") + 1),
        ("say_imperative_hits", lambda s: s.astype(str).map(_imperative_hits)),
        ("say_question_marks", lambda s: s.astype(str).map(_question_hits)),
        ("say_please_hits", lambda s: s.astype(str).map(_please_hits)),
        ("say_hedge_hits", lambda s: s.astype(str).map(_hedge_hits)),
        ("say_entity_hits", lambda s: s.astype(str).map(_entity_hits)),
    ]:
        out[col] = fn(out["say_raw"])

    out["analysis_char_len"] = out["analysis_raw"].astype(str).str.len()
    out["plan_complexity"] = out["plan_atom_count"].fillna(0).astype(int)
    return out
