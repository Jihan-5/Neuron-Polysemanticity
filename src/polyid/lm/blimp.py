r"""BLiMP minimal pairs as forced-choice items (E9's agreement task).

BLiMP (Warstadt et al., TACL 2020) has 67 paradigms of 1000 minimal pairs each, generated from
expert-written grammars; aggregate human agreement with its labels is 96.4 %. It is distributed
under CC-BY 4.0. E9 uses the subject–verb agreement paradigms that support BLiMP's "one prefix"
method: a shared prefix followed by a good or bad next word, which is exactly a
:class:`ChoiceItem`. The two ``distractor_agreement_*`` paradigms put an attractor noun between
subject and verb, the design E9's agreement task was meant to have. No item is written or
labelled here (rule L7).

The files are vendored in ``data/blimp/`` with their source and SHA-256 checksums
(``data/blimp/README.md``); :func:`load_blimp` refuses a file whose checksum differs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from polyid.lm.templates import ChoiceItem

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "blimp"

AGREEMENT_PARADIGMS: dict[str, str] = {
    "distractor_agreement_relational_noun": "37c1553f4c6594b162256edce0f6b40bad1898b311c8aaa69605b526b6494a34",
    "distractor_agreement_relative_clause": "854728b6db8f23f12583f7332b654835351eaedb50783e6f2a3bf6c101b8c422",
    "irregular_plural_subject_verb_agreement_1": "0066174143671bffa54232d89254d14221afab666273edfefdfb625fbb75e226",
    "regular_plural_subject_verb_agreement_1": "1a18d94062c8e792c0a200a3b00dff0e051a92bb084c05e9b5e5c73bee32a620",
}


def load_blimp(paradigms: list[str] | None = None, data_dir: str | Path | None = None,
               task: str = "agreement") -> list[ChoiceItem]:
    """One :class:`ChoiceItem` per minimal pair: ``prompt`` = BLiMP's one-prefix prefix, choices =
    its good and bad next word (with a leading space). ``family`` = ``"<paradigm>#<pairID>"``."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DIR
    items = []
    for uid in paradigms or list(AGREEMENT_PARADIGMS):
        path = data_dir / f"{uid}.jsonl"
        raw = path.read_bytes()
        want = AGREEMENT_PARADIGMS.get(uid)
        if want and hashlib.sha256(raw).hexdigest() != want:
            raise ValueError(f"{path}: SHA-256 differs from the vendored BLiMP file")
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("one_prefix_method"):
                raise ValueError(f"{uid} does not support the one-prefix method")
            items.append(ChoiceItem(prompt=r["one_prefix_prefix"], correct=" " + r["one_prefix_word_good"],
                                    incorrect=" " + r["one_prefix_word_bad"], task=task,
                                    family=f"{r['UID']}#{r['pairID']}"))
    return items
