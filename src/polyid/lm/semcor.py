r"""SemCor swap pairs as E9's semantic task.

Items come from SemCor 3.0 (hand sense-annotated Brown Corpus text; Princeton WordNet licence,
notice in ``data/semcor``), built deterministically by ``scripts/build_semcor_items.py``. Each
pair holds two real sentences that use the same common noun in different WordNet supersenses,
each cut right after the noun into a prefix and a continuation. A pair becomes two
:class:`ChoiceItem`\ s with the same ``family``:

* prompt ``prefix_a``: correct ``cont_a``, incorrect ``cont_b``;
* prompt ``prefix_b``: correct ``cont_b``, incorrect ``cont_a``.

Summed within the family (``run_ablation(paired_tasks={"semantic"})``) their scores give
matched-minus-swapped log-probability, in which every prefix and every continuation enters once
on each side, so length and frequency differences cancel and only the sense match counts.

Splits are by lemma (``dev_fraction`` of lemmas by hash): ``dev`` is for the pre-check that
the model is above chance before E9 counts; ``test`` is what E9 reports. No lemma is in both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from polyid.lm.templates import ChoiceItem

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "semcor" / "semcor_noun_swap_pairs.jsonl"
SHA256 = "b1052c8e8c4f018e31824e6544a3dc4e4e3008ec2d15921c79056a7c78b70aee"


def _split_of(lemma: str, dev_fraction: float) -> str:
    h = int(hashlib.sha256(f"semcor|{lemma}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_fraction else "test"


def load_semcor_pairs(split: str = "test", path: str | Path | None = None, dev_fraction: float = 0.2) -> list[dict]:
    raw = Path(path or DEFAULT_PATH).read_bytes()
    if path is None and hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("data/semcor pairs differ from the committed build (re-run scripts/build_semcor_items.py)")
    pairs = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if split == "all":
        return pairs
    return [p for p in pairs if _split_of(p["lemma"], dev_fraction) == split]


def semcor_choice_items(split: str = "test", path: str | Path | None = None, dev_fraction: float = 0.2,
                        task: str = "semantic") -> list[ChoiceItem]:
    items = []
    for p in load_semcor_pairs(split, path, dev_fraction):
        fam = p["pair_id"]  # "<lemma>#..." so paired intervals cluster on the lemma
        items.append(ChoiceItem(p["prefix_a"], p["cont_a"], p["cont_b"], task, fam))
        items.append(ChoiceItem(p["prefix_b"], p["cont_b"], p["cont_a"], task, fam))
    return items
