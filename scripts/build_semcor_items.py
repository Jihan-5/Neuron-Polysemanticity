"""Build E9's semantic items from SemCor 3.0 (deterministic; no item is written by hand or by a model).

Usage:  python -I scripts/build_semcor_items.py <path/to/semcor.zip> <out.jsonl>

Input: the NLTK ``semcor.zip`` package (SemCor 3.0, sha256 below). SemCor's sense labels were
annotated by hand on Brown Corpus text against WordNet 1.6 and mapped automatically to WordNet
3.0 (semcor/README).

Each output line is one *swap pair*: two sentences that use the same single-word common noun in
different WordNet supersenses (lexicographer files, e.g. noun.group vs noun.object). Each
sentence is cut right after the noun into a prefix and a continuation of up to ``MAX_CONT``
tokens. E9 scores a pair as

    [log P(cont_a | prefix_a) + log P(cont_b | prefix_b)] - [log P(cont_b | prefix_a) + log P(cont_a | prefix_b)],

so every continuation and every prefix enters once on each side: length and frequency
differences cancel, and only the sense match counts. Text is detokenized mechanically
(underscores to spaces, Brown quote tokens dropped, standard spacing before punctuation).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

SEMCOR_ZIP_SHA256 = "126fa2e829ab63edd5b3fd9de45ef1d60d6880e01e25abc55b5ac7918a824655"
MIN_PREFIX = 5  # tokens of context before and including the noun
MIN_CONT, MAX_CONT = 4, 10
MIN_PER_SENSE = 3  # occurrences per (lemma, supersense) for the lemma to be used
MAX_PAIRS_PER_SENSE_PAIR = 12
MAX_PAIRS_PER_LEMMA = 12  # keeps generic nouns ('thing', 'part') from dominating
# WordNet lexicographer files for nouns (supersenses)
LEXNAMES = {"03": "Tops", "04": "act", "05": "animal", "06": "artifact", "07": "attribute", "08": "body",
            "09": "cognition", "10": "communication", "11": "event", "12": "feeling", "13": "food", "14": "group",
            "15": "location", "16": "motive", "17": "object", "18": "person", "19": "phenomenon", "20": "plant",
            "21": "possession", "22": "process", "23": "quantity", "24": "relation", "25": "shape", "26": "state",
            "27": "substance", "28": "time"}
DROP = {"``", "''", "`", "'"}


def detok(tokens: list[str]) -> str:
    s = " ".join(t for t in tokens if t not in DROP)
    s = re.sub(r" ([.,;:!?)%])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    return s


def sentences(zf: zipfile.ZipFile):
    for name in sorted(n for n in zf.namelist() if re.search(r"semcor/brown[12]/tagfiles/.+\.xml$", n)):
        root = ET.fromstring(zf.read(name))
        for s in root.iter("s"):
            toks = []
            for el in s:
                text = (el.text or "").replace("_", " ")
                info = None
                if el.tag == "wf" and el.get("cmd") == "done" and el.get("lexsn") and el.get("wnsn", "0") != "0":
                    info = {"lemma": el.get("lemma", ""), "pos": el.get("pos", ""), "lexsn": el.get("lexsn"),
                            "proper": el.get("pn") is not None or el.get("rdf") is not None}
                toks.append((text, info))
            yield f"{name.split('/')[-1][:-4]}#{s.get('snum')}", toks


def occurrences(zf):
    occ = defaultdict(list)  # (lemma, supersense) -> list of (sid, prefix, cont)
    for sid, toks in sentences(zf):
        words = [t for t, _ in toks]
        for i, (text, info) in enumerate(toks):
            if not info or info["proper"] or info["pos"] not in ("NN", "NNS") or "_" in info["lemma"]:
                continue
            if " " in text or not text.isalpha():
                continue
            ss_type, lexfile = info["lexsn"].split(":")[:2]
            if ss_type != "1":
                continue
            pre, cont = words[:i + 1], words[i + 1:i + 1 + MAX_CONT]
            if len([w for w in pre if w not in DROP]) < MIN_PREFIX or len([w for w in cont if w not in DROP]) < MIN_CONT:
                continue
            prefix_text = detok(pre)
            full = detok(pre + cont)
            if not full.startswith(prefix_text):
                continue
            occ[(info["lemma"], f"noun.{LEXNAMES.get(lexfile, lexfile)}")].append(
                (f"{sid}#{i}", prefix_text, full[len(prefix_text):]))
    return occ


def build(zip_path: str) -> list[dict]:
    raw = open(zip_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != SEMCOR_ZIP_SHA256:
        raise SystemExit("semcor.zip checksum differs from the NLTK package this script was built for")
    import io

    occ = occurrences(zipfile.ZipFile(io.BytesIO(raw)))
    by_lemma = defaultdict(dict)
    for (lemma, ss), xs in occ.items():
        if len(xs) >= MIN_PER_SENSE:
            by_lemma[lemma][ss] = sorted(xs)
    pairs = []
    for lemma in sorted(by_lemma):
        senses = by_lemma[lemma]
        if len(senses) < 2:
            continue
        per_pair = []
        for sa, sb in itertools.combinations(sorted(senses), 2):
            xa, xb = senses[sa], senses[sb]
            per_pair.append([(sa, sb, k, xa[k], xb[-1 - k])  # opposite ends: spread over documents
                             for k in range(min(len(xa), len(xb), MAX_PAIRS_PER_SENSE_PAIR))])
        # round-robin over sense pairs until the per-lemma cap
        chosen = [x for x in itertools.chain.from_iterable(itertools.zip_longest(*per_pair)) if x][:MAX_PAIRS_PER_LEMMA]
        for sa, sb, k, (ida, pa, ca), (idb, pb, cb) in chosen:
            pairs.append({"pair_id": f"{lemma}#{sa}#{sb}#{k}", "lemma": lemma, "sense_a": sa, "sense_b": sb,
                          "source_a": ida, "source_b": idb, "prefix_a": pa, "cont_a": ca,
                          "prefix_b": pb, "cont_b": cb})
    return pairs


if __name__ == "__main__":
    out = build(sys.argv[1])
    with open(sys.argv[2], "w") as fh:
        for p in out:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(out)} pairs over {len({p['lemma'] for p in out})} lemmas -> {sys.argv[2]}")
