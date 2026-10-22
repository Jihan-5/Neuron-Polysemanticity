# SemCor swap pairs (E9 semantic task)

`semcor_noun_swap_pairs.jsonl` holds 4,011 pairs over 558 common nouns, derived deterministically
from SemCor 3.0 by `scripts/build_semcor_items.py`. No item was written by hand or by a model
(rule L7). Every sentence is real Brown Corpus text; only the cut point and mechanical
detokenization come from code.

**Pair.** Two sentences use the same noun (single word, not a proper name) in different WordNet
supersenses, i.e. different lexicographer files such as `noun.act` vs `noun.artifact`. Each
sentence is cut right after the noun into a prefix (at least 5 tokens) and a continuation
(4–10 tokens). E9 scores a pair as matched-minus-swapped log-probability (`lm/semcor.py`), so
every prefix and continuation enters once on each side.

**Selection.** A noun is used if at least two of its supersenses have at least 3 usable
occurrences in `brown1` and `brown2` (the files where all content words are tagged). Up to 12
pairs per supersense pair are taken from opposite ends of the occurrence list, round-robin
across supersense pairs, with at most 12 pairs per noun.

**Splits.** By noun hash (`lm/semcor.py`): 727 dev pairs (103 nouns) for the pre-check that the
model beats chance; 3,284 test pairs (455 nouns) for E9. No noun is in both.

**Source.** NLTK data package `semcor` ("SemCor 3.0", Rada Mihalcea), sha256
`126fa2e829ab63edd5b3fd9de45ef1d60d6880e01e25abc55b5ac7918a824655`, downloaded 2026-10-22 from
`https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/semcor.zip`.
SemCor 1.6 was sense-annotated by hand at Princeton against WordNet 1.6. SemCor 3.0 maps those
annotations to WordNet 3.0 automatically, and a few senses were dropped in the mapping (those
words are skipped here). Output sha256:
`b1052c8e8c4f018e31824e6544a3dc4e4e3008ec2d15921c79056a7c78b70aee` (checked by the loader).

**Licence.** The Princeton WordNet / Semantic Concordance licence allows use, copying,
modification and distribution for any purpose, provided the copyright notice and statements,
including the disclaimer, appear on all copies. They are in `LICENSE` (verbatim) and in
`NOTICE_SemCor3.0_README_excerpt.txt` (the SemCor 3.0 mapping notice, verbatim).
