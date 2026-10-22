# BLiMP subject–verb agreement paradigms (vendored)

Four of the 67 BLiMP paradigms, copied unchanged from
`https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/<paradigm>.jsonl`
on 2026-10-22. E9 uses them as its agreement (control) task through
`polyid.lm.blimp.load_blimp`, which checks these checksums.

| Paradigm | Pairs | SHA-256 |
|---|---|---|
| `distractor_agreement_relational_noun` | 1000 | `37c1553f4c6594b162256edce0f6b40bad1898b311c8aaa69605b526b6494a34` |
| `distractor_agreement_relative_clause` | 1000 | `854728b6db8f23f12583f7332b654835351eaedb50783e6f2a3bf6c101b8c422` |
| `irregular_plural_subject_verb_agreement_1` | 1000 | `0066174143671bffa54232d89254d14221afab666273edfefdfb625fbb75e226` |
| `regular_plural_subject_verb_agreement_1` | 1000 | `1a18d94062c8e792c0a200a3b00dff0e051a92bb084c05e9b5e5c73bee32a620` |

All four support BLiMP's one-prefix method (a shared prefix, then a good or bad next word).
The other two agreement paradigms (`*_subject_verb_agreement_2`) do not, so they are not used.

**Source and licence.** Warstadt, Parrish, Liu, Mohananey, Peng, Wang & Bowman (2020), "BLiMP:
The Benchmark of Linguistic Minimal Pairs for English", *TACL* 8:377–392. Generated from
expert-written grammars; aggregate human agreement with the labels is 96.4 %. Distributed under
[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
