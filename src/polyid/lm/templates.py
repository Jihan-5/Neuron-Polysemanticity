r"""Controlled text: lexical/semantic contrasts and forced-choice evaluation tasks.

Plan p.9 "Four semantic controls": same-word/same-sense, same-word/different-
sense, different-word/same-sense, different-word/different-sense (syllabus S27:
lexical identity confounds semantic-overlap claims). Plan p.9 "Causal
validation": held-out semantic decision contexts and unrelated-task contexts,
scoring a predefined continuation choice.

Everything here is generated deterministically from small banks. **Provenance
correction (2026-10-22):** these banks were drafted by the AI coding assistant
during development, not written by a person, so under rule L7 (no LLM-written
items) they are development placeholders only. E9 refuses them outside smoke
runs; its reported agreement task comes from BLiMP (``lm/blimp.py``), and the
semantic task needs a human-validated source (README open items).
Each item carries a ``family`` id (the frame that generated it); splits are
made by family, so test items come from held-out frames, not paraphrases of
training frames.

The banks are deliberately small and auditable; extending them (more words,
human-checked frames) is listed as an open item in docs/EXPERIMENT_PLAN.md.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

# homograph -> sense -> (frames containing {w}, sense-associated single-token continuations)
HOMOGRAPHS: dict[str, dict[str, dict[str, list[str]]]] = {
    "bank": {
        "finance": {"frames": ["She deposited her paycheck at the {w} before lunch.",
                               "The {w} approved the loan after a long review.",
                               "He asked the {w} teller about the interest rate.",
                               "Robbers broke into the {w} vault overnight."],
                    "assoc": [" money", " cash", " account", " loan"]},
        "river": {"frames": ["They had a picnic on the grassy {w} of the stream.",
                             "The canoe drifted toward the muddy {w} of the river.",
                             "Ducks rested along the {w} near the water.",
                             "Erosion slowly wore away the river {w}."],
                  "assoc": [" water", " river", " mud", " fish"]},
    },
    "bat": {
        "animal": {"frames": ["A {w} flew out of the cave at dusk.",
                              "The {w} hung upside down from the branch.",
                              "Scientists tracked the {w} with a tiny radio tag.",
                              "The fruit {w} sleeps during the day."],
                   "assoc": [" wings", " cave", " night", " flying"]},
        "sports": {"frames": ["He swung the {w} and hit the ball over the fence.",
                              "The wooden {w} cracked during the game.",
                              "She gripped the {w} tightly at the plate.",
                              "The coach handed him a lighter {w}."],
                   "assoc": [" ball", " game", " swing", " hit"]},
    },
    "spring": {
        "season": {"frames": ["Flowers bloom early in the {w} this year.",
                              "The {w} rains filled the reservoir.",
                              "We plant tomatoes every {w}.",
                              "Birds return north in the {w}."],
                   "assoc": [" flowers", " season", " weather", " rain"]},
        "coil": {"frames": ["The mattress {w} squeaked loudly.",
                            "A broken {w} jammed the old clock.",
                            "He compressed the metal {w} with pliers.",
                            "The pen's tiny {w} popped out."],
                 "assoc": [" metal", " coil", " tension", " steel"]},
    },
    "crane": {
        "bird": {"frames": ["A {w} waded through the shallow marsh.",
                            "The whooping {w} is an endangered bird.",
                            "We watched a {w} spread its long wings.",
                            "The {w} caught a frog in its beak."],
                 "assoc": [" bird", " wings", " feathers", " marsh"]},
        "machine": {"frames": ["The {w} lifted the steel beam onto the roof.",
                               "Workers waited while the {w} moved the container.",
                               "A tower {w} loomed over the construction site.",
                               "The {w} operator checked the cables."],
                    "assoc": [" steel", " construction", " lifting", " site"]},
    },
    "seal": {
        "animal": {"frames": ["A {w} basked on the rocks near the harbor.",
                              "The baby {w} swam close to its mother.",
                              "Fishermen spotted a {w} chasing fish.",
                              "The {w} barked at the passing boat."],
                   "assoc": [" ocean", " fish", " swimming", " beach"]},
        "closure": {"frames": ["Check the {w} on the jar before storing it.",
                               "The rubber {w} around the door had cracked.",
                               "She broke the wax {w} on the letter.",
                               "A tight {w} keeps air out of the container."],
                    "assoc": [" tight", " airtight", " lid", " leak"]},
    },
    "pitch": {
        "music": {"frames": ["The singer adjusted the {w} of her voice.",
                             "Perfect {w} lets him name any note.",
                             "The violin's {w} drifted during the concert.",
                             "She hummed at a higher {w}."],
                  "assoc": [" note", " tone", " voice", " music"]},
        "field": {"frames": ["The players walked onto the muddy {w}.",
                             "Groundskeepers rolled the {w} before the match.",
                             "Rain flooded the football {w}.",
                             "Fans ran onto the {w} after the goal."],
                  "assoc": [" grass", " match", " players", " stadium"]},
    },
    "mole": {
        "animal": {"frames": ["A {w} dug tunnels under the garden.",
                              "The {w} pushed up a mound of soil.",
                              "Gardeners tried to trap the {w}.",
                              "The blind {w} sensed vibrations underground."],
                   "assoc": [" soil", " tunnel", " garden", " digging"]},
        "spy": {"frames": ["The agency suspected a {w} inside the department.",
                           "The {w} leaked documents to a rival government.",
                           "Investigators finally exposed the {w}.",
                           "He worked for years as a {w} in the embassy."],
                "assoc": [" spy", " secret", " agent", " intelligence"]},
    },
    "palm": {
        "tree": {"frames": ["A tall {w} swayed in the island breeze.",
                            "Coconuts fell from the {w} onto the sand.",
                            "They rested in the shade of a {w}.",
                            "The beach was lined with {w} trees."],
                 "assoc": [" tree", " beach", " coconut", " tropical"]},
        "hand": {"frames": ["She read the lines on his {w}.",
                            "He held the coin flat on his {w}.",
                            "Sweat formed on her {w} before the speech.",
                            "The splinter was stuck in his {w}."],
                 "assoc": [" hand", " fingers", " skin", " wrist"]},
    },
}

# different-word / same-sense partners (synonyms in the given sense)
SYNONYMS: dict[tuple[str, str], str] = {
    ("bank", "river"): "shore",
    ("bat", "sports"): "club",
    ("spring", "season"): "springtime",
    ("crane", "machine"): "hoist",
    ("seal", "closure"): "gasket",
    ("pitch", "field"): "field",
    ("mole", "spy"): "informant",
    ("palm", "tree"): "coconut tree",
}

# unrelated task: subject–verb number agreement with an attractor noun
AGREEMENT_SUBJECTS = [("key", "keys"), ("author", "authors"), ("dog", "dogs"), ("teacher", "teachers"),
                      ("window", "windows"), ("report", "reports"), ("bottle", "bottles"), ("player", "players")]
AGREEMENT_ATTRACTORS = [("cabinet", "cabinets"), ("garden", "gardens"), ("museum", "museums"), ("project", "projects"),
                        ("village", "villages"), ("kitchen", "kitchens")]
AGREEMENT_FRAMES = ["The {s} near the {a}", "The {s} behind the {a}", "The {s} from the {a}", "The {s} beside the {a}"]


@dataclass(frozen=True)
class ContrastItem:
    text: str
    word: str  # surface word whose final subword is the target position
    sense: str
    lemma: str  # homograph lemma the sense belongs to
    family: str  # generating frame id (split unit)


@dataclass(frozen=True)
class ChoiceItem:
    prompt: str
    correct: str
    incorrect: str
    task: str
    family: str


def family_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def contrast_items() -> list[ContrastItem]:
    out = []
    for lemma, senses in HOMOGRAPHS.items():
        for sense, spec in senses.items():
            for fi, frame in enumerate(spec["frames"]):
                fam = family_id(lemma, sense, str(fi))
                out.append(ContrastItem(frame.format(w=lemma), lemma, sense, lemma, fam))
                syn = SYNONYMS.get((lemma, sense))
                if syn:
                    out.append(ContrastItem(frame.format(w=syn), syn, sense, lemma, fam))
    return out


def four_way_pairs(items: list[ContrastItem]) -> dict[str, list[tuple[ContrastItem, ContrastItem]]]:
    """The four control types of plan p.9, as pairs of items from *different* families."""
    pairs: dict[str, list] = {"same_word_same_sense": [], "same_word_diff_sense": [],
                              "diff_word_same_sense": [], "diff_word_diff_sense": []}
    for a, b in itertools.combinations(items, 2):
        if a.family == b.family or a.lemma != b.lemma:
            continue
        same_word = a.word == b.word
        same_sense = a.sense == b.sense
        key = f"{'same' if same_word else 'diff'}_word_{'same' if same_sense else 'diff'}_sense"
        pairs[key].append((a, b))
    return pairs


def semantic_choice_items() -> list[ChoiceItem]:
    """Forced choice between a sense-consistent and a sense-inconsistent associate."""
    out = []
    for lemma, senses in HOMOGRAPHS.items():
        names = list(senses)
        for sense in names:
            other = [s for s in names if s != sense][0]
            for fi, frame in enumerate(senses[sense]["frames"]):
                fam = family_id(lemma, sense, str(fi))
                for good, bad in zip(senses[sense]["assoc"], senses[other]["assoc"]):
                    for lead in (" It reminded everyone of", " The story was mostly about", " Nearby there was"):
                        out.append(ChoiceItem(frame.format(w=lemma) + lead, good, bad, "semantic", fam))
    return out


def agreement_items() -> list[ChoiceItem]:
    out = []
    for fi, frame in enumerate(AGREEMENT_FRAMES):
        for (s_sg, s_pl), (a_sg, a_pl) in itertools.product(AGREEMENT_SUBJECTS, AGREEMENT_ATTRACTORS):
            fam = family_id("agree", str(fi), s_sg)
            for subj, attr, good, bad in ((s_sg, a_pl, " is", " are"), (s_pl, a_sg, " are", " is")):
                out.append(ChoiceItem(frame.format(s=subj, a=attr), good, bad, "agreement", fam))
    return out


def split_by_family(items, fractions=(0.6, 0.2, 0.2), salt: str = "polyid") -> dict[str, list]:
    """Deterministic family-level split into fit / dev / test (no family crosses splits)."""
    cut1, cut2 = fractions[0], fractions[0] + fractions[1]
    out: dict[str, list] = {"fit": [], "dev": [], "test": []}
    for it in items:
        h = int(hashlib.sha256(f"{salt}|{it.family}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out["fit" if h < cut1 else "dev" if h < cut2 else "test"].append(it)
    return out
