r"""Six declared environments, document-level splits and the token-selection rule.

Plan p.9 "Corpus construction": six environments from independently sourced
or generated text — balanced topics, shifted topic frequencies, lexical
variants and controlled semantic contrasts; "freeze the model and vary only
the sampling distribution"; 50 000 selected token contexts per environment,
split **by source document** 60 / 20 / 20 into fit / development / locked
test; contexts from one document are clustered.

Environments (declared before any activation is computed)
---------------------------------------------------------
``topics_balanced``     AG News, the four topics equally weighted
``topics_world``        AG News, 70 % World / 10 % each other topic
``topics_scitech``      AG News, 70 % Sci/Tech / 10 % each other topic
``encyclopedic``        WikiText-103 paragraphs (independent source)
``lexical_variants``    generated frames with homographs *and* their synonyms
``semantic_contrasts``  generated frames, homographs in both senses

AG News and WikiText require Hugging Face ``datasets`` and network access;
``offline=True`` substitutes deterministic template text for them so the
pipeline can be exercised end-to-end without downloads — such runs are
labelled ``offline`` in their manifest and must not be reported as results.

Token-selection rule (plan p.9 "Token selection")
-------------------------------------------------
* generic text: positions ``>= min_position`` (default 16) in the tokenised
  document, at most ``max_per_doc`` positions drawn uniformly with a seed
  derived from the document id — never chosen by activation value;
* controlled contrasts: the **final subword** of the target word span.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from polyid.lm.templates import HOMOGRAPHS, SYNONYMS, contrast_items

AG_NEWS = {"path": "ag_news", "revision": "main", "labels": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}}
WIKITEXT = {"path": "wikitext", "name": "wikitext-103-raw-v1", "revision": "main"}

ENVIRONMENTS = ["topics_balanced", "topics_world", "topics_scitech", "encyclopedic", "lexical_variants",
                "semantic_contrasts"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    env: str
    text: str
    family: str  # split unit: source document, or generating frame for templates
    target_word: str | None = None
    sense: str | None = None


def doc_split(family: str, salt: str = "polyid-corpus", fractions=(0.6, 0.2, 0.2)) -> str:
    h = int(hashlib.sha256(f"{salt}|{family}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "fit" if h < fractions[0] else "dev" if h < fractions[0] + fractions[1] else "test"


def _offline_topic_text(topic: str, i: int, rng: np.random.Generator) -> str:
    vocab = {
        "World": ["minister", "election", "border", "treaty", "capital", "protest", "summit", "embassy"],
        "Sports": ["match", "season", "coach", "goal", "league", "tournament", "striker", "record"],
        "Business": ["shares", "profit", "merger", "market", "investors", "quarter", "revenue", "bank"],
        "Sci/Tech": ["software", "researchers", "chip", "satellite", "network", "genome", "laptop", "data"],
    }[topic]
    words = rng.choice(vocab, size=60)
    return " ".join(f"The {w} was reported again." if k % 6 == 0 else w for k, w in enumerate(words))


def topic_documents(env: str, n_docs: int, seed: int, offline: bool) -> list[Document]:
    weights = {"topics_balanced": [0.25, 0.25, 0.25, 0.25],
               "topics_world": [0.7, 0.1, 0.1, 0.1],
               "topics_scitech": [0.1, 0.1, 0.1, 0.7]}[env]
    rng = np.random.default_rng(int(hashlib.sha256(f"{env}{seed}".encode()).hexdigest()[:8], 16))
    labels = rng.choice(4, size=n_docs, p=weights)
    docs = []
    if offline:
        for i, lab in enumerate(labels):
            topic = AG_NEWS["labels"][int(lab)]
            did = f"offline-ag-{lab}-{i}"
            docs.append(Document(did, env, _offline_topic_text(topic, i, rng), did))
        return docs
    from datasets import load_dataset

    ds = load_dataset(AG_NEWS["path"], split="train", revision=AG_NEWS["revision"])
    by_label = {k: np.flatnonzero(np.asarray(ds["label"]) == k) for k in range(4)}
    for i, lab in enumerate(labels):
        j = int(rng.choice(by_label[int(lab)]))
        did = f"agnews-train-{j}"
        docs.append(Document(did, env, ds[j]["text"], did))  # the same article may appear in several envs
    return docs


def encyclopedic_documents(n_docs: int, seed: int, offline: bool) -> list[Document]:
    rng = np.random.default_rng(seed + 7)
    if offline:
        return [Document(f"offline-wiki-{i}", "encyclopedic", _offline_topic_text("World", i, rng), f"offline-wiki-{i}")
                for i in range(n_docs)]
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT["path"], WIKITEXT["name"], split="train", revision=WIKITEXT["revision"])
    paras = [i for i, t in enumerate(ds["text"]) if len(t) > 400 and not t.strip().startswith("=")]
    pick = rng.choice(paras, size=min(n_docs, len(paras)), replace=False)
    return [Document(f"wikitext-{int(i)}", "encyclopedic", ds[int(i)]["text"], f"wikitext-{int(i)}") for i in pick]


def template_documents(env: str) -> list[Document]:
    items = contrast_items()
    if env == "semantic_contrasts":
        items = [it for it in items if it.word == it.lemma]
    docs = []
    for k, it in enumerate(items):
        docs.append(Document(f"{env}-{it.family}-{k}", env, it.text, it.family, target_word=it.word, sense=it.sense))
    return docs


def build_environments(n_docs: int = 2000, seed: int = 0, offline: bool = False) -> dict[str, list[Document]]:
    envs = {e: topic_documents(e, n_docs, seed, offline) for e in ("topics_balanced", "topics_world", "topics_scitech")}
    envs["encyclopedic"] = encyclopedic_documents(n_docs, seed, offline)
    envs["lexical_variants"] = template_documents("lexical_variants")
    envs["semantic_contrasts"] = template_documents("semantic_contrasts")
    return envs


def select_positions(input_ids: list[int], doc_id: str, min_position: int = 16, max_per_doc: int = 8) -> list[int]:
    """Predeclared generic rule: uniform positions >= min_position, seeded by the document id."""
    valid = np.arange(min_position, len(input_ids))
    if valid.size == 0:
        return []
    rng = np.random.default_rng(int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16))
    k = min(max_per_doc, valid.size)
    return sorted(rng.choice(valid, size=k, replace=False).tolist())


def target_position(tokenizer, text: str, word: str) -> int | None:
    """Index of the final subword of the first occurrence of ``word`` (offset-aligned)."""
    start = text.find(word)
    if start < 0:
        return None
    end = start + len(word)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    last = None
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s < end and e > start:
            last = i
    return last


def corpus_manifest(envs: dict[str, list[Document]], offline: bool) -> dict:
    h = hashlib.sha256()
    for e in sorted(envs):
        for d in envs[e]:
            h.update(f"{e}|{d.doc_id}|{d.text}".encode())
    return {"environments": {e: len(v) for e, v in envs.items()}, "corpus_sha256": h.hexdigest(), "offline": offline,
            "sources": {"ag_news": AG_NEWS, "wikitext": WIKITEXT}, "homographs": sorted(HOMOGRAPHS),
            "synonyms": {f"{k[0]}:{k[1]}": v for k, v in SYNONYMS.items()}}


def doc_to_dict(d: Document) -> dict:
    return asdict(d)
