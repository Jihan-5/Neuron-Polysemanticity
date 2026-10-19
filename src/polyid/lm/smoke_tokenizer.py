"""A tiny byte-level tokenizer with the subset of the HF interface the pipeline uses.

Only for offline plumbing tests with :func:`polyid.lm.hooks.tiny_random_gptneox`;
never used for reported results.
"""

from __future__ import annotations

import torch


class ByteTokenizer:
    pad_token = "\x00"
    eos_token = "\x00"
    padding_side = "right"

    def _encode(self, text: str):
        ids, offs = [], []
        for i, ch in enumerate(text):
            ids.append(min(ord(ch), 50000) + 1)
            offs.append((i, i + 1))
        return ids, offs

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None,
                 add_special_tokens=True, return_offsets_mapping=False):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        enc = [self._encode(t) for t in batch]
        if truncation and max_length:
            enc = [(i[:max_length], o[:max_length]) for i, o in enc]
        if return_tensors == "pt":
            L = max(len(i) for i, _ in enc)
            ids = torch.zeros(len(enc), L, dtype=torch.long)
            mask = torch.zeros(len(enc), L, dtype=torch.long)
            for b, (i, _) in enumerate(enc):
                ids[b, :len(i)] = torch.tensor(i)
                mask[b, :len(i)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        if single:
            out = {"input_ids": enc[0][0]}
            if return_offsets_mapping:
                out["offset_mapping"] = enc[0][1]
            return out
        return {"input_ids": [i for i, _ in enc]}
