r"""The identification map (plan p.10: "the key plot ... where the answer is recoverable,
uncertain, or not identified").

Each (dataset, neuron) result of an interval method is classified as

* **recovered**           — singleton interval equal to the truth;
* **uncertain (sampling)** — wider than the population identified set, covering;
* **not identified**      — the population identified set itself is wider than a point
  (structural ambiguity) and the interval covers the truth;
* **confidently wrong**   — singleton (or any interval) that excludes the truth.

Cells are 100 %-stacked bars per condition, faceted by sample size; status colours are the
reserved status palette and every segment class is also named in the legend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATUS_ORDER = ["recovered", "uncertain (sampling)", "not identified", "wrong"]


def classify(rows: pd.DataFrame) -> pd.Series:
    covered = rows["covered"].astype(float) > 0.5
    single = rows["width"] == 0
    id_w = (rows["idset_upper"] - rows["idset_lower"]) if "idset_upper" in rows else pd.Series(0, index=rows.index)
    out = np.where(~covered, "wrong",
                   np.where(single, "recovered", np.where(id_w > 0, "not identified", "uncertain (sampling)")))
    return pd.Series(out, index=rows.index)


def identification_map(rows: pd.DataFrame, method: str, y: str, facet: str | None, out_path: str | Path,
                       y_order: list | None = None, title: str | None = None) -> Path:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    colors = {"recovered": style.STATUS["identified"], "uncertain (sampling)": style.STATUS["partial"],
              "not identified": style.STATUS["misspecified"], "wrong": style.STATUS["confidently wrong"]}
    df = rows[rows["method"] == method].copy()
    df["status"] = classify(df)
    facets = [None] if facet is None else sorted(df[facet].unique())
    ys = y_order or sorted(df[y].unique())
    fig, axes = plt.subplots(1, len(facets), figsize=(3.6 * len(facets) + 1.6, 0.36 * len(ys) + 1.5),
                             sharey=True, squeeze=False)
    for ax, fv in zip(axes[0], facets):
        sub = df if fv is None else df[df[facet] == fv]
        frac = (sub.groupby(y)["status"].value_counts(normalize=True).unstack(fill_value=0.0)
                .reindex(index=ys, columns=STATUS_ORDER, fill_value=0.0))
        left = np.zeros(len(ys))
        for st in STATUS_ORDER:
            w = frac[st].to_numpy()
            ax.barh(np.arange(len(ys)), w, left=left, color=colors[st], height=0.62, label=st,
                    edgecolor=style.SURFACE, linewidth=1.0)
            left += w
        ax.set_xlim(0, 1)
        ax.set_yticks(np.arange(len(ys)), [str(v).replace("_", " ") for v in ys])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of neuron-level results")
        if fv is not None:
            ax.set_title(f"{facet} = {fv}", fontsize=10)
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", x=0.02, ha="left")
    out_path = Path(out_path)
    style.finish(fig, out_path)
    return out_path
