"""Small I/O helpers: tidy result tables and JSON with numpy support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=NumpyJSONEncoder))
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_table(path: str | Path, rows: list[dict] | pd.DataFrame) -> Path:
    """Write a tidy table as CSV (always) — portable and diff-friendly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path
