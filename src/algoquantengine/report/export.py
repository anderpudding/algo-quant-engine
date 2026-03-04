from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def export_weights_csv(tickers: list[str], w: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["ticker,weight"]
    for t, wi in zip(tickers, w.tolist()):
        lines.append(f"{t},{wi}")
    Path(path).write_text("\n".join(lines) + "\n")


def export_frontier_csv(frontier: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["target_return,return,vol,sharpe"]
    for p in frontier:
        lines.append(f'{p["target_return"]},{p["return"]},{p["vol"]},{p["sharpe"]}')
    Path(path).write_text("\n".join(lines) + "\n")


def export_report_json(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n")

def export_group_caps_json(caps: list[tuple[list[int], float]], path: str) -> None:
    from pathlib import Path
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = [{"indices": idxs, "cap": cap} for idxs, cap in caps]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")