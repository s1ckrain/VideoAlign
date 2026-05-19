"""
Phase D: compute pointwise + pairwise accuracy on the Wan2.1 distribution.

Inputs:
    --reward_scores      outputs/reward_scores.csv     (Phase B)
    --human_pointwise    outputs/human_pointwise.csv   (Phase C, filled by annotators)
    --human_pairwise     outputs/human_pairwise.csv    (Phase C, filled by annotators)

Outputs:
    outputs/accuracy_report.json
    + pretty-printed table to stdout

Metrics:
    Pointwise (per dimension):
        Spearman rho, Pearson r, Kendall tau between (model_reward, human_score)
        Top-1 accuracy per prompt (model's argmax video == human's argmax video)
    Pairwise (per dimension):
        Pairwise accuracy with-ties and without-ties (same algorithm as
        VideoAlign/eval_videogen_rewardbench.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute pointwise + pairwise reward-model accuracy on Wan2.1 distribution.")
    p.add_argument("--reward_scores", required=True)
    p.add_argument("--human_pointwise", default=None,
                   help="Filled-in human_pointwise.csv. If omitted, pointwise metrics are skipped.")
    p.add_argument("--human_pairwise", default=None,
                   help="Filled-in human_pairwise.csv. If omitted, pairwise metrics are skipped.")
    p.add_argument("--videoalign_dir", default="..",
                   help="Path to VideoAlign repo root (uses its calc_accuracy.py for parity). "
                        "Defaults to '..' since this script lives at VideoAlign/wan_eval/.")
    p.add_argument("--output_json", default=None,
                   help="Where to dump the metrics. Defaults to dir of --reward_scores.")
    return p.parse_args()


# ------------------------------ pointwise metrics ----------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import spearmanr  # type: ignore
    mask = (~a.isna()) & (~b.isna())
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import pearsonr  # type: ignore
    mask = (~a.isna()) & (~b.isna())
    if mask.sum() < 3:
        return float("nan")
    r, _ = pearsonr(a[mask], b[mask])
    return float(r)


def _kendall(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import kendalltau  # type: ignore
    mask = (~a.isna()) & (~b.isna())
    if mask.sum() < 3:
        return float("nan")
    tau, _ = kendalltau(a[mask], b[mask])
    return float(tau)


def compute_pointwise(
    reward_df: pd.DataFrame, human_df: pd.DataFrame, dims: list[str],
) -> dict:
    """Spearman/Pearson/Kendall and per-prompt top-1 accuracy."""
    merged = reward_df.merge(human_df, on="video_id", how="inner",
                             suffixes=("_r", "_h"))
    if len(merged) == 0:
        raise ValueError("No overlapping video_id between reward_scores and human_pointwise.")
    for dim in dims:
        merged[f"human_{dim}"] = _to_numeric(merged[f"human_{dim}"])

    results: dict = {"n_videos": int(len(merged))}
    for dim in dims:
        r_col = f"reward_{dim}"
        h_col = f"human_{dim}"

        # Correlations (global, across all videos).
        results[dim] = {
            "spearman": _spearman(merged[r_col], merged[h_col]),
            "pearson":  _pearson(merged[r_col], merged[h_col]),
            "kendall":  _kendall(merged[r_col], merged[h_col]),
        }

        # Per-prompt top-1 accuracy: for each prompt, does model's argmax video
        # coincide with human's argmax video? (ties broken arbitrarily.)
        if "prompt_id_r" in merged.columns:
            prompt_col = "prompt_id_r"
        elif "prompt_id" in merged.columns:
            prompt_col = "prompt_id"
        else:
            prompt_col = None

        if prompt_col is not None:
            hits = total = 0
            for _, group in merged.groupby(prompt_col):
                g = group.dropna(subset=[h_col])
                if len(g) < 2:
                    continue
                model_top = g.loc[g[r_col].idxmax(), "video_id"]
                human_top = g.loc[g[h_col].idxmax(), "video_id"]
                hits += int(model_top == human_top)
                total += 1
            results[dim]["top1_per_prompt"] = float(hits / total) if total else float("nan")
            results[dim]["top1_n_prompts"]  = int(total)
    return results


# ------------------------------ pairwise metrics -----------------------------

def compute_pairwise(
    reward_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    dims: list[str],
    videoalign_dir: Path,
) -> dict:
    """Pairwise accuracy (with/without ties), using VideoAlign's calc_accuracy."""
    sys.path.insert(0, str(videoalign_dir))
    from calc_accuracy import calc_accuracy_with_ties, calc_accuracy_without_ties  # type: ignore

    score_map = {row["video_id"]: row for _, row in reward_df.iterrows()}
    results: dict = {"n_pairs_total": int(len(pair_df))}
    label_map = {"A": 1, "a": 1, "B": -1, "b": -1, "same": 0, "tie": 0, "Same": 0}

    for dim in dims:
        r_col = f"reward_{dim}"
        h_col = f"human_{dim}"
        labels: list[int] = []
        diffs: list[float] = []
        skipped = 0
        for _, row in pair_df.iterrows():
            raw = str(row.get(h_col, "")).strip()
            if raw == "" or raw == "nan":
                skipped += 1
                continue
            if raw not in label_map:
                skipped += 1
                continue
            a_id, b_id = row["video_A_id"], row["video_B_id"]
            if a_id not in score_map or b_id not in score_map:
                skipped += 1
                continue
            labels.append(label_map[raw])
            diffs.append(float(score_map[a_id][r_col]) - float(score_map[b_id][r_col]))

        results[dim] = {
            "n_pairs_used": int(len(labels)),
            "n_pairs_skipped": int(skipped),
            "with_ties":    calc_accuracy_with_ties(labels, diffs) if labels else float("nan"),
            "without_ties": calc_accuracy_without_ties(labels, diffs) if labels else float("nan"),
        }
    return results


# ------------------------------ pretty printing ------------------------------

def _print_table(title: str, rows: list[tuple]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


# ------------------------------ main -----------------------------------------

def main() -> None:
    args = parse_args()

    reward_df = pd.read_csv(args.reward_scores)
    required = {"video_id", "reward_VQ", "reward_MQ", "reward_TA", "reward_Overall"}
    missing = required - set(reward_df.columns)
    if missing:
        raise ValueError(f"reward_scores.csv is missing columns: {missing}")
    dims = ["VQ", "MQ", "TA", "Overall"]

    report: dict = {
        "source": {
            "reward_scores":   str(Path(args.reward_scores).resolve()),
            "human_pointwise": str(Path(args.human_pointwise).resolve()) if args.human_pointwise else None,
            "human_pairwise":  str(Path(args.human_pairwise).resolve())  if args.human_pairwise  else None,
        }
    }

    if args.human_pointwise:
        human_pw = pd.read_csv(args.human_pointwise)
        report["pointwise"] = compute_pointwise(reward_df, human_pw, dims=dims)
        rows = [("dim", "Spearman", "Pearson", "Kendall", "Top-1/prompt")]
        for dim in dims:
            d = report["pointwise"][dim]
            rows.append((
                dim,
                f"{d['spearman']:.4f}" if not np.isnan(d['spearman']) else "nan",
                f"{d['pearson']:.4f}" if not np.isnan(d['pearson']) else "nan",
                f"{d['kendall']:.4f}" if not np.isnan(d['kendall']) else "nan",
                f"{d['top1_per_prompt']:.4f}" if 'top1_per_prompt' in d else "n/a",
            ))
        _print_table(f"Pointwise (N videos = {report['pointwise']['n_videos']})", rows)

    if args.human_pairwise:
        human_pair = pd.read_csv(args.human_pairwise)
        videoalign_dir = Path(args.videoalign_dir).resolve()
        if not (videoalign_dir / "calc_accuracy.py").exists():
            raise FileNotFoundError(f"calc_accuracy.py not found under {videoalign_dir}.")
        report["pairwise"] = compute_pairwise(reward_df, human_pair, dims, videoalign_dir)
        rows = [("dim", "with_ties", "without_ties", "used", "skipped")]
        for dim in dims:
            d = report["pairwise"][dim]
            rows.append((
                dim,
                f"{d['with_ties']:.4f}" if not np.isnan(d['with_ties']) else "nan",
                f"{d['without_ties']:.4f}" if not np.isnan(d['without_ties']) else "nan",
                str(d['n_pairs_used']),
                str(d['n_pairs_skipped']),
            ))
        _print_table(f"Pairwise (total annotated rows = {report['pairwise']['n_pairs_total']})", rows)

    out_path = Path(args.output_json) if args.output_json else Path(args.reward_scores).parent / "accuracy_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
