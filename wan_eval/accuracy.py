"""
Phase D: compute pointwise + pairwise accuracy on the Wan2.1 distribution.

Inputs:
    --reward_scores      outputs/reward_scores.csv     (Phase B)
    --human_pointwise    outputs/human_pointwise.csv   (Phase C, filled by annotators)
    --human_pairwise     outputs/human_pairwise.csv    (Phase C, filled by annotators)

Outputs:
    outputs/accuracy_report.json
    + pretty-printed table to stdout

Metrics evaluated on FIVE dimensions:
    VQ / MQ / TA / Overall  -- individual VideoReward heads
    composite               -- vq_coef*VQ + mq_coef*MQ + ta_coef*TA, EXACTLY the
                               signal DanceGRPO would feed into GRPO advantages.
                               With the default DanceGRPO coefs (1.0, 0.0, 0.0),
                               composite == VQ, so its acc == VQ acc.

    Pointwise (per dimension):
        Spearman rho, Pearson r, Kendall tau between (model_reward, human_score).
        Top-1 accuracy per prompt (model argmax video == human argmax video).
        For 'composite', the human side is composite_human =
            vq_coef*human_VQ + mq_coef*human_MQ + ta_coef*human_TA.
    Pairwise (per dimension):
        Pairwise accuracy with-ties / without-ties (VideoAlign's calc_accuracy).
        For 'composite', the human side is the weighted average of per-dim
        signed labels (A=+1, B=-1, same=0).
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
    # Composite weights: if not provided, auto-loaded from {reward_scores}.coefs.json
    # (written by score.py). Falls back to SAGE-GRPO codebase default (1.0, 1.0, 1.0).
    p.add_argument("--vq_coef", type=float, default=None,
                   help="Composite weight on VQ. Default: load from .coefs.json or 1.0.")
    p.add_argument("--mq_coef", type=float, default=None,
                   help="Composite weight on MQ. Default: load from .coefs.json or 1.0.")
    p.add_argument("--ta_coef", type=float, default=None,
                   help="Composite weight on TA. Default: load from .coefs.json or 1.0.")
    return p.parse_args()


def resolve_coefs(args: argparse.Namespace) -> dict:
    """Resolve composite coefs: CLI > sidecar .coefs.json > SAGE-GRPO codebase defaults (1:1:1)."""
    coefs = {"vq_coef": 1.0, "mq_coef": 1.0, "ta_coef": 1.0}
    sidecar = Path(args.reward_scores).with_suffix(".coefs.json")
    if sidecar.exists():
        try:
            coefs.update(json.loads(sidecar.read_text()))
        except Exception as e:  # noqa: BLE001
            print(f"[acc] WARN: failed to read {sidecar}: {e}")
    for k in ("vq_coef", "mq_coef", "ta_coef"):
        v = getattr(args, k)
        if v is not None:
            coefs[k] = v
    return coefs


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
    reward_df: pd.DataFrame, human_df: pd.DataFrame, dims: list[str], coefs: dict,
) -> dict:
    """Spearman/Pearson/Kendall and per-prompt top-1 accuracy.

    Also computes a 'composite' dim:
        model_composite = coefs.vq*reward_VQ + coefs.mq*reward_MQ + coefs.ta*reward_TA
        human_composite = coefs.vq*human_VQ  + coefs.mq*human_MQ  + coefs.ta*human_TA
    This is the EXACT training signal DanceGRPO would use.
    """
    merged = reward_df.merge(human_df, on="video_id", how="inner",
                             suffixes=("_r", "_h"))
    if len(merged) == 0:
        raise ValueError("No overlapping video_id between reward_scores and human_pointwise.")
    for dim in ("VQ", "MQ", "TA"):
        merged[f"human_{dim}"] = _to_numeric(merged[f"human_{dim}"])
    # Build composite columns for both sides.
    vq, mq, ta = coefs["vq_coef"], coefs["mq_coef"], coefs["ta_coef"]
    if "reward_composite" not in merged.columns:
        merged["reward_composite"] = (
            vq * merged["reward_VQ"] + mq * merged["reward_MQ"] + ta * merged["reward_TA"]
        )
    merged["human_composite"] = (
        vq * merged["human_VQ"] + mq * merged["human_MQ"] + ta * merged["human_TA"]
    )

    results: dict = {"n_videos": int(len(merged)), "composite_coefs": coefs}
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
    coefs: dict,
) -> dict:
    """Pairwise accuracy (with/without ties), using VideoAlign's calc_accuracy.

    For dim == 'composite':
      - model side: sign of weighted reward diff
            comp_diff = vq*(VQ_A - VQ_B) + mq*(MQ_A - MQ_B) + ta*(TA_A - TA_B)
      - human side: weighted average of per-dim signed labels
            comp_label = vq*lab_VQ + mq*lab_MQ + ta*lab_TA   (each label in {-1,0,+1})
            then re-discretized to {-1,0,+1} via sign() (ties when |sum|<1e-9).
    Pair is SKIPPED if any contributing per-dim cell is missing.
    """
    sys.path.insert(0, str(videoalign_dir))
    from calc_accuracy import calc_accuracy_with_ties, calc_accuracy_without_ties  # type: ignore

    score_map = {row["video_id"]: row for _, row in reward_df.iterrows()}
    results: dict = {"n_pairs_total": int(len(pair_df)), "composite_coefs": coefs}
    label_map = {"A": 1, "a": 1, "B": -1, "b": -1, "same": 0, "tie": 0, "Same": 0}
    vq, mq, ta = coefs["vq_coef"], coefs["mq_coef"], coefs["ta_coef"]

    def _per_dim(r_col: str, h_col: str) -> tuple[list[int], list[float], int]:
        labels: list[int] = []
        diffs: list[float] = []
        skipped = 0
        for _, row in pair_df.iterrows():
            raw = str(row.get(h_col, "")).strip()
            if raw in ("", "nan") or raw not in label_map:
                skipped += 1
                continue
            a_id, b_id = row["video_A_id"], row["video_B_id"]
            if a_id not in score_map or b_id not in score_map:
                skipped += 1
                continue
            labels.append(label_map[raw])
            diffs.append(float(score_map[a_id][r_col]) - float(score_map[b_id][r_col]))
        return labels, diffs, skipped

    def _composite() -> tuple[list[int], list[float], int]:
        labels: list[int] = []
        diffs: list[float] = []
        skipped = 0
        for _, row in pair_df.iterrows():
            try:
                a_id, b_id = row["video_A_id"], row["video_B_id"]
                if a_id not in score_map or b_id not in score_map:
                    skipped += 1
                    continue
                # Per-dim labels: skip the pair if any required dim is empty
                # AND that dim has a non-zero coef.
                lab_vq = lab_mq = lab_ta = 0
                if vq != 0:
                    raw = str(row.get("human_VQ", "")).strip()
                    if raw in ("", "nan") or raw not in label_map:
                        skipped += 1
                        continue
                    lab_vq = label_map[raw]
                if mq != 0:
                    raw = str(row.get("human_MQ", "")).strip()
                    if raw in ("", "nan") or raw not in label_map:
                        skipped += 1
                        continue
                    lab_mq = label_map[raw]
                if ta != 0:
                    raw = str(row.get("human_TA", "")).strip()
                    if raw in ("", "nan") or raw not in label_map:
                        skipped += 1
                        continue
                    lab_ta = label_map[raw]
                # Weighted human label, then sign() to get {-1, 0, +1}.
                comp = vq * lab_vq + mq * lab_mq + ta * lab_ta
                if abs(comp) < 1e-9:
                    human_label = 0
                elif comp > 0:
                    human_label = 1
                else:
                    human_label = -1
                # Weighted model diff.
                a, b = score_map[a_id], score_map[b_id]
                comp_diff = (
                    vq * (float(a["reward_VQ"]) - float(b["reward_VQ"]))
                    + mq * (float(a["reward_MQ"]) - float(b["reward_MQ"]))
                    + ta * (float(a["reward_TA"]) - float(b["reward_TA"]))
                )
                labels.append(human_label)
                diffs.append(comp_diff)
            except Exception:  # noqa: BLE001
                skipped += 1
        return labels, diffs, skipped

    for dim in dims:
        if dim == "composite":
            labels, diffs, skipped = _composite()
        else:
            labels, diffs, skipped = _per_dim(f"reward_{dim}", f"human_{dim}")

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
    # composite is reported last so it stands out as "the DanceGRPO training signal".
    dims = ["VQ", "MQ", "TA", "Overall", "composite"]

    coefs = resolve_coefs(args)
    _presets = {
        (1.0, 0.0, 0.0): "DanceGRPO official (VQ only)",
        (1.0, 1.0, 1.0): "SAGE-GRPO codebase default (1:1:1 averaging)",
        (0.5, 0.5, 1.0): "SAGE-GRPO paper Setting B (HunyuanVideo-tuned)",
    }
    _key = (coefs["vq_coef"], coefs["mq_coef"], coefs["ta_coef"])
    print(f"[acc] composite coefs: {coefs}  -- {_presets.get(_key, 'custom')}")

    report: dict = {
        "source": {
            "reward_scores":   str(Path(args.reward_scores).resolve()),
            "human_pointwise": str(Path(args.human_pointwise).resolve()) if args.human_pointwise else None,
            "human_pairwise":  str(Path(args.human_pairwise).resolve())  if args.human_pairwise  else None,
        },
        "composite_coefs": coefs,
    }

    if args.human_pointwise:
        human_pw = pd.read_csv(args.human_pointwise)
        report["pointwise"] = compute_pointwise(reward_df, human_pw, dims=dims, coefs=coefs)
        rows = [("dim", "Spearman", "Pearson", "Kendall", "Top-1/prompt")]
        for dim in dims:
            d = report["pointwise"][dim]
            marker = "  *DanceGRPO signal*" if dim == "composite" else ""
            rows.append((
                dim + marker,
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
        report["pairwise"] = compute_pairwise(reward_df, human_pair, dims, videoalign_dir, coefs=coefs)
        rows = [("dim", "with_ties", "without_ties", "used", "skipped")]
        for dim in dims:
            d = report["pairwise"][dim]
            marker = "  *DanceGRPO signal*" if dim == "composite" else ""
            rows.append((
                dim + marker,
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
