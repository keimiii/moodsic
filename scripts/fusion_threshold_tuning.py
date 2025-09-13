from __future__ import annotations

"""
Fusion threshold sweep for stability under occlusion/low light.

This tool evaluates inverse-variance fusion with optional gating on a small
validation slice by sweeping thresholds and logging stability metrics.

Inputs (CSV expected):
- frame_id: any sortable identifier
- scene_v, scene_a: scene predictions in [-1, 1]
- scene_var_v, scene_var_a: optional scene variances (>=0) — can be empty
- face_v, face_a: face predictions in [-1, 1]
- face_var_v, face_var_a: optional face variances (>=0) — can be empty
- face_score: optional detection/quality score in [0, 1]
- brightness: optional frame luma [0, 255]

Outputs:
- metrics.csv: jitter and gating stats per threshold combination
- report.html: (optional) simple chart of a selected configuration if Altair is installed

Usage example:
  uv run python scripts/fusion_threshold_tuning.py \
      --csv logs/fe_val_predictions.csv \
      --face-score-thresholds 0,0.3,0.5,0.7 \
      --face-max-sigma  ,0.6,0.8,1.0 \
      --brightness-thresholds  ,30,60 \
      --output-dir experiments/fusion_tuning

Notes:
- Threshold lists accept empty entries to denote "disabled".
- If CSV lacks some columns, the script will treat them as missing and fall back
  to fixed-weight fusion where appropriate.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def parse_list_floats(arg: str) -> Iterable[Optional[float]]:
    vals = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok == "" or tok.lower() in {"none", "null"}:
            vals.append(None)
        else:
            vals.append(float(tok))
    return vals


def iv_or_fixed(
    scene_mean: float,
    scene_var: Optional[float],
    face_mean: float,
    face_var: Optional[float],
    *,
    use_iv: bool,
    scene_w: float,
    face_w: float,
) -> Tuple[float, Optional[float], str]:
    """Fuse a single dimension with inverse-variance if possible, else fixed.

    Returns (fused_mean, fused_var_or_None, mode).
    """
    if use_iv and _finite_pos(scene_var) and _finite_pos(face_var):
        w1 = 1.0 / (float(scene_var) + 1e-6)
        w2 = 1.0 / (float(face_var) + 1e-6)
        tot = w1 + w2
        if tot > 0 and math.isfinite(tot):
            mean = (w1 * scene_mean + w2 * face_mean) / tot
            var = 1.0 / tot
            return float(np.clip(mean, -1.0, 1.0)), var, "iv"
    # fixed fallback
    tot = scene_w + face_w
    if tot <= 0:
        scene_w = face_w = 0.5
        tot = 1.0
    mean = (scene_w * scene_mean + face_w * face_mean) / tot
    return float(np.clip(mean, -1.0, 1.0)), None, "fixed"


def _finite_pos(x: Optional[float]) -> bool:
    try:
        return x is not None and math.isfinite(float(x)) and float(x) > 0
    except Exception:
        return False


def compute_fused(
    row: pd.Series,
    *,
    face_score_threshold: Optional[float],
    face_max_sigma: Optional[float],
    brightness_threshold: Optional[float],
    scene_weight: float,
    face_weight: float,
    use_variance_weighting: bool,
) -> Tuple[float, float, Optional[float], Optional[float], bool]:
    """Compute fused V, A, and whether face was gated off for this row."""
    # Base validity for face: require available predictions
    face_available = not (pd.isna(row.get("face_v")) or pd.isna(row.get("face_a")))
    face_valid = bool(face_available)

    # Gating by score
    if face_valid and face_score_threshold is not None:
        score = float(row.get("face_score", 0.0) or 0.0)
        if score < face_score_threshold:
            face_valid = False

    # Gating by sigma
    if face_valid and face_max_sigma is not None:
        fv_var = row.get("face_var_v")
        fa_var = row.get("face_var_a")
        sig_v = math.sqrt(float(fv_var)) if _finite_pos(fv_var) else float("inf")
        sig_a = math.sqrt(float(fa_var)) if _finite_pos(fa_var) else float("inf")
        if sig_v > face_max_sigma or sig_a > face_max_sigma:
            face_valid = False

    # Gating by brightness
    if face_valid and brightness_threshold is not None:
        bri = float(row.get("brightness", 255.0) or 255.0)
        if bri < brightness_threshold:
            face_valid = False

    # Prepare inputs
    sv = float(row.get("scene_v", 0.0) or 0.0)
    sa = float(row.get("scene_a", 0.0) or 0.0)
    s_vv = row.get("scene_var_v")
    s_va = row.get("scene_var_a")

    if face_valid:
        fv = float(row.get("face_v", 0.0) or 0.0)
        fa = float(row.get("face_a", 0.0) or 0.0)
        f_vv = row.get("face_var_v")
        f_va = row.get("face_var_a")
        v, vv, _ = iv_or_fixed(sv, s_vv, fv, f_vv, use_iv=use_variance_weighting, scene_w=scene_weight, face_w=face_weight)
        a, av, _ = iv_or_fixed(sa, s_va, fa, f_va, use_iv=use_variance_weighting, scene_w=scene_weight, face_w=face_weight)
    else:
        # passthrough scene
        v, a = sv, sa
        vv = s_vv if _finite_pos(s_vv) else None
        av = s_va if _finite_pos(s_va) else None

    return v, a, vv, av, not face_valid


def sweep_thresholds(df: pd.DataFrame, *,
                     face_score_thresholds: Iterable[Optional[float]],
                     face_max_sigmas: Iterable[Optional[float]],
                     brightness_thresholds: Iterable[Optional[float]],
                     scene_weight: float,
                     face_weight: float,
                     use_variance_weighting: bool) -> pd.DataFrame:
    rows = []
    # sort by frame if available to make jitter meaningful
    order_key = "frame_id" if "frame_id" in df.columns else None
    if order_key is not None:
        df = df.sort_values(order_key)

    for s_thr in face_score_thresholds:
        for sig_thr in face_max_sigmas:
            for b_thr in brightness_thresholds:
                v_vals, a_vals = [], []
                gated_count = 0
                for _, r in df.iterrows():
                    v, a, _, _, gated = compute_fused(
                        r,
                        face_score_threshold=s_thr,
                        face_max_sigma=sig_thr,
                        brightness_threshold=b_thr,
                        scene_weight=scene_weight,
                        face_weight=face_weight,
                        use_variance_weighting=use_variance_weighting,
                    )
                    v_vals.append(v)
                    a_vals.append(a)
                    gated_count += int(gated)

                # jitter as mean absolute first-difference
                v_arr = np.asarray(v_vals, dtype=np.float32)
                a_arr = np.asarray(a_vals, dtype=np.float32)
                v_jitter = float(np.mean(np.abs(np.diff(v_arr)))) if len(v_arr) > 1 else 0.0
                a_jitter = float(np.mean(np.abs(np.diff(a_arr)))) if len(a_arr) > 1 else 0.0
                rows.append({
                    "face_score_thr": s_thr,
                    "face_sigma_thr": sig_thr,
                    "brightness_thr": b_thr,
                    "jitter_v": v_jitter,
                    "jitter_a": a_jitter,
                    "jitter_mean": (v_jitter + a_jitter) / 2.0,
                    "frames": len(df),
                    "gated_frames": int(gated_count),
                    "gated_ratio": float(gated_count) / max(1, len(df)),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Sweep fusion gating thresholds for stability")
    ap.add_argument("--csv", required=True, help="CSV with per-frame predictions")
    ap.add_argument("--face-score-thresholds", default="0.0,0.3,0.5,0.7", help="Comma-separated list; empty for disabled")
    ap.add_argument("--face-max-sigma", default=",0.6,0.8,1.0", help="Comma-separated sigmas; empty for disabled")
    ap.add_argument("--brightness-thresholds", default=",40,60", help="Comma-separated luma thresholds; empty for disabled")
    ap.add_argument("--scene-weight", type=float, default=0.6)
    ap.add_argument("--face-weight", type=float, default=0.4)
    ap.add_argument("--no-variance-weighting", action="store_true", help="Disable inverse-variance weighting (use fixed weights)")
    ap.add_argument("--output-dir", default="experiments/fusion_tuning", help="Directory to write metrics and report")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_list = parse_list_floats(args.face_score_thresholds)
    sig_list = parse_list_floats(args.face_max_sigma)
    b_list = parse_list_floats(args.brightness_thresholds)

    metrics = sweep_thresholds(
        df,
        face_score_thresholds=s_list,
        face_max_sigmas=sig_list,
        brightness_thresholds=b_list,
        scene_weight=args.scene_weight,
        face_weight=args.face_weight,
        use_variance_weighting=not args.no_variance_weighting,
    )

    metrics_path = out_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[fusion_tuning] Wrote metrics: {metrics_path}")

    # Try to create a simple report with Altair if available
    try:
        import altair as alt  # type: ignore

        best = metrics.sort_values("jitter_mean").head(1)
        if not best.empty:
            s_thr = best.iloc[0]["face_score_thr"]
            sig_thr = best.iloc[0]["face_sigma_thr"]
            b_thr = best.iloc[0]["brightness_thr"]

            # Recompute the best time series for plotting
            rows = []
            df_plot = df.copy()
            if "frame_id" not in df_plot.columns:
                df_plot = df_plot.reset_index().rename(columns={"index": "frame_id"})
            for _, r in df_plot.iterrows():
                v, a, _, _, gated = compute_fused(
                    r,
                    face_score_threshold=s_thr,
                    face_max_sigma=sig_thr,
                    brightness_threshold=b_thr,
                    scene_weight=args.scene_weight,
                    face_weight=args.face_weight,
                    use_variance_weighting=not args.no_variance_weighting,
                )
                rows.append({
                    "frame_id": r["frame_id"],
                    "valence": v,
                    "arousal": a,
                    "gated": int(gated),
                })
            df_series = pd.DataFrame(rows)

            chart_v = alt.Chart(df_series).mark_line().encode(
                x="frame_id:N", y="valence:Q", color=alt.value("steelblue")
            )
            chart_a = alt.Chart(df_series).mark_line().encode(
                x="frame_id:N", y="arousal:Q", color=alt.value("orange")
            )
            chart_g = alt.Chart(df_series).mark_area(opacity=0.15).encode(
                x="frame_id:N", y="gated:Q", color=alt.value("red")
            )
            report = (chart_v + chart_a + chart_g).properties(
                title=f"Best thresholds: score>={s_thr}, sigma<={sig_thr}, brightness>={b_thr}"
            )
            report_path = out_dir / "report.html"
            report.save(str(report_path))
            print(f"[fusion_tuning] Wrote report: {report_path}")
    except Exception as e:
        print(f"[fusion_tuning] Skipping report generation: {e}")


if __name__ == "__main__":
    main()

