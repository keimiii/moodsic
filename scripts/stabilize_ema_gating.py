#!/usr/bin/env python3

import csv
import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Tuple


class AdaptiveStabilizer:
    def __init__(self, alpha: float = 0.7, uncertainty_threshold: float = 0.4, window_size: int = 60):
        self.alpha = alpha
        self.uncertainty_threshold = uncertainty_threshold
        self.window_size = window_size
        self.ema_valence: Optional[float] = None
        self.ema_arousal: Optional[float] = None
        self.last_stable_valence: float = 0.0
        self.last_stable_arousal: float = 0.0
        self.history: Deque[Tuple[float, float]] = deque(maxlen=window_size)

    def update(self, valence: float, arousal: float, variance: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        if self.ema_valence is None:
            self.ema_valence = valence
            self.ema_arousal = arousal
            self.last_stable_valence = valence
            self.last_stable_arousal = arousal
            self.history.append((valence, arousal))
            return valence, arousal

        self.ema_valence = self.alpha * valence + (1 - self.alpha) * self.ema_valence
        self.ema_arousal = self.alpha * arousal + (1 - self.alpha) * self.ema_arousal

        if variance is not None:
            v_var, a_var = variance
            if v_var > self.uncertainty_threshold:
                out_v = self.last_stable_valence
            else:
                out_v = self.ema_valence
                self.last_stable_valence = out_v

            if a_var > self.uncertainty_threshold:
                out_a = self.last_stable_arousal
            else:
                out_a = self.ema_arousal
                self.last_stable_arousal = out_a
        else:
            out_v = self.ema_valence
            out_a = self.ema_arousal
            self.last_stable_valence = out_v
            self.last_stable_arousal = out_a

        self.history.append((out_v, out_a))
        return out_v, out_a


def read_fusion_csv(path: Path) -> List[Tuple[int, float, float, Optional[float], Optional[float]]]:
    """
    Read a fusion CSV file and return a list of tuples containing the frame, valence, arousal, variance_valence, and variance_arousal.
    Columns:frame,valence,arousal,variance_valence,variance_arousal
    """
    rows: List[Tuple[int, float, float, Optional[float], Optional[float]]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            frame = int(r["frame"]) if "frame" in r and r["frame"] else len(rows) + 1
            valence = float(r["valence"]) if "valence" in r and r["valence"] else 0.0
            arousal = float(r["arousal"]) if "arousal" in r and r["arousal"] else 0.0
            v_var = float(r["variance_valence"]) if "variance_valence" in r and r["variance_valence"] else None
            a_var = float(r["variance_arousal"]) if "variance_arousal" in r and r["variance_arousal"] else None
            rows.append((frame, valence, arousal, v_var, a_var))
    return rows


def stabilize_dataset(
    input_csv: Path,
    alpha: float = 0.7,
    uncertainty_threshold: float = 0.4,
    window_size: int = 60,
    write_debug: bool = True,
    debug_out_csv: Optional[Path] = None,
) -> Tuple[float, float]:
    data = read_fusion_csv(input_csv)
    stabilizer = AdaptiveStabilizer(alpha=alpha, uncertainty_threshold=uncertainty_threshold, window_size=window_size)

    debug_rows: List[Tuple[int, float, float, Optional[float], Optional[float], float, float]] = []

    out_v, out_a = 0.0, 0.0
    for frame, v, a, v_var, a_var in data:
        variance = (v_var, a_var) if v_var is not None and a_var is not None else None
        out_v, out_a = stabilizer.update(v, a, variance=variance)
        if write_debug:
            debug_rows.append((frame, v, a, v_var, a_var, out_v, out_a))

    if write_debug:
        if debug_out_csv is None:
            debug_out_csv = input_csv.with_name(input_csv.stem + "_stabilized.csv")
        with debug_out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "valence", "arousal", "variance_valence", "variance_arousal", "out_valence", "out_arousal"])
            for row in debug_rows:
                writer.writerow(row)

    return out_v, out_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilize fusion outputs with EMA and optional gating.")
    parser.add_argument("input_csv", type=str, help="Path to input CSV with columns: frame,valence,arousal[,variance_valence,variance_arousal]")
    parser.add_argument("--alpha", type=float, default=0.7, help="EMA smoothing factor (default: 0.7)")
    parser.add_argument("--tau", type=float, default=0.4, help="Uncertainty threshold for gating (default: 0.4)")
    parser.add_argument("--window", type=int, default=60, help="History window size (default: 60)")
    parser.add_argument("--no-debug", action="store_true", help="Disable writing per-frame stabilized CSV")
    parser.add_argument("--summary-out", type=str, default=None, help="Optional path to write single-row summary CSV with final valence/arousal")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    out_v, out_a = stabilize_dataset(
        input_csv=input_path,
        alpha=args.alpha,
        uncertainty_threshold=args.tau,
        window_size=args.window,
        write_debug=(not args.no_debug),
    )

    print(f"Final stabilized valence, arousal: {out_v:.4f}, {out_a:.4f}")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["valence", "arousal"])
            writer.writerow([f"{out_v:.6f}", f"{out_a:.6f}"])


if __name__ == "__main__":
    main()
