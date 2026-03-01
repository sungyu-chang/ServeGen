#!/usr/bin/env python3
"""
Generate a 10-minute workload trace at 100 req/s with m-large model,
save it as a CSV, and visualize the arrival pattern with matplotlib.
"""

import os
import sys

# Ensure the package is importable when run from the trace/ subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt

from servegen import Category, ClientPool
from servegen.construct import generate_workload
from servegen.utils import get_constant_rate_fn, save_requests_to_csv

DURATION   = 600      # 10 minutes in seconds
TARGET_RATE = 100.0   # requests per second
SEED       = 42
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "workload.csv")
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "arrival_pattern.png")


def main():
    # ── 1. Load client pool ──────────────────────────────────────────────────
    print("Loading ClientPool (LANGUAGE / m-large)...")
    pool = ClientPool(Category.LANGUAGE, "m-large")
    print(f"  {len(pool.clients)} clients loaded.")

    # ── 2. Build constant-rate function & generate workload ──────────────────
    rate_fn = get_constant_rate_fn(pool, target_rate=TARGET_RATE)
    print(f"Generating workload  ({TARGET_RATE} req/s × {DURATION}s)...")
    requests = generate_workload(pool, rate_fn, duration=DURATION, seed=SEED)
    print(f"  {len(requests)} requests generated.")

    # ── 3. Save CSV ──────────────────────────────────────────────────────────
    save_requests_to_csv(requests, OUTPUT_CSV)
    print(f"  Saved trace → {OUTPUT_CSV}")

    # ── 4. Visualize ─────────────────────────────────────────────────────────
    timestamps = np.array([r.timestamp for r in requests])

    # Bin into 10-second windows for a smooth rate plot
    bin_width = 10  # seconds
    bins = np.arange(0, DURATION + bin_width, bin_width)
    counts, edges = np.histogram(timestamps, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    rate_per_bin = counts / bin_width        # convert count → req/s

    # Inter-arrival times
    iats = np.diff(np.sort(timestamps)) * 1000  # ms

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"ServeGen Arrival Pattern — m-large, {TARGET_RATE} req/s, {DURATION}s",
        fontsize=14, fontweight="bold"
    )

    # ── Panel 1: Request rate over time ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(bin_centers, rate_per_bin, color="steelblue", linewidth=1.2)
    ax.axhline(TARGET_RATE, color="crimson", linestyle="--", linewidth=1,
               label=f"Target {TARGET_RATE} req/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Request rate (req/s)")
    ax.set_title("Instantaneous Request Rate (10 s bins)")
    ax.legend()
    ax.set_xlim(0, DURATION)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Cumulative requests ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(np.sort(timestamps), np.arange(1, len(timestamps) + 1),
            color="darkorange", linewidth=1.2, label="Actual")
    ideal_t = np.array([0, DURATION])
    ax.plot(ideal_t, ideal_t * TARGET_RATE, "r--", linewidth=1,
            label=f"Ideal ({TARGET_RATE} req/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative requests")
    ax.set_title("Cumulative Arrivals")
    ax.legend()
    ax.set_xlim(0, DURATION)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: IAT histogram ────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(iats, bins=80, color="mediumseagreen", edgecolor="none", density=True)
    ax.set_xlabel("Inter-arrival time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inter-Arrival Time Distribution")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Input & output token lengths ─────────────────────────────────
    ax = axes[1, 1]
    input_tokens  = np.array([r.data["input_tokens"]  for r in requests])
    output_tokens = np.array([r.data["output_tokens"] for r in requests])
    ax.hist(input_tokens,  bins=60, alpha=0.6, color="royalblue",   label="Input tokens",  density=True)
    ax.hist(output_tokens, bins=60, alpha=0.6, color="darkorange",  label="Output tokens", density=True)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Density")
    ax.set_title("Input / Output Token Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"  Saved figure → {OUTPUT_PNG}")

    # ── 5. Summary stats ──────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"  Total requests    : {len(requests)}")
    print(f"  Duration          : {DURATION}s")
    print(f"  Observed avg rate : {len(requests)/DURATION:.2f} req/s")
    print(f"  Input tokens      — mean: {input_tokens.mean():.1f}  median: {np.median(input_tokens):.1f}  p99: {np.percentile(input_tokens, 99):.1f}")
    print(f"  Output tokens     — mean: {output_tokens.mean():.1f}  median: {np.median(output_tokens):.1f}  p99: {np.percentile(output_tokens, 99):.1f}")
    print(f"  IAT (ms)          — mean: {iats.mean():.2f}  p50: {np.median(iats):.2f}  p99: {np.percentile(iats, 99):.2f}")


if __name__ == "__main__":
    main()
