#!/usr/bin/env python3
"""
Generate a 10-minute bursty workload trace with m-large model at 100 req/s average.

Burstiness strategy
-------------------
ServeGen models burstiness through each client's *coefficient of variation* (CV)
of inter-arrival times.  CV = 1 is a Poisson process; CV > 1 means heavier tails
(more frequent back-to-back bursts separated by long silences).

This script selects only high-CV clients (CV >= 2) so that, even though the
*average* aggregate rate is identical to the uniform baseline, the instantaneous
rate swings much more dramatically — producing clear burst spikes and quiet valleys.

Two traces are generated so the figure can compare them side-by-side:
  - Uniform  : all clients (any CV), constant 100 req/s target
  - Bursty   : high-CV clients only (CV >= 2), same 100 req/s target

Outputs (not committed to git):
  bursty_workload.csv
  bursty_vs_uniform.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from servegen import Category, ClientPool
from servegen.construct import generate_workload
from servegen.utils import get_constant_rate_fn, save_requests_to_csv

TARGET_RATE = 100.0   # req/s — same for both traces
MIN_CV      = 2.0     # CV threshold for "bursty" clients
MAX_CV      = 1000.0
DURATION    = 600     # 10 minutes
SEED        = 42
OUT_CSV     = os.path.join(os.path.dirname(__file__), "bursty_workload.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "bursty_vs_uniform.png")


def compute_rate_series(requests, duration, bin_width=5):
    """Bin requests into time windows and return (centers, req/s) arrays."""
    ts    = np.array([r.timestamp for r in requests])
    bins  = np.arange(0, duration + bin_width, bin_width)
    counts, edges = np.histogram(ts, bins=bins)
    return (edges[:-1] + edges[1:]) / 2, counts / bin_width


def main():
    # ── Load pools ───────────────────────────────────────────────────────────
    print("Loading ClientPool (LANGUAGE / m-large)...")
    pool = ClientPool(Category.LANGUAGE, "m-large")

    bursty_view = pool.filter_by_cv(MIN_CV, MAX_CV)

    n_all    = len(pool.span(0, DURATION).get())
    n_bursty = len(bursty_view.span(0, DURATION).get())
    print(f"  {n_all} windows in 10-min span  →  "
          f"{n_bursty} high-CV windows (CV ≥ {MIN_CV})")

    # ── Generate traces ───────────────────────────────────────────────────────
    print(f"\nGenerating uniform trace  ({TARGET_RATE} req/s × {DURATION} s)...")
    uniform_rate_fn = get_constant_rate_fn(pool, TARGET_RATE)
    uniform_reqs    = generate_workload(pool, uniform_rate_fn, duration=DURATION, seed=SEED)
    print(f"  {len(uniform_reqs)} requests  |  "
          f"avg rate = {len(uniform_reqs)/DURATION:.2f} req/s")

    print(f"\nGenerating bursty trace   (CV ≥ {MIN_CV}, {TARGET_RATE} req/s × {DURATION} s)...")
    bursty_rate_fn = get_constant_rate_fn(bursty_view, TARGET_RATE)
    bursty_reqs    = generate_workload(bursty_view, bursty_rate_fn, duration=DURATION, seed=SEED)
    print(f"  {len(bursty_reqs)} requests  |  "
          f"avg rate = {len(bursty_reqs)/DURATION:.2f} req/s")

    save_requests_to_csv(bursty_reqs, OUT_CSV)
    print(f"  Saved → {OUT_CSV}")

    # ── Derived metrics ───────────────────────────────────────────────────────
    u_ts   = np.array([r.timestamp for r in uniform_reqs])
    b_ts   = np.array([r.timestamp for r in bursty_reqs])
    u_iats = np.diff(np.sort(u_ts)) * 1000   # ms
    b_iats = np.diff(np.sort(b_ts)) * 1000

    BIN = 5  # seconds per bin
    u_centers, u_rate = compute_rate_series(uniform_reqs, DURATION, BIN)
    b_centers, b_rate = compute_rate_series(bursty_reqs,  DURATION, BIN)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Bursty vs Uniform Traffic  —  m-large, {TARGET_RATE} req/s target, {DURATION} s\n"
        f"Bursty = clients with CV ≥ {MIN_CV}  (high inter-arrival variance)",
        fontsize=13, fontweight="bold"
    )

    # Panel 1 — Instantaneous rate
    ax = axes[0, 0]
    ax.plot(u_centers, u_rate, color="steelblue", linewidth=1.0, alpha=0.75, label="Uniform")
    ax.plot(b_centers, b_rate, color="crimson",   linewidth=1.2, alpha=0.85, label="Bursty")
    ax.axhline(TARGET_RATE, color="black", linestyle="--", linewidth=0.9, label="Target")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Request rate (req/s)")
    ax.set_title(f"Instantaneous Rate ({BIN} s bins)")
    ax.legend()
    ax.set_xlim(0, DURATION)
    ax.grid(True, alpha=0.3)

    # Panel 2 — Cumulative arrivals
    ax = axes[0, 1]
    ax.plot(np.sort(u_ts), np.arange(1, len(u_ts) + 1),
            color="steelblue", linewidth=1.0, alpha=0.75, label="Uniform")
    ax.plot(np.sort(b_ts), np.arange(1, len(b_ts) + 1),
            color="crimson",   linewidth=1.2, alpha=0.85, label="Bursty")
    ideal_x = np.array([0, DURATION])
    ax.plot(ideal_x, ideal_x * TARGET_RATE, "k--", linewidth=0.8, label="Ideal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative requests")
    ax.set_title("Cumulative Arrivals")
    ax.legend()
    ax.set_xlim(0, DURATION)
    ax.grid(True, alpha=0.3)

    # Panel 3 — IAT distributions (clipped at 99th pct)
    ax = axes[1, 0]
    clip = np.percentile(np.concatenate([u_iats, b_iats]), 99)
    ax.hist(u_iats[u_iats <= clip], bins=80, density=True,
            color="steelblue", alpha=0.6, edgecolor="none", label="Uniform")
    ax.hist(b_iats[b_iats <= clip], bins=80, density=True,
            color="crimson",   alpha=0.6, edgecolor="none", label="Bursty")
    ax.set_xlabel("Inter-arrival time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("IAT Distribution (clipped at 99th pct)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4 — Burstiness metrics bar chart
    ax = axes[1, 1]
    metric_labels = ["Avg rate\n(req/s)", "Rate std\n(req/s)", "IAT std\n(ms)", "IAT p99\n(ms)"]
    u_vals = [
        len(uniform_reqs) / DURATION,
        u_rate.std(),
        u_iats.std(),
        np.percentile(u_iats, 99),
    ]
    b_vals = [
        len(bursty_reqs) / DURATION,
        b_rate.std(),
        b_iats.std(),
        np.percentile(b_iats, 99),
    ]
    x     = np.arange(len(metric_labels))
    width = 0.35
    bars_u = ax.bar(x - width/2, u_vals, width, color="steelblue", alpha=0.8, label="Uniform")
    bars_b = ax.bar(x + width/2, b_vals, width, color="crimson",   alpha=0.8, label="Bursty")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title("Burstiness Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in list(bars_u) + list(bars_b):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure → {OUT_PNG}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    for label, reqs, iats, rates in [
        ("Uniform", uniform_reqs, u_iats, u_rate),
        ("Bursty ", bursty_reqs,  b_iats, b_rate),
    ]:
        print(
            f"  {label}  requests={len(reqs):6d}  "
            f"avg_rate={len(reqs)/DURATION:6.2f} req/s  "
            f"rate_std={rates.std():6.2f}  "
            f"IAT_mean={iats.mean():7.2f} ms  "
            f"IAT_p99={np.percentile(iats, 99):7.2f} ms"
        )


if __name__ == "__main__":
    main()
