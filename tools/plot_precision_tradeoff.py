"""Render exportable PR/latency figures from evaluate_precision_tradeoff.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
from sklearn.metrics import precision_recall_curve

MODES = ("fp32", "fp16", "bf16", "int8", "int4", "int8-fp32-input")
NAMES = ("FP32", "FP16", "BF16", "INT8", "INT4", "INT8 hybrid")
COLORS = ("#334155", "#0072B2", "#8B5CF6", "#009E73", "#D55E00", "#B8860B")
MARKERS = ("o", "s", "D", "^", "v", "P")
STYLES = dict(zip(MODES, zip(NAMES, COLORS, MARKERS)))
SUITES = {"ten": "TEN", "voxconverse": "VoxConverse"}


def save(fig, folder, stem):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def style(ax):
    ax.set_facecolor("#FFFFFF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", length=0, pad=7)
    ax.grid(color="#E2E8F0", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def tradeoff(folder, data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharex="col", sharey="row")
    fig.subplots_adjust(left=.09, right=.97, bottom=.19, top=.82, hspace=.28, wspace=.16)
    fig.suptitle("Akiki precision vs. latency", x=.09, y=.968, ha="left", fontsize=23, weight="bold", color="#0F172A")
    fig.text(.09, .921, f"Package {data['package_version']}  ·  Same model across all precisions  ·  FFT fixed at FP32", fontsize=11, color="#475569")
    fig.text(.09, .882, "Higher average precision and lower latency are better. Axes are zoomed to show small differences.", fontsize=10, color="#475569")
    offsets = {"fp32": (0, 16), "fp16": (0, -24), "bf16": (-8, 17), "int8": (0, 16), "int4": (0, 16), "int8-fp32-input": (13, -31)}
    for row, (suite, display) in enumerate(SUITES.items()):
        metrics = data["evaluations"][suite]
        aps = np.asarray([metrics["models"][mode]["average_precision"] * 100 for mode in MODES])
        span = max(float(np.ptp(aps)), .04)
        for col, case in enumerate(("rust_ffi", "rust_push")):
            ax = axes[row, col]
            style(ax)
            ax.set_ylim(float(aps.min()) - .28 * span, float(aps.max()) + .38 * span)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            xs = [data["latency"][m][case]["median_us"] for m in MODES]
            xspan = max(xs) - min(xs)
            ax.set_xlim(min(xs) - .18 * xspan, max(xs) + .20 * xspan)
            for mode, ap in zip(MODES, aps):
                label, color, marker = STYLES[mode]
                timing = data["latency"][mode][case]
                x = timing["median_us"]
                rounds = timing["round_medians_us"]
                ax.errorbar(x, ap, xerr=[[x - min(rounds)], [max(rounds) - x]],
                            fmt=marker, color=color, markersize=8.5, markeredgecolor="white",
                            markeredgewidth=1, elinewidth=1.4, capsize=3, zorder=4)
                ax.annotate(label, (x, ap), xytext=offsets[mode], textcoords="offset points",
                            ha="center", va="center", fontsize=9.5, color=color, weight="bold",
                            arrowprops=dict(arrowstyle="-", color=color, linewidth=.6, shrinkB=6))
            ax.text(.025, .045, f"{display} · {metrics['recordings']} recordings", transform=ax.transAxes,
                    fontsize=10, color="#475569")
            if col == 0:
                ax.set_ylabel("Average precision (%)", labelpad=12)
            if row == 0:
                ax.set_title("Native call" if col == 0 else "Python VAD.push()", fontsize=13, weight="bold", pad=13, color="#0F172A")
            if row == 1:
                ax.set_xlabel("Median latency per 16 ms hop (µs)", labelpad=12)
    fig.text(.09, .106, "Ryzen 9 9900X · one pinned CPU core · 3 fresh runs · 2,000 warmup + 7,500 timed calls per run", fontsize=9.5, color="#475569")
    fig.text(.09, .080, "Bars: range of run medians. Native: ctypes call with precomputed pointers. Python: validation, buffering, inference and frame output.", fontsize=9, color="#475569")
    fig.text(.09, .054, "AP: exact pooled precision–recall step integral, on 32 ms pair-mean outputs. No model loading, audio decoding or audio arrival time in latency.", fontsize=9, color="#475569")
    fig.text(.09, .028, "INT8 hybrid = int8-fp32-input. These are development benchmarks; small AP differences do not establish statistical significance.", fontsize=9, color="#475569")
    save(fig, folder, "precision-vs-latency")


def curves(folder, data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.subplots_adjust(left=.075, right=.97, top=.74, bottom=.23, wspace=.23)
    fig.suptitle("Precision–recall curves by model precision", x=.075, y=.96, ha="left", fontsize=21, weight="bold", color="#0F172A")
    fig.text(.075, .9, f"Akiki {data['package_version']}  ·  Rust inference  ·  FP32 FFT  ·  High-recall region", color="#475569", fontsize=11)
    handles = [Line2D([0], [0], color=STYLES[m][1], lw=2, label=STYLES[m][0]) for m in MODES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065, .865), ncol=6, frameon=False, fontsize=10)
    for ax, (suite, display) in zip(axes, SUITES.items()):
        style(ax)
        with np.load(folder / (suite + "-predictions.npz")) as z:
            for index, mode in enumerate(MODES):
                precision, recall, _ = precision_recall_curve(z["labels"], z[mode], drop_intermediate=True)
                # Render a bounded-size curve; AP in the table uses all raw values.
                precision, recall = precision[::-1], recall[::-1]
                grid = np.linspace(.8, 1., 2001)
                indexes = np.minimum(np.searchsorted(recall, grid, side="left"), len(recall) - 1)
                ax.plot(grid * 100, precision[indexes] * 100, color=STYLES[mode][1],
                        lw=1.8, alpha=.95, linestyle=("-", "--", "-.", "-", ":", "--")[index])
        ax.set(xlim=(80, 100), ylim=(74, 100.7), xlabel="Recall (%)", ylabel="Precision (%)")
        ax.set_title(display, fontsize=13, weight="bold", pad=13, color="#0F172A")
    fig.text(.075, .135, "Curves use the same frozen 32 ms labels and pair-mean predictions as the AP/latency plot.", fontsize=10, color="#475569")
    fig.text(.075, .096, "Rendering samples 2,001 recall positions; average precision is computed exactly from all predictions, over the full recall range.", fontsize=9, color="#475569")
    fig.text(.075, .057, "INT8 hybrid = int8-fp32-input. Previously inspected development benchmarks; no threshold tuning or model training.", fontsize=9, color="#475569")
    save(fig, folder, "precision-recall-curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    data = json.loads((args.folder / "results.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#334155",
                         "figure.facecolor": "#F8FAFC", "svg.fonttype": "none", "pdf.fonttype": 42})
    tradeoff(args.folder, data)
    curves(args.folder, data)


if __name__ == "__main__":
    main()
