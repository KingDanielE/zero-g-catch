"""Hour 9: result figures from the Monte Carlo CSV, plus one example throw.

    python -m scripts.plots [--csv out/monte_carlo_300.csv]

Writes out/fig_success.png (catch rate vs speed, spin and direction), out/fig_outcomes.png
(what happened, by speed), out/fig_contact.png (contact speed and finger kick at the grasp) and
out/fig_example.png (one throw, ground truth, phases shaded).
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.monte_carlo import SEED0, rate, sample_throw
from sim.evaluate import run_throw

# Reference palette (light): categorical slots in fixed order, text inks, recessive grid.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9.5, "legend.frameon": False,
    "legend.fontsize": 9,
})
OUT = Path("out")


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    num = lambda v: float(v) if v not in ("", "None") else None  # noqa: E731
    for r in rows:
        for k, v in r.items():
            if k not in ("outcome", "phase", "early_touch"):
                r[k] = num(v)
    return rows


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def binned(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        k = sum(r["outcome"] == "caught" for r in sel)
        out.append((lo, min(hi, edges[-1]), k, len(sel)))
    return out


def fig_success(rows):
    panels = [("speed_cms", [2, 5, 8, 11, 15.001], "drift speed (cm/s)", "{:.0f}–{:.0f}"),
              ("spin_deg", [0, 10, 20, 30.001], "spin rate (deg/s)", "{:.0f}–{:.0f}"),
              ("cone_deg", [0, 10, 20, 30.001], "direction, off straight across (deg)",
               "{:.0f}–{:.0f}")]
    n = len(rows)
    k_all = sum(r["outcome"] == "caught" for r in rows)
    p_all = k_all / n
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (key, edges, label, fmt) in zip(axes, panels):
        bins = binned(rows, key, edges)
        x = np.arange(len(bins))
        for xi, (lo, hi, k, m) in zip(x, bins):
            p, a, b = wilson(k, m)
            ax.plot([xi, xi], [100 * a, 100 * b], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(xi, 100 * p, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(f"{100 * p:.0f}%", (xi, 100 * p), xytext=(9, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
            ax.text(xi, 41, f"n={m}", ha="center", color=MUTED, fontsize=8)
        ax.axhline(100 * p_all, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(x, [fmt.format(lo, hi) for lo, hi, _, _ in bins])
        ax.set_xlim(-0.5, len(bins) - 0.5)
        ax.set_xlabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(38, 100)
    axes[0].set_ylabel("caught and brought to rest (%)")
    fig.suptitle(f"Catch rate over {n} randomized throws (dots: rate, bars: 95% interval, "
                 f"dashed: all throws, {100 * p_all:.0f}%)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_success.png", dpi=150)
    plt.close(fig)


def fig_outcomes(rows):
    edges = [2, 5, 8, 11, 15.001]
    groups = [("caught", ("caught",), BLUE), ("no reachable catch", ("no_plan",), ORANGE),
              ("aborted (estimate too far off to close)", ("aborted",), AQUA),
              ("held but not at rest / out of time", ("not_at_rest", "timeout"), YELLOW)]
    labels, fracs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["speed_cms"] < hi]
        labels.append(f"{lo:.0f}–{min(hi, 15):.0f} cm/s")
        fracs.append([sum(r["outcome"] in oc for r in sel) / len(sel) for _, oc, _ in groups])
    fracs = np.array(fracs) * 100
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for j, (name, _, color) in enumerate(groups):
        ax.barh(y, fracs[:, j], left=left, height=0.62, color=color, edgecolor=SURFACE,
                linewidth=2, label=name)
        for yi, l, w in zip(y, left, fracs[:, j]):
            if w >= 6:
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center",
                        color=SURFACE if j == 0 else INK, fontsize=8.5)
        left += fracs[:, j]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of throws (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handlelength=1, fontsize=8.5,
              columnspacing=1.2)
    ax.set_title("What happened, by drift speed", pad=28)
    fig.tight_layout()
    fig.savefig(OUT / "fig_outcomes.png", dpi=150)
    plt.close(fig)


def fig_contact(rows):
    g = [r for r in rows if r["kick"] is not None]
    rel = np.array([r["grasp_rel_speed"] for r in g]) * 100
    kick = np.array([r["kick"] for r in g]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, data, bins, label, title in (
        (axes[0], rel, np.arange(0, 3.01, 0.15), "relative speed, gripper vs bar (cm/s)",
         "Contact speed when the gripper starts closing"),
        (axes[1], kick, np.arange(0, 10.01, 0.5), "change in the bar's velocity (cm/s)",
         "Kick the closing fingers give the bar"),
    ):
        ax.hist(data, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
        med = np.median(data)
        ax.axvline(med, color=INK, lw=1.2)
        ax.annotate(f"median {med:.1f}", (med, ax.get_ylim()[1] * 0.92), xytext=(-5, 0),
                    textcoords="offset points", ha="right", color=INK, fontsize=8.5)
        ax.set_xlabel(label)
        ax.set_ylabel("throws")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].axvline(2.0, color=INK2, lw=1, ls=(0, (4, 3)))
    axes[0].text(2.33, axes[0].get_ylim()[1] * 0.55, "dashed: closing gate,\n2 cm/s on the estimate",
                 color=INK2, fontsize=8.5)
    fig.text(0.01, 0.01, f"{len(g)} throws where the gripper closed; ground truth",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_contact.png", dpi=150)
    plt.close(fig)


def pick_example(rows):
    """A caught throw with a brisk drift and spin, not an outlier on any metric."""
    ok = [r for r in rows if r["outcome"] == "caught" and r["speed_cms"] > 8
          and r["spin_deg"] > 15 and r["aborts"] == 0 and r["kick"] < 0.05]
    return int(min(ok, key=lambda r: r["throw"])["throw"])


PHASES = [("approach", "approach", "#eef4fc"), ("ride", "ride", "#e3eefa"),
          ("close_in", "close in", "#d6e6f8"), ("grasp", "fingers close", "#fdeee6"),
          ("brake", "brake", "#e5f6ef"), ("stopped", "at rest", "#f3f3f1")]


def fig_example(throw):
    seed = SEED0 + throw
    target, params = sample_throw(seed)
    r = run_throw(seed, target, keep=True)
    L, c = r["log"], r["catcher"]
    t = L["t"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 9.5), sharex=True)
    series = [(L["dist"] * 100, "tool point to bar centre (cm)", "linear"),
              (np.maximum(L["rel_speed"] * 100, 0.05), "gripper vs its target on the bar (cm/s)", "log"),
              (L["speed"] * 100, "bar drift speed (cm/s)", "linear"),
              (L["spin"], "bar spin (deg/s)", "linear"),
              (L["force"], "grip force on bar (N)", "linear")]
    for ax, (y, label, scale) in zip(axes, series):
        for key, _, color in PHASES:
            sel = L["phase"] == key
            if sel.any():
                ax.axvspan(t[sel][0], t[sel][-1], color=color, lw=0)
        ax.plot(t, y, color=BLUE)
        ax.set_yscale(scale)
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    top = axes[0]
    for key, name, _ in PHASES:
        sel = L["phase"] == key
        if sel.any():
            top.text((t[sel][0] + t[sel][-1]) / 2, 1.03, name, transform=top.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8, color=INK2,
                     rotation=0 if t[sel][-1] - t[sel][0] > 0.25 else 90)
    axes[-1].set_xlabel("time since release (s)")
    axes[-1].set_xlim(t[0], t[-1])
    fig.suptitle(f"One throw, ground truth: {params['speed'] * 100:.1f} cm/s drift, "
                 f"{params['spin_deg']:.0f} deg/s spin, {params['cone_deg']:.0f} deg off straight "
                 f"across (throw {throw})", x=0.01, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    fig.text(0.01, 0.005, f"planned {c.plan.planned_at:.2f} s after release for a catch at "
             f"{c.plan.t:.2f} s; brought to rest in {c.brake_T:.2f} s", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(OUT / "fig_example.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/monte_carlo_300.csv")
    ap.add_argument("--example", type=int, default=None, help="throw index for fig_example")
    args = ap.parse_args()
    rows = load(args.csv)
    OUT.mkdir(exist_ok=True)
    fig_success(rows)
    fig_outcomes(rows)
    fig_contact(rows)
    throw = args.example if args.example is not None else pick_example(rows)
    fig_example(throw)
    n, k = len(rows), sum(r["outcome"] == "caught" for r in rows)
    print(f"{args.csv}: caught {rate(k, n)}; example throw {throw}")
    print("saved out/fig_success.png, fig_outcomes.png, fig_contact.png, fig_example.png")


if __name__ == "__main__":
    main()
