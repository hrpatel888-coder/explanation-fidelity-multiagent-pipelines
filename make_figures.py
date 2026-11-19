"""Regenerates the four figures from results.json (2D and 3D versions, 300 dpi).

Run from the repository root:  python make_figures.py
Output: figures/2d/*.png and figures/3d/*.png

Error bars show one standard deviation. In the 3D versions the lower end of an
error bar is clipped at zero so that it does not pass through the floor of the plot.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
BLUE, GREEN, RED, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#DD8452"

with open("results.json") as f:
    r = json.load(f)
fid, con, fth, perf = r["fidelity"], r["consistency"], r["faithfulness"], r["performance"]

os.makedirs("figures/2d", exist_ok=True)
os.makedirs("figures/3d", exist_ok=True)

# data for each figure -------------------------------------------------------
fig1 = dict(
    labels=["Hop 1\n(Retriever)", "Hop 2\n(Selector)", "Hop 3\n(Classifier)", "Baseline\n(Single-agent)"],
    values=[fid["hop1_mean"], fid["hop2_mean"], fid["hop3_mean"], fid["baseline_mean"]],
    errors=[fid["hop1_std"], fid["hop2_std"], fid["hop3_std"], fid["baseline_std"]],
    colors=[BLUE, BLUE, BLUE, GREEN],
    ylabel="Explanation fidelity\n(precision@k vs. ground truth)",
    title="Figure 1. Explanation fidelity across pipeline hops",
    name="figure1_fidelity")
fig2 = dict(
    labels=["Hop 1 to Hop 2", "Hop 2 to Hop 3", "Hop 1 to Hop 3\n(cumulative)"],
    values=[con["hop1_to_hop2_mean"], con["hop2_to_hop3_mean"], con["hop1_to_hop3_mean"]],
    errors=None, colors=[RED] * 3,
    ylabel="Explanation consistency\n(retention rate)",
    title="Figure 2. Hop-to-hop explanation consistency",
    name="figure2_consistency")
fig3 = dict(
    labels=["Chained\n(3-hop)", "Baseline\n(single-agent)"],
    values=[fth["chained_mean"], fth["baseline_mean"]],
    errors=[fth["chained_std"], fth["baseline_std"]],
    colors=[BLUE, GREEN],
    ylabel="Faithfulness\n(deletion-based probability drop)",
    title="Figure 3. Explanation faithfulness",
    name="figure3_faithfulness")
single = [fig1, fig2, fig3]

# 2D -------------------------------------------------------------------------
for d in single:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(d["values"]))
    ax.bar(x, d["values"], yerr=d["errors"], capsize=5, color=d["colors"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(d["labels"])
    ax.set_ylabel(d["ylabel"])
    ax.set_title(d["title"])
    ax.set_ylim(0, 1.1)
    top = np.array(d["values"]) + (np.array(d["errors"]) if d["errors"] else 0)
    for xi, v, t in zip(x, d["values"], top):
        ax.text(xi, max(v, 0) + (0.03 if not d["errors"] else 0.03), f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"figures/2d/{d['name']}.png", dpi=DPI)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.6))
x = np.arange(2)
w = 0.35
acc = [perf["chained_accuracy"], perf["baseline_accuracy"]]
f1 = [perf["chained_macro_f1"], perf["baseline_macro_f1"]]
ax.bar(x - w / 2, acc, w, label="Accuracy", color=BLUE, edgecolor="black", linewidth=0.6)
ax.bar(x + w / 2, f1, w, label="Macro-F1", color=ORANGE, edgecolor="black", linewidth=0.6)
for xi, a, b in zip(x, acc, f1):
    ax.text(xi - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=10)
    ax.text(xi + w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(["Chained\n(3-hop)", "Baseline\n(single-agent)"])
ax.set_ylabel("Score")
ax.set_title("Figure 4. Predictive performance")
ax.set_ylim(0, 1.15)
ax.legend()
fig.tight_layout()
fig.savefig("figures/2d/figure4_performance.png", dpi=DPI)
plt.close(fig)

# 3D -------------------------------------------------------------------------
def style_3d(ax, ylabel_z, title, zmax, xmax):
    ax.set_xlim(-0.4, xmax)
    ax.set_ylim(-0.4, 1.0)
    ax.set_yticks([])
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((xmax * 0.55, 0.7, 1.0))
    ax.set_zlabel(ylabel_z, labelpad=14, fontsize=10)
    ax.set_title(title, fontsize=12, pad=14)
    ax.view_init(elev=20, azim=-78)

for d in single:
    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    n = len(d["values"])
    for i, (v, c) in enumerate(zip(d["values"], d["colors"])):
        ax.bar3d(i, 0, 0, 0.6, 0.6, v, color=c, edgecolor="black", linewidth=0.6, shade=True)
        cx, cy = i + 0.3, 0.3
        top = v
        if d["errors"]:
            e = d["errors"][i]
            lo, hi = max(v - e, 0), v + e
            ax.plot([cx, cx], [cy, cy], [lo, hi], color="black", linewidth=1.3, marker="_", markersize=7)
            top = hi
        ax.text(cx, cy, top + 0.1, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(n) + 0.3)
    ax.set_xticklabels(d["labels"], fontsize=9)
    style_3d(ax, d["ylabel"], d["title"] + " (3D)", 1.2, n + 0.2)
    fig.savefig(f"figures/3d/{d['name']}_3d.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

fig = plt.figure(figsize=(8.5, 6))
ax = fig.add_subplot(111, projection="3d")
for g, (a, b) in enumerate(zip(acc, f1)):
    x0 = g * 2.0
    ax.bar3d(x0, 0, 0, 0.6, 0.6, a, color=BLUE, edgecolor="black", linewidth=0.6, shade=True)
    ax.bar3d(x0 + 0.8, 0, 0, 0.6, 0.6, b, color=ORANGE, edgecolor="black", linewidth=0.6, shade=True)
    ax.text(x0 + 0.3, 0.3, a + 0.1, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(x0 + 1.1, 0.3, b + 0.1, f"{b:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks([0.7, 2.7])
ax.set_xticklabels(["Chained\n(3-hop)", "Baseline\n(single-agent)"], fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Accuracy"), Patch(color=ORANGE, label="Macro-F1")], loc="upper left")
style_3d(ax, "Score", "Figure 4. Predictive performance (3D)", 1.2, 4.2)
fig.savefig("figures/3d/figure4_performance_3d.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Figures written to figures/2d and figures/3d")
