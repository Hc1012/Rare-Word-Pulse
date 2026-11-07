"""Create a readable overview from the verified saved analysis; CPU only.

Run `python reproduce.py` first to reconstruct every plotted mean and interval
from raw_scores.npz. This script only changes the presentation.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent


def main():
    arm = json.loads((ROOT / "analysis.json").read_text())["arms"]["lr_5e5"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
    fig = plt.figure(figsize=(9, 9), dpi=160, facecolor="#f8fafb")
    ink, muted = "#152d40", "#506274"
    fig.text(.085, .95, "RARE-WORD LEARNING  /  CONTROLLED EXPERIMENT",
             fontsize=10.5, weight="bold", color="#167a79")
    fig.text(.085, .903, "One exposure. Two contexts.", fontsize=25,
             weight="bold", color=ink)
    fig.text(.085, .859, "Nearly the same score increase.", fontsize=23,
             weight="bold", color=ink)
    fig.text(.085, .817, "98M-parameter model · 200 matched word pairs", fontsize=14, color=muted)

    ax = fig.add_axes([.145, .365, .79, .38], facecolor="#f8fafb")
    for key, label, color, style in [
        ("own", "Fits the context", "#087f8c", "-"),
        ("other", "Doesn’t fit the context", "#bf571a", "--"),
    ]:
        rows = arm["curves"][key]
        x = [r["offset"] for r in rows]
        y = [r["mean_nats"] for r in rows]
        ax.fill_between(x, [r["ci95_nats"][0] for r in rows],
                        [r["ci95_nats"][1] for r in rows], color=color, alpha=.10, lw=0)
        ax.plot(x, y, label=label, color=color, ls=style, lw=3)
    ax.axhline(0, color="#718392", lw=.8)
    ax.set(xlim=(-1, 51), ylim=(-.025, .52))
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.set_xlabel("Training updates after exposure", fontsize=14, labelpad=11)
    ax.set_ylabel("Word-score gain vs controls (nats)", fontsize=12.5, labelpad=10)
    ax.tick_params(colors=muted, labelsize=12, length=0, pad=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#bdc9d1")
    ax.grid(axis="y", color="#dce4e9", lw=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(-.065, 1.17),
              frameon=False, fontsize=13, handlelength=2.2, columnspacing=1.3)

    fig.text(.085, .268, "A persistent context-specific gain was not resolved.",
             fontsize=16.5, weight="bold", color=ink)
    result = arm["components"]["context"]
    lo, hi = result["ci95_nats"]
    fig.text(.085, .227,
             f"Difference: {result['mean_nats']:+.3f} nats   |   95% CI: {lo:+.3f} to {hi:+.3f}",
             fontsize=14.5, color=ink)
    fig.text(.085, .174, "Difference averages updates 10–50. Lines show changes from",
             fontsize=11.5, color=muted)
    fig.text(.085, .148, "pre-exposure scores, minus matched controls. Shading: pointwise 95% CIs.",
             fontsize=11.5, color=muted)
    fig.text(.085, .108, "One training seed · One completed learning rate · Single exposures",
             fontsize=11.5, color=muted)
    fig.text(.085, .056, "Hadi Chamas  /  github.com/Hc1012/rare-word-pulse",
             fontsize=11.5, weight="bold", color=ink)
    output = ROOT / "rare_word_linkedin.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
