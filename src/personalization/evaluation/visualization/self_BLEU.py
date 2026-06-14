import json
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

_DEFAULT_PERSONAS_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../personas/personas_all.json"
)

_THESIS_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


def _load_persona_labels(personas_path=None):
    """Return {persona_id: display_name} with the leading 'The ' stripped."""
    path = personas_path or _DEFAULT_PERSONAS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    labels = {}
    for p in data:
        name = p["display_name"]
        if name.startswith("The "):
            name = name[4:]
        labels[p["persona_id"]] = name
    return labels


def _wrap(label, width=14):
    return "\n".join(textwrap.wrap(label, width, break_long_words=False))


def plot_intra(results, save_path=None, personas_path=None):
    persona_labels = _load_persona_labels(personas_path)
    personas = list(results["intra"].keys())

    df = pd.DataFrame([
        {"persona": _wrap(persona_labels.get(p, p)), "score": s}
        for p in personas
        for s in results["intra"][p]["scores"]
    ])

    n = len(personas)

    with plt.rc_context(_THESIS_RC):
        fig, ax = plt.subplots(figsize=(4 + n * 1.0, 4.5))

        palette = ["#b8cfe8"] * n

        sns.boxplot(
            data=df, x="persona", y="score",
            palette=palette, width=0.5, fliersize=3,
            linewidth=0.9, ax=ax,
            flierprops={"marker": "o", "markerfacecolor": "#888888",
                        "markersize": 3, "alpha": 0.5, "linestyle": "none"},
            medianprops={"color": "#1a4c7a", "linewidth": 1.8},
            boxprops={"edgecolor": "#555555"},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )

        ax.set_xlabel("Persona", labelpad=6)
        ax.set_ylabel("Self-BLEU", labelpad=6)
        ax.set_title(None)
        ax.tick_params(axis="x", rotation=0, labelsize=9)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#cccccc")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            print(f"Saved → {save_path}")
    return fig


def plot_heatmap(results, save_path=None, personas_path=None):
    persona_labels = _load_persona_labels(personas_path)
    personas = list(results["intra"].keys())
    n = len(personas)

    tick_labels = [_wrap(persona_labels.get(p, p), width=12) for p in personas]

    matrix = pd.DataFrame(
        np.zeros((n, n)), index=tick_labels, columns=tick_labels
    )
    for p, lbl in zip(personas, tick_labels):
        matrix.loc[lbl, lbl] = results["intra"][p]["mean"]
    for pair, data in results["inter"].items():
        p_a, p_b = [x.strip() for x in pair.split("vs")]
        la = _wrap(persona_labels.get(p_a, p_a), width=12)
        lb = _wrap(persona_labels.get(p_b, p_b), width=12)
        matrix.loc[la, lb] = matrix.loc[lb, la] = data["mean"]

    with plt.rc_context(_THESIS_RC):
        fig, ax = plt.subplots(figsize=(2 + n * 1.3, 1.5 + n * 1.1))

        sns.heatmap(
            matrix, ax=ax,
            annot=True, fmt=".3f", annot_kws={"size": 9},
            cmap="Blues", linewidths=0.4, linecolor="#e0e0e0",
            cbar_kws={"label": "Średnie Self-BLEU", "shrink": 0.75},
        )
        for i in range(n):
            ax.add_patch(mpatches.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor="#444444", lw=1.8
            ))

        ax.set_title(None)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Średnie Self-BLEU", fontsize=9)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            print(f"Saved → {save_path}")
    return fig

def print_results(results: dict) -> None:
    print("=" * 60)
    print("INTRA-PERSONA SELF-BLEU  (lower = more diverse within)")
    print("=" * 60)
    for persona, data in sorted(results["intra"].items()):
        print(f"  {persona:<20}  mean = {data['mean']:.4f} ± {data['std']:.4f}")

    print()
    print("=" * 60)
    print("INTER-PERSONA SELF-BLEU  (lower = more different between)")
    print("=" * 60)
    for pair, data in sorted(results["inter"].items()):
        print(f"  {pair:<35}  mean = {data['mean']:.4f} ± {data['std']:.4f} "
              f"(A→B={data['mean_a2b']:.4f} ± {data['std_a2b']:.4f} , B→A={data['mean_b2a']:.4f} ± {data['std_b2a']:.4f})")