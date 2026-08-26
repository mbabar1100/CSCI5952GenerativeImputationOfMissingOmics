from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "repro"
FIGURES = ROOT / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

NAVY = "#17324D"
BLUE = "#2878B5"
TEAL = "#31A6A0"
ORANGE = "#E58B3D"
RED = "#C94C4C"
LIGHT = "#F2F6F8"
GRAY = "#657786"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    boxes = [
        (0.35, 2.95, 2.35, 1.05, "TCGA-BRCA", "769 patients\nRNA, methylation, and miRNA"),
        (3.83, 2.95, 2.35, 1.05, "Feature selection", "1,000 RNA + 1,000 DNA methylation\n+ 503 miRNA features"),
        (7.30, 2.95, 2.35, 1.05, "Similarity graph", "2,503 feature nodes\ncosine kNN; k = 25"),
        (7.30, 0.55, 2.35, 1.05, "GAT embedding", "4 layers; 64 dimensions\nPAM50-guided regression"),
        (3.83, 0.55, 2.35, 1.05, "Leiden partition", "weighted RB objective\nresolution = 0.25"),
    ]
    colors = [NAVY, BLUE, TEAL, ORANGE, RED]
    for (x, y, w, h, title, subtitle), color in zip(boxes, colors):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=1.1,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(x + w / 2, y + 0.72, title, ha="center", va="center", weight="bold", color=color, fontsize=9)
        ax.text(x + w / 2, y + 0.30, subtitle, ha="center", va="center", fontsize=6.65, color=NAVY)

    for left, right in zip(boxes[:2], boxes[1:3]):
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.08, 3.48),
                (right[0] - 0.08, 3.48),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color=GRAY,
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (8.47, 2.88),
            (8.47, 1.68),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=GRAY,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.20, 1.08),
            (6.28, 1.08),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=GRAY,
        )
    )

    branch_x, branch_y, branch_w, branch_h = 0.35, 0.55, 2.35, 1.05
    ax.add_patch(
        FancyBboxPatch(
            (branch_x, branch_y),
            branch_w,
            branch_h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#7B4FA3",
            facecolor="#F7F2FA",
        )
    )
    ax.text(
        branch_x + branch_w / 2,
        branch_y + 0.72,
        "Optional K-means refinement",
        ha="center",
        va="center",
        weight="bold",
        fontsize=8.8,
        color="#6B3F8E",
    )
    ax.text(
        branch_x + branch_w / 2,
        branch_y + 0.30,
        "K-means if a module has >30 nodes\nat most 3 submodules",
        ha="center",
        va="center",
        fontsize=7.1,
        color=NAVY,
    )
    ax.add_patch(
        FancyArrowPatch(
            (3.72, 1.08),
            (2.78, 1.08),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color="#7B4FA3",
        )
    )
    ax.text(5.0, 5.0, "BioNeuralNet Leiden extension and evaluation workflow", ha="center", weight="bold", fontsize=11, color=NAVY)
    ax.text(5.0, 4.55, "Primary task: phenotype-guided multi-omics feature-module discovery", ha="center", fontsize=8.5, color=GRAY)
    save(fig, "figure1_workflow")


def umap_figure() -> None:
    data = np.load(RESULTS / "feature_results.npz", allow_pickle=True)
    coords, pure, hybrid = data["umap"], data["pure"], data["hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), constrained_layout=True)
    for ax, labels, title in [
        (axes[0], pure, "A  Leiden: 12 modules"),
        (axes[1], hybrid, "B  Leiden + local K-means: 22 modules"),
    ]:
        unique, counts = np.unique(labels, return_counts=True)
        order = unique[np.argsort(counts)[::-1]]
        palette = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(order)))
        mapping = {label: color for label, color in zip(order, palette)}
        colors = np.array([mapping[label] for label in labels])
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5.0, alpha=0.78, linewidths=0, rasterized=True)
        for label in order[:5]:
            mask = labels == label
            center = np.median(coords[mask], axis=0)
            ax.text(
                center[0],
                center[1],
                str(int(label)),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color="black",
                bbox={"boxstyle": "circle,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
        ax.set_title(title, loc="left", weight="bold", color=NAVY)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(True)
        for spine in ax.spines.values():
            spine.set_color("#D7E0E5")
    save(fig, "figure2_umap")


def metrics_figure() -> None:
    with (RESULTS / "metrics.json").open() as handle:
        metrics = json.load(handle)
    module, patient = metrics["module"], metrics["patient"]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), constrained_layout=True)

    pure_sizes = module["pure_sizes"]
    hybrid_sizes = module["hybrid_sizes"]
    axes[0, 0].bar(np.arange(len(pure_sizes)) - 0.18, pure_sizes, width=0.36, color=BLUE, label="Leiden")
    axes[0, 0].bar(np.arange(len(hybrid_sizes)) + 0.18, hybrid_sizes, width=0.36, color=ORANGE, label="Hybrid")
    axes[0, 0].set_title("A  Module-size distributions", loc="left", weight="bold", color=NAVY)
    axes[0, 0].set_xlabel("Modules ranked by size")
    axes[0, 0].set_ylabel("Number of features")
    axes[0, 0].legend(frameon=False, fontsize=8)

    names = ["Graph\nmodularity", "Embedding\nsilhouette", "Expression\nsilhouette"]
    pure_values = [module["modularity_pure"], module["silhouette_embedding_pure"], module["silhouette_expression_pure"]]
    hybrid_values = [module["modularity_hybrid"], module["silhouette_embedding_hybrid"], module["silhouette_expression_hybrid"]]
    x = np.arange(3)
    axes[0, 1].bar(x - 0.18, pure_values, width=0.36, color=BLUE, label="Leiden")
    axes[0, 1].bar(x + 0.18, hybrid_values, width=0.36, color=ORANGE, label="Hybrid")
    axes[0, 1].axhline(0, color="#AAB7C0", linewidth=0.8)
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylim(-0.18, 0.78)
    axes[0, 1].set_title("B  Structural and geometric quality", loc="left", weight="bold", color=NAVY)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].bar([0, 1], [module["davies_bouldin_pure"], module["davies_bouldin_hybrid"]], color=[BLUE, ORANGE], width=0.6)
    axes[1, 0].set_xticks([0, 1], ["Leiden", "Hybrid"])
    axes[1, 0].set_ylabel("Davies-Bouldin index (lower is better)")
    axes[1, 0].set_title("C  Embedding-space dispersion", loc="left", weight="bold", color=NAVY)
    for i, value in enumerate([module["davies_bouldin_pure"], module["davies_bouldin_hybrid"]]):
        axes[1, 0].text(i, value + 1.0, f"{value:.1f}", ha="center", fontsize=8)

    methods = ["K-means", "Leiden", "Hybrid"]
    ari = [patient["kmeans_ari"], patient["leiden_ari"], patient["hybrid_ari"]]
    nmi = [patient["kmeans_nmi"], patient["leiden_nmi"], patient["hybrid_nmi"]]
    x = np.arange(3)
    axes[1, 1].bar(x - 0.18, ari, width=0.36, color=TEAL, label="ARI")
    axes[1, 1].bar(x + 0.18, nmi, width=0.36, color="#7B4FA3", label="NMI")
    axes[1, 1].set_xticks(x, methods)
    axes[1, 1].set_ylim(0, 0.56)
    axes[1, 1].set_ylabel("Agreement with PAM50")
    axes[1, 1].set_title("D  Secondary patient-clustering stress test", loc="left", weight="bold", color=NAVY)
    axes[1, 1].legend(frameon=False, fontsize=8)
    save(fig, "figure3_metrics")


if __name__ == "__main__":
    workflow()
    umap_figure()
    metrics_figure()
