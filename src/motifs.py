import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from utils import motifs, to_heatmap, ccdf  # assumes these are defined in utils


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logger():
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "motifs.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logger()


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate motif heatmaps and CCDF plots")
    parser.add_argument("--dataset", type=str, required=True, help="Path to parquet dataset")
    parser.add_argument("--cycles-dir", type=str, required=True, help="Directory containing cycles")
    parser.add_argument("--time-delta", type=float, default=1.0, help="Time window delta in minutes for motif counting")
    return parser.parse_args()


# ---------------------------------------------------------------------
# CREATE OUTPUT STRUCTURE
# ---------------------------------------------------------------------
def create_output_dirs(base_dir):
    dirs = {}
    for category in ["normal", "automated", "wash", "ratio"]:
        path = base_dir / category
        path.mkdir(parents=True, exist_ok=True)
        dirs[category] = path
    return dirs


# ---------------------------------------------------------------------
# PLOT HEATMAP
# ---------------------------------------------------------------------
def plot_heatmap(motif_array, out_file, title="Motifs Count"):
    x_labels = ["PRE-III", "PRE-IIO", "PRE-IOI", "PRE-IOO", "MID-III", "MID-IIO"]
    y_labels = ["PRE-III", "PRE-IIO", "PRE-IOI", "PRE-IOO", "MID-III", "MID-IIO"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(motif_array, annot=True, fmt=".2e", cmap="YlGnBu",
                xticklabels=x_labels, yticklabels=y_labels, ax=ax,
                annot_kws={"size": 10})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Third Transaction")
    ax.set_ylabel("First Two Transactions")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


# ---------------------------------------------------------------------
# PLOT CCDF
# ---------------------------------------------------------------------
def plot_ccdf(motifs_dict, out_file, title="CCDF Motifs per Category"):
    categorias = {
        "2 nodes, one direction": [24, 25, 26, 27],
        "2 nodes, mixed directions": [28, 29, 30, 31],
        "3 node star, all incoming": list(range(0, 8)),
        "3 node star, all outgoing": list(range(8, 16)),
        "3 node star, mixed directions": list(range(16, 24)),
        "Triangles": list(range(32, 40))
    }

    conteos_por_categoria = {cat: [] for cat in categorias}

    for nodo_id, counts in motifs_dict.items():
        for categoria, indices in categorias.items():
            total = sum(counts[i] for i in indices)
            conteos_por_categoria[categoria].append(total)

    plt.figure(figsize=(10, 6))
    for categoria, counts in conteos_por_categoria.items():
        counts = np.array(counts)
        x, y = ccdf(counts)
        if len(x) > 0 and len(y) > 0:
            plt.loglog(x, y, label=categoria, linewidth=2)

    plt.title(title + " (NFT, δ = 1 min)", fontsize=14)
    plt.xlabel("Number of Motifs (log scale)", fontsize=12)
    plt.ylabel("P(X ≥ x) (log scale)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    cycles_dir = Path(args.cycles_dir)
    output_base = Path("output/motifs")
    dirs = create_output_dirs(output_base)

    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset loaded: {dataset_path}")
    
    # Assuming cycles_dir contains pickles per collection
    cycle_files = list(cycles_dir.glob("*_cycles.pkl"))

    for cycle_file in cycle_files:
        collection = cycle_file.stem.replace("_cycles", "")
        logger.info(f"Processing collection: {collection}")
        cycles = pd.read_pickle(cycle_file)

        # Compute motifs
        motif_array, normal_array, automated_array, wash_array = motifs(df, cycles)

        # Convert to heatmaps
        normal_heatmap = to_heatmap(normal_array)
        automated_heatmap = to_heatmap(automated_array)
        wash_heatmap = to_heatmap(wash_array)
        ratio_heatmap = automated_heatmap / (normal_heatmap + 1e-9)

        # Save heatmaps
        plot_heatmap(normal_heatmap, dirs["normal"] / f"{collection}_normal.png", title=f"{collection} - Normal Motifs")
        plot_heatmap(automated_heatmap, dirs["automated"] / f"{collection}_automated.png", title=f"{collection} - Automated Motifs")
        plot_heatmap(wash_heatmap, dirs["wash"] / f"{collection}_wash.png", title=f"{collection} - Wash Motifs")
        plot_heatmap(ratio_heatmap, dirs["ratio"] / f"{collection}_ratio.png", title=f"{collection} - Automated / Normal Ratio")

        # Plot CCDFs
        plot_ccdf(normal_array, dirs["normal"] / f"{collection}_normal_CCDF.png", title=f"{collection} - Normal")
        plot_ccdf(automated_array, dirs["automated"] / f"{collection}_automated_CCDF.png", title=f"{collection} - Automated")

        logger.info(f"Motifs heatmaps and CCDFs saved for collection: {collection}")


if __name__ == "__main__":
    main()
