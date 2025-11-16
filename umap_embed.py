#!/usr/bin/env python3
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# Matches grayscale  "1x1", "32x18"
# and color pixels   "1x1_r", "1x1_g", "1x1_b"
PIXEL_RE = re.compile(r"^\d+x\d+(?:_[rgb])?$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run UMAP on pixel columns (<x>x<y>[,_r/_g/_b]) of a trajectories CSV, "
            "append 2D coordinates (x,y), and save a scatter plot colored by 'line'."
        )
    )
    parser.add_argument(
        "csv",
        help="Input CSV (output of the trajectory generator script)",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        default="trajectories_umap.csv",
        help="Output CSV with appended x,y columns (default: trajectories_umap.csv)",
    )
    parser.add_argument(
        "--plot",
        "-p",
        default="trajectories_umap.png",
        help="Output scatterplot image file (default: trajectories_umap.png)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="UMAP metric (default: euclidean)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP (default: 42)",
    )
    return parser.parse_args()


def find_pixel_columns(df: pd.DataFrame):
    pixel_cols = [c for c in df.columns if PIXEL_RE.match(c)]
    if not pixel_cols:
        raise SystemExit(
            "No pixel columns found. Expected columns like '1x1', '1x2', "
            "'1x1_r', etc."
        )
    return pixel_cols


def run_umap(X: np.ndarray, args) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        n_components=2,
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def make_scatter(df: pd.DataFrame, plot_path: str):
    # Color by 'line' as categorical
    if "line" not in df.columns:
        raise SystemExit("Column 'line' not found in CSV; needed for coloring.")

    if "x" not in df.columns or "y" not in df.columns:
        raise SystemExit("Columns 'x' and 'y' not found; UMAP embedding missing.")

    cat = pd.Categorical(df["line"])
    codes = cat.codes  # 0..K-1
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        df["x"],
        df["y"],
        c=codes,
        s=5,
        alpha=0.7,
        cmap="tab20",  # categorical-ish palette
    )
    ax.set_xlabel("x (UMAP)")
    ax.set_ylabel("y (UMAP)")
    ax.set_title("UMAP projection of frames (colored by line)")

    # Optional colorbar (useful even if many lines)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("line (trajectory/day id)")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Get pixel feature matrix
    pixel_cols = find_pixel_columns(df)
    X = df[pixel_cols].to_numpy(dtype=np.float32)

    # Run UMAP
    embedding = run_umap(X, args)

    # Append coordinates
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]

    # Save updated CSV
    df.to_csv(args.output_csv, index=False)

    # Scatter plot colored by 'line'
    make_scatter(df, args.plot)


if __name__ == "__main__":
    main()
