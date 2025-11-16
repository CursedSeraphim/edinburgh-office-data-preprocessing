#!/usr/bin/env python3
import argparse
import os
import re
import tarfile
from datetime import datetime
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


FILENAME_RE = re.compile(
    r"inspacecam163_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.jpg$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Edinburgh office monitoring tar files and plot "
            "time-of-day coverage per day."
        )
    )
    parser.add_argument(
        "folder",
        help="Folder containing day_*.tar files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="coverage_histogram.png",
        help="Output image file for the histogram (default: coverage_histogram.png)",
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=60,
        help=(
            "Number of seconds required to mark a minute as covered "
            "(default: 60 for all seconds 0–59 present)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print some debug information while processing",
    )
    return parser.parse_args()


def find_tar_files(folder):
    tar_files = []
    for name in os.listdir(folder):
        if name.endswith(".tar") and name.startswith("day_"):
            full = os.path.join(folder, name)
            tar_files.append(full)
    # Sort by day index if possible, otherwise lexicographically
    def sort_key(path):
        base = os.path.basename(path)
        m = re.match(r"day_(\d+)\.tar$", base)
        if m:
            return int(m.group(1))
        return base

    tar_files.sort(key=sort_key)
    return tar_files


def parse_timestamp_from_name(basename):
    m = FILENAME_RE.match(basename)
    if not m:
        return None
    year, month, day, hour, minute, second = map(int, m.groups())
    return datetime(year, month, day, hour, minute, second)


def collect_per_day_minute_coverage(tar_paths, verbose=False):
    """
    Returns:
        coverage_per_day: OrderedDict[day_label -> np.array shape (1440,), dtype=bool]
        day_dates: dict[day_label -> first datetime encountered (for reference)]
    """
    coverage_per_day = OrderedDict()
    day_dates = {}

    for tar_idx, tar_path in enumerate(tar_paths):
        day_label = os.path.splitext(os.path.basename(tar_path))[0]  # e.g., "day_1"
        if verbose:
            print(f"Processing {day_label}: {tar_path}")

        # minute_index -> set(seconds)
        minute_to_seconds = defaultdict(set)
        first_dt = None

        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                basename = os.path.basename(member.name)
                dt = parse_timestamp_from_name(basename)
                if dt is None:
                    continue
                if first_dt is None:
                    first_dt = dt
                minute_index = dt.hour * 60 + dt.minute  # 0..1439
                minute_to_seconds[minute_index].add(dt.second)

        if first_dt is not None:
            day_dates[day_label] = first_dt

        # Build boolean array of length 1440 (one per minute of day)
        coverage = np.zeros(1440, dtype=bool)
        for minute_index, seconds in minute_to_seconds.items():
            # Mark as covered only if all needed seconds are present
            # (will be checked outside using `min_coverage` threshold)
            coverage[minute_index] = True if len(seconds) > 0 else False

        # Store both coverage and per-minute second counts so the threshold
        # can be applied later.
        coverage_per_day[day_label] = {
            "minute_to_seconds": minute_to_seconds,
            "first_dt": first_dt,
        }

        if verbose:
            num_minutes = len(minute_to_seconds)
            print(
                f"  Found frames in {num_minutes} distinct minute-of-day bins "
                f"(first timestamp: {first_dt})"
            )

    return coverage_per_day, day_dates


def build_coverage_matrix(coverage_per_day, min_coverage):
    """
    Convert per-day minute->seconds mapping into a 2D matrix.

    Returns:
        day_labels: list of day labels in consistent order
        minute_indices: np.array of minute indices that have any coverage
        matrix: np.array shape (n_days, len(minute_indices)), bool
    """
    day_labels = list(coverage_per_day.keys())
    n_days = len(day_labels)

    # Collect all minute indices that appear in at least one day
    all_minutes = set()
    for info in coverage_per_day.values():
        all_minutes.update(info["minute_to_seconds"].keys())

    if not all_minutes:
        raise RuntimeError("No frames found in any tar file.")

    minute_indices = np.array(sorted(all_minutes))

    matrix = np.zeros((n_days, len(minute_indices)), dtype=bool)

    for day_idx, day_label in enumerate(day_labels):
        info = coverage_per_day[day_label]
        m2s = info["minute_to_seconds"]
        for col_idx, minute_index in enumerate(minute_indices):
            seconds = m2s.get(minute_index, set())
            if len(seconds) >= min_coverage:
                matrix[day_idx, col_idx] = True

    return day_labels, minute_indices, matrix


def make_day_colormap(n_days):
    """
    Create a discrete colormap from #e5f5f9 to #2ca25f with n_days colors.
    """
    c1 = "#e5f5f9"
    c2 = "#2ca25f"
    cmap = LinearSegmentedColormap.from_list("day_cmap", [c1, c2], N=n_days)
    colors = [cmap(i) for i in range(n_days)]
    return colors


def plot_stacked_coverage(day_labels, minute_indices, matrix, output_path):
    """
    Stacked bar chart:
      x-axis: time-of-day minute bins
      y-axis: number of days covering that minute
      stack: each day is a layer with its own color
    """
    n_days, n_minutes = matrix.shape
    colors = make_day_colormap(n_days)

    # Use continuous minute indices on x-axis
    x = minute_indices
    width = 1.0  # width of each minute bin in arbitrary units

    # Stack bars
    bottoms = np.zeros(n_minutes, dtype=int)

    fig, ax = plt.subplots(figsize=(16, 6))

    for day_idx, day_label in enumerate(day_labels):
        heights = matrix[day_idx].astype(int)
        # Only draw where this day has coverage
        mask = heights > 0
        ax.bar(
            x[mask],
            heights[mask],
            width=width,
            bottom=bottoms[mask],
            label=day_label,
            color=colors[day_idx],
            linewidth=0,
        )
        bottoms[mask] += heights[mask]

    # X-axis ticks: show time-of-day (e.g., every hour)
    # minute_indices are 0..1439 but we may only show a subset; map to HH:MM
    if len(x) > 0:
        # pick ticks at whole hours present in the data
        hours = sorted(set(m // 60 for m in x))
        tick_minutes = [h * 60 for h in hours]
        tick_labels = [f"{h:02d}:00" for h in hours]
        ax.set_xticks(tick_minutes)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Time of day (minute bins)")
    ax.set_ylabel("Number of days with full coverage for that minute")
    ax.set_title("Time-of-day coverage across days (Edinburgh office dataset)")
    ax.legend(loc="upper right", ncol=2, fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    tar_files = find_tar_files(args.folder)
    if not tar_files:
        raise SystemExit(f"No day_*.tar files found in folder: {args.folder}")

    if args.verbose:
        print(f"Found {len(tar_files)} tar files:")
        for p in tar_files:
            print("  ", p)

    coverage_per_day, day_dates = collect_per_day_minute_coverage(
        tar_files, verbose=args.verbose
    )

    if args.verbose:
        print("\nPer-day first timestamps (for reference):")
        for day_label, dt in day_dates.items():
            print(f"  {day_label}: {dt}")

    day_labels, minute_indices, matrix = build_coverage_matrix(
        coverage_per_day, min_coverage=args.min_coverage
    )

    if args.verbose:
        print(
            f"\nBuilding stacked coverage for {len(day_labels)} days "
            f"over {len(minute_indices)} minute bins "
            f"(min_coverage={args.min_coverage} seconds)."
        )

    plot_stacked_coverage(day_labels, minute_indices, matrix, args.output)

    if args.verbose:
        print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
