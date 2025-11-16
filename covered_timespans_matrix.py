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
            "time-of-day coverage across days."
        )
    )
    parser.add_argument(
        "folder",
        help="Folder containing day_*.tar files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="coverage_matrix.png",
        help="Output image file (default: coverage_matrix.png)",
    )
    parser.add_argument(
        "--resolution",
        choices=["second", "minute"],
        default="second",
        help=(
            "Time resolution of the matrix: "
            "'second' = one column per second of day (default), "
            "'minute' = one column per minute of day."
        ),
    )
    parser.add_argument(
        "--view",
        choices=["matrix", "stacked"],
        default="matrix",
        help=(
            "Visualization type: 'matrix' (days x time) or "
            "'stacked' histogram (minute resolution only). "
            "Default: matrix."
        ),
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=60,
        help=(
            "For minute resolution: number of seconds required to mark "
            "a minute as covered (default: 60)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information while processing",
    )
    return parser.parse_args()


def find_tar_files(folder):
    tar_files = []
    for name in os.listdir(folder):
        if name.endswith(".tar") and name.startswith("day_"):
            full = os.path.join(folder, name)
            tar_files.append(full)

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


def collect_per_day_coverage(tar_paths, verbose=False):
    """
    Collect coverage information per day.

    Returns:
        coverage_per_day: OrderedDict[
            day_label -> {
                "seconds": set of second indices (0..86399),
                "minute_counts": dict[minute_idx -> count of seconds],
                "first_dt": first datetime seen in that tar,
            }
        ]
        day_dates: dict[day_label -> first datetime]
    """
    coverage_per_day = OrderedDict()
    day_dates = {}

    for tar_idx, tar_path in enumerate(tar_paths):
        day_label = os.path.splitext(os.path.basename(tar_path))[0]  # e.g., "day_1"
        if verbose:
            print(f"Processing {day_label}: {tar_path}")

        seconds_set = set()
        minute_counts = defaultdict(int)
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

                second_idx = dt.hour * 3600 + dt.minute * 60 + dt.second  # 0..86399
                seconds_set.add(second_idx)

                minute_idx = dt.hour * 60 + dt.minute  # 0..1439
                minute_counts[minute_idx] += 1

        if first_dt is not None:
            day_dates[day_label] = first_dt

        coverage_per_day[day_label] = {
            "seconds": seconds_set,
            "minute_counts": minute_counts,
            "first_dt": first_dt,
        }

        if verbose:
            num_seconds = len(seconds_set)
            num_minutes = len(minute_counts)
            print(
                f"  Found {num_seconds} frames in {num_minutes} distinct minutes "
                f"(first timestamp: {first_dt})"
            )

    return coverage_per_day, day_dates


def build_second_matrix(coverage_per_day):
    """
    Build a matrix with one row per day and one column per second of day.

    Returns:
        day_labels: list of day labels
        matrix: np.array shape (n_days, 86400), dtype=bool
    """
    day_labels = list(coverage_per_day.keys())
    n_days = len(day_labels)
    n_seconds = 24 * 60 * 60  # 86400

    matrix = np.zeros((n_days, n_seconds), dtype=bool)

    for day_idx, day_label in enumerate(day_labels):
        seconds_set = coverage_per_day[day_label]["seconds"]
        for s in seconds_set:
            if 0 <= s < n_seconds:
                matrix[day_idx, s] = True

    return day_labels, matrix


def build_minute_matrix(coverage_per_day, min_coverage):
    """
    Build a matrix with one row per day and one column per minute of day.

    Returns:
        day_labels: list of day labels
        matrix: np.array shape (n_days, 1440), dtype=bool
    """
    day_labels = list(coverage_per_day.keys())
    n_days = len(day_labels)
    n_minutes = 24 * 60  # 1440

    matrix = np.zeros((n_days, n_minutes), dtype=bool)

    for day_idx, day_label in enumerate(day_labels):
        minute_counts = coverage_per_day[day_label]["minute_counts"]
        for minute_idx, count in minute_counts.items():
            if 0 <= minute_idx < n_minutes and count >= min_coverage:
                matrix[day_idx, minute_idx] = True

    return day_labels, matrix


def make_sequential_cmap():
    """
    Sequential colormap from #e5f5f9 to #2ca25f.
    """
    return LinearSegmentedColormap.from_list(
        "seq_cmap", ["#e5f5f9", "#2ca25f"]
    )


def plot_coverage_matrix(day_labels, matrix, resolution, output_path):
    """
    Matrix view:
      x-axis: time-of-day bins (seconds or minutes)
      y-axis: days (one row per tar file)
      color: frame present (1) vs no frame (0)
    """
    n_days, n_bins = matrix.shape
    cmap = make_sequential_cmap()

    # Choose figure height based on number of days
    fig_height = max(4, 0.4 * n_days)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # X-axis ticks in time-of-day
    if resolution == "second":
        total_seconds = n_bins
        hours = range(0, 24)
        tick_positions = [h * 3600 for h in hours if h * 3600 < total_seconds]
        tick_labels = [f"{h:02d}:00" for h in hours if h * 3600 < total_seconds]
    else:  # "minute"
        total_minutes = n_bins
        hours = range(0, 24)
        tick_positions = [h * 60 for h in hours if h * 60 < total_minutes]
        tick_labels = [f"{h:02d}:00" for h in hours if h * 60 < total_minutes]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Y-axis: one tick per day
    ax.set_yticks(np.arange(len(day_labels)))
    ax.set_yticklabels(day_labels)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Day (tar file)")

    title_res = "per frame (1s)" if resolution == "second" else "per minute"
    ax.set_title(f"Time-of-day coverage matrix ({title_res})")

    # cbar = fig.colorbar(im, ax=ax, pad=0.02)
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(["no frame", "frame present"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_day_colormap(n_days):
    """
    Discrete colormap for days from #e5f5f9 to #2ca25f.
    Used only for stacked view.
    """
    base_cmap = LinearSegmentedColormap.from_list(
        "day_cmap", ["#e5f5f9", "#2ca25f"], N=n_days
    )
    return [base_cmap(i) for i in range(n_days)]


def plot_stacked_coverage(day_labels, matrix, output_path):
    """
    Stacked bar chart:
      x-axis: minute-of-day bins (0..1439)
      y-axis: number of days covering that minute
      each day is a colored layer in the stack.
    """
    n_days, n_minutes = matrix.shape
    minute_indices = np.arange(n_minutes)
    colors = make_day_colormap(n_days)

    x = minute_indices
    width = 1.0

    bottoms = np.zeros(n_minutes, dtype=int)

    fig, ax = plt.subplots(figsize=(16, 6))

    for day_idx, day_label in enumerate(day_labels):
        heights = matrix[day_idx].astype(int)
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

    hours = range(0, 24)
    tick_minutes = [h * 60 for h in hours]
    tick_labels = [f"{h:02d}:00" for h in hours]
    ax.set_xticks(tick_minutes)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Time of day (minute bins)")
    ax.set_ylabel("Number of days with full coverage for that minute")
    ax.set_title("Time-of-day coverage across days (stacked minute histogram)")
    # ax.legend(loc="upper right", ncol=2, fontsize="small")
    ax.get_legend().remove()

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

    coverage_per_day, day_dates = collect_per_day_coverage(
        tar_files, verbose=args.verbose
    )

    if args.verbose:
        print("\nPer-day first timestamps (for reference):")
        for day_label, dt in day_dates.items():
            print(f"  {day_label}: {dt}")

    if args.resolution == "second":
        if args.view == "stacked":
            raise SystemExit(
                "Stacked view requires minute resolution. "
                "Use --resolution minute --view stacked."
            )
        day_labels, matrix = build_second_matrix(coverage_per_day)
    else:  # minute resolution
        day_labels, matrix = build_minute_matrix(
            coverage_per_day, min_coverage=args.min_coverage
        )

    if args.verbose:
        print(
            f"\nBuilt matrix with shape {matrix.shape} "
            f"(resolution={args.resolution}, view={args.view})"
        )

    if args.view == "matrix":
        plot_coverage_matrix(day_labels, matrix, args.resolution, args.output)
    else:
        plot_stacked_coverage(day_labels, matrix, args.output)

    if args.verbose:
        print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
