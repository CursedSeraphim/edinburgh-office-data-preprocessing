#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import tarfile

from PIL import Image  # pip install pillow

FILENAME_RE = re.compile(
    r"inspacecam163_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.jpg$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build trajectory CSV from Edinburgh office monitoring tar files.\n"
            "Uses frames of inspacecam163, optionally filtered by time window, "
            "converts them to downsampled grayscale or color and writes one row per frame."
        )
    )
    parser.add_argument(
        "folder",
        help="Folder containing day_*.tar files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="trajectories.csv",
        help="Output CSV file (default: trajectories.csv)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="Downsampled image width in pixels (default: 32)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=18,
        help="Downsampled image height in pixels (default: 18)",
    )
    mono_group = parser.add_mutually_exclusive_group()
    mono_group.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="Convert frames to grayscale (default)",
    )
    mono_group.add_argument(
        "--color",
        dest="grayscale",
        action="store_false",
        help="Keep frames in RGB color",
    )
    parser.set_defaults(grayscale=True)
    parser.add_argument(
        "--start-time",
        type=str,
        default="15:00:00",
        help=(
            "Start time (HH:MM or HH:MM:SS) of window to extract. "
            "Default: 15:00:00. Ignored if --use-all-times is set."
        ),
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default="15:06:59",
        help=(
            "End time (HH:MM or HH:MM:SS) of window to extract (inclusive). "
            "Default: 15:06:59. Ignored if --use-all-times is set."
        ),
    )
    parser.add_argument(
        "--use-all-times",
        action="store_true",
        help="Use all available frames in each tar file, ignoring time window.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "Use every N-th frame in time order (subsampling). "
            "1 = use all frames (default), 10 = one frame every 10 frames, etc."
        ),
    )
    return parser.parse_args()


def list_tar_files(folder):
    pattern = os.path.join(folder, "day_*.tar")
    tar_paths = sorted(glob.glob(pattern))
    if not tar_paths:
        raise SystemExit(f"No tar files matching 'day_*.tar' found in {folder}")
    return tar_paths


def parse_time_from_filename(name):
    """
    Returns (hour, minute, second) or None if the filename does not match.
    """
    base = os.path.basename(name)
    m = FILENAME_RE.match(base)
    if not m:
        return None
    h = int(m.group(4))
    m_ = int(m.group(5))
    s = int(m.group(6))
    return h, m_, s


def seconds_since_midnight(h, m, s):
    return h * 3600 + m * 60 + s


def parse_time_string(s):
    parts = s.split(":")
    if len(parts) == 2:
        h, m = parts
        s_val = 0
    elif len(parts) == 3:
        h, m, s_val = parts
    else:
        raise ValueError(f"Invalid time format: {s!r}. Expected HH:MM or HH:MM:SS")
    h_i = int(h)
    m_i = int(m)
    s_i = int(s_val)
    if not (0 <= h_i < 24 and 0 <= m_i < 60 and 0 <= s_i < 60):
        raise ValueError(f"Time out of range: {s!r}")
    return seconds_since_midnight(h_i, m_i, s_i)


def time_filter(day_sec, start_sec, end_sec, use_all_times):
    if use_all_times:
        return True
    return start_sec <= day_sec <= end_sec


def first_pass_lengths(tar_paths, start_sec, end_sec, use_all_times, frame_step):
    """
    First pass: determine, for each tar, how many *kept* frames (after subsampling)
    fall into the specified time window (or all times if use_all_times is True).
    Returns:
      - line_index_by_path: {tar_path -> line_id}
      - line_length_by_path: {tar_path -> number_of_kept_frames}
      - global_max_step: maximum step index across all lines
    """
    line_index_by_path = {}
    line_length_by_path = {}
    line_lengths = []

    line_counter = 0

    for path in tar_paths:
        with tarfile.open(path, "r") as tar:
            times = []
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                t = parse_time_from_filename(member.name)
                if t is None:
                    continue
                h, m, s = t
                day_sec = seconds_since_midnight(h, m, s)
                if time_filter(day_sec, start_sec, end_sec, use_all_times):
                    times.append(day_sec)

        times.sort()
        if times:
            # Number of frames after subsampling (0, step, 2*step, ...)
            kept_len = (len(times) + frame_step - 1) // frame_step
            line_index_by_path[path] = line_counter
            line_length_by_path[path] = kept_len
            line_lengths.append(kept_len)
            line_counter += 1

    if not line_lengths:
        raise SystemExit("No frames found in the specified time window.")

    max_len = max(line_lengths)
    global_max_step = max_len - 1 if max_len > 1 else 0

    return line_index_by_path, line_length_by_path, global_max_step


def build_header(width, height, grayscale):
    header = [
        "id",
        "line",
        "label",
        "step",
        "daytimestamp",
        "daytime",
        "action",
        "age",
        "age_global",
    ]
    # Pixel columns; for grayscale: <x>x<y>
    # For color: <x>x<y>_r, <x>x<y>_g, <x>x<y>_b
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if grayscale:
                header.append(f"{x}x{y}")
            else:
                header.append(f"{x}x{y}_r")
                header.append(f"{x}x{y}_g")
                header.append(f"{x}x{y}_b")
    return header


def extract_pixels(img_fileobj, width, height, grayscale):
    """
    Open image, convert to target mode and size, and return a flat list of pixel values.
    Grayscale: one value per pixel.
    Color: three values (R, G, B) per pixel.
    """
    img = Image.open(img_fileobj)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize((width, height), resample=Image.BILINEAR)

    if grayscale:
        pixels = img.load()
        values = []
        # x = 0..width-1, y = 0..height-1 -> 1x1..widthxheight
        for x in range(width):
            for y in range(height):
                values.append(pixels[x, y])
        return values
    else:
        pixels = img.load()
        values = []
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                values.extend([r, g, b])
        return values


def second_pass_write_csv(
    tar_paths,
    line_index_by_path,
    line_length_by_path,
    global_max_step,
    start_sec,
    end_sec,
    use_all_times,
    frame_step,
    width,
    height,
    grayscale,
    output_path,
):
    header = build_header(width, height, grayscale)
    global_id = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for path in tar_paths:
            if path not in line_index_by_path:
                continue  # no frames in window

            line_id = line_index_by_path[path]
            line_len = line_length_by_path[path]

            # Collect frames (time, daytimestamp, daytime, member)
            frames = []
            with tarfile.open(path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    t = parse_time_from_filename(member.name)
                    if t is None:
                        continue
                    h, m, s = t
                    day_sec = seconds_since_midnight(h, m, s)
                    if not time_filter(day_sec, start_sec, end_sec, use_all_times):
                        continue
                    time_str = f"{h:02d}:{m:02d}:{s:02d}"
                    frames.append((day_sec, time_str, member))

                frames.sort(key=lambda x: x[0])

                # Apply subsampling: keep every N-th frame in time order
                frames = frames[::frame_step]

                for step, (day_sec, time_str, member) in enumerate(frames):
                    if line_len > 1:
                        age = step / (line_len - 1)
                    else:
                        age = 0.0

                    if global_max_step > 0:
                        age_global = step / global_max_step
                    else:
                        age_global = 0.0

                    img_fileobj = tar.extractfile(member)
                    if img_fileobj is None:
                        continue

                    pixel_values = extract_pixels(
                        img_fileobj, width=width, height=height, grayscale=grayscale
                    )

                    row = [
                        global_id,          # id
                        line_id,            # line (trajectory/day id)
                        "",                 # label (blank)
                        step,               # step
                        day_sec,            # daytimestamp (seconds since midnight)
                        time_str,           # daytime (HH:MM:SS)
                        "",                 # action (blank)
                        f"{age:.6f}",       # age
                        f"{age_global:.6f}" # age_global
                    ] + pixel_values

                    writer.writerow(row)
                    global_id += 1


def main():
    args = parse_args()
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be >= 1")

    tar_paths = list_tar_files(args.folder)

    if args.use_all_times:
        start_sec = 0
        end_sec = 24 * 3600 - 1
    else:
        start_sec = parse_time_string(args.start_time)
        end_sec = parse_time_string(args.end_time)
        if end_sec < start_sec:
            raise SystemExit("end-time must be >= start-time")

    (
        line_index_by_path,
        line_length_by_path,
        global_max_step,
    ) = first_pass_lengths(
        tar_paths, start_sec, end_sec, args.use_all_times, args.frame_step
    )

    second_pass_write_csv(
        tar_paths,
        line_index_by_path,
        line_length_by_path,
        global_max_step,
        start_sec,
        end_sec,
        args.use_all_times,
        args.frame_step,
        args.width,
        args.height,
        args.grayscale,
        args.output,
    )


if __name__ == "__main__":
    main()
