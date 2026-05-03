from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

STATUSES = ("InUse", "Pending", "Free")
ALLOCATORS = (
    ("storage_buffer_stats", "Storage buffers"),
    ("readback_buffer_stats", "Readback buffers"),
)


def bytes_to_mib(value: float) -> float:
    return value / (1024 * 1024)


def persistent_in_use_baseline(
    snapshots: list[dict[str, Any]], key: str
) -> Counter[int]:
    """Return InUse capacities present in every non-empty snapshot.

    The MNIST example keeps dataloader tensors alive for the whole training run.
    They are real allocations, but they dominate the y-axis and hide allocator
    churn. Treating stable InUse buffers as a baseline makes the graph useful for
    seeing free/pending/cache behavior.
    """
    common: Counter[int] | None = None

    for snapshot in snapshots:
        buffers = snapshot.get(key, {}).get("buffers", [])
        if not buffers:
            continue

        counts = Counter(
            int(buffer.get("capacity", 0))
            for buffer in buffers
            if buffer.get("status") == "InUse"
        )

        common = counts if common is None else common & counts

    return common or Counter()


def summarize_allocator(
    snapshot: dict[str, Any],
    key: str,
    excluded_in_use: Counter[int] | None = None,
) -> dict[str, int]:
    totals = {status: 0 for status in STATUSES}
    buffers = snapshot.get(key, {}).get("buffers", [])
    excluded_in_use = Counter(excluded_in_use or {})

    for buffer in buffers:
        status = buffer.get("status")
        capacity = int(buffer.get("capacity", 0))

        if status == "InUse" and excluded_in_use[capacity] > 0:
            excluded_in_use[capacity] -= 1
            continue

        if status not in totals:
            totals[status] = 0

        totals[status] += capacity

    return totals


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        snapshots = json.load(file)

    if not isinstance(snapshots, list):
        raise ValueError(f"{path} must contain an array of RuntimeStats snapshots")

    return snapshots


def format_mib(value: int) -> str:
    return f"{bytes_to_mib(value):.2f} MiB"


def plot_allocations(
    input_path: Path,
    output_path: Path,
    show: bool,
    offset_persistent_in_use: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: python -m pip install matplotlib"
        ) from exc

    snapshots = load_snapshots(input_path)
    if not snapshots:
        raise ValueError(f"{input_path} does not contain any snapshots")

    x = list(range(len(snapshots)))
    x_labels = [
        "Start" if idx == 0 else "Data Loaded" if idx == 1 else str(idx - 1)
        for idx in x
    ]
    baselines = {
        key: persistent_in_use_baseline(snapshots, key)
        if offset_persistent_in_use
        else Counter()
        for key, _ in ALLOCATORS
    }
    colors = {
        "InUse": "#2563eb",
        "Pending": "#f59e0b",
        "Free": "#16a34a",
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    for axis, (key, title) in zip(axes, ALLOCATORS):
        series = {status: [] for status in STATUSES}

        for snapshot in snapshots:
            totals = summarize_allocator(snapshot, key)
            for status in STATUSES:
                series[status].append(bytes_to_mib(totals.get(status, 0)))

        axis.stackplot(
            x,
            [series[status] for status in STATUSES],
            labels=STATUSES,
            colors=[colors[status] for status in STATUSES],
            alpha=0.85,
        )

        total = [
            sum(values) for values in zip(*(series[status] for status in STATUSES))
        ]
        axis.plot(x, total, color="#111827", linewidth=1.5, label="Total")

        baseline_bytes = sum(capacity * count for capacity, count in baselines[key].items())
        if baseline_bytes:
            title = f"{title} (y-axis starts at {format_mib(baseline_bytes)} persistent InUse baseline)"

        axis.set_title(title)
        axis.set_ylabel("Capacity (MiB)")
        axis.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
        legend = axis.legend(loc="upper left", ncols=4, frameon=True)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#d1d5db")
        legend.get_frame().set_alpha(0.95)

        if baseline_bytes:
            axis.set_ylim(bottom=bytes_to_mib(baseline_bytes))

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(x_labels)

    axes[-1].set_xlabel("Stage / Epoch")
    title = "Torchic Buffer Allocation Over Time"
    if offset_persistent_in_use:
        title += " (Storage Axis Offset)"
    fig.suptitle(title, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)

    if show:
        plt.show()

    print(f"Wrote allocation graph to {output_path}")
    for key, title in ALLOCATORS:
        baseline_bytes = sum(capacity * count for capacity, count in baselines[key].items())
        if baseline_bytes:
            print(f"Offset {title.lower()} y-axis by baseline: {format_mib(baseline_bytes)}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Plot Torchic RuntimeStats buffer allocation snapshots."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=repo_root / "stats.json",
        help="Path to stats.json. Defaults to repo-root stats.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo_root / "scripts" / "buffer_allocation_graph.png",
        help="Output image path. Defaults to scripts/buffer_allocation_graph.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after writing the file.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw stats with y-axes starting at zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_allocations(args.input, args.output, args.show, not args.raw)


if __name__ == "__main__":
    main()
