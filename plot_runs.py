import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


# Matches numeric name=value pairs, including scientific notation.
VALUE_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

EPOCH_PATTERN = re.compile(r"\bepoch\s+(\d+)\b")


def parse_log(log_path):
    """Parse epoch numbers and all numeric name=value pairs."""
    epochs = {}

    with open(log_path, "r") as file:
        for line in file:
            # Skip lines that don't contain an epoch number.
            epoch_match = EPOCH_PATTERN.search(line)

            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))

            # Extract all numeric name=value pairs from the line.
            # Using a generic pattern allows new metrics to be added without
            # modifying the parser.
            metrics = {
                name: float(value)
                for name, value in VALUE_PATTERN.findall(line)
            }

            epochs[epoch] = metrics

            # print(f"epoch {epoch}: {metrics}")

    return epochs


def plot_runs(runs, output=None):
    """Plot all metrics from all runs."""

    # Find all metrics that appear in at least one run.
    all_metrics = set()

    for epochs in runs.values():
        for metrics in epochs.values():
            all_metrics.update(metrics.keys())

    # Order the panels: train loss, learning rate, validation metrics, remaining metrics.
    ordered_metrics = []

    if "train_loss" in all_metrics:
        ordered_metrics.append("train_loss")

    if "lr" in all_metrics:
        ordered_metrics.append("lr")

    validation_metrics = sorted(
        metric for metric in all_metrics
        if metric.startswith("val_")
    )
    ordered_metrics.extend(validation_metrics)

    other_metrics = sorted(
        all_metrics - set(ordered_metrics) - {"lr"}
    )
    ordered_metrics.extend(other_metrics)

    if not ordered_metrics:
        print("No metrics found to plot.")
        return

    # Create a grid with two columns and enough rows for all metrics.
    ncols = 2
    nrows = (len(ordered_metrics) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    # Plot each metric in its own subplot.
    for ax, metric in zip(axes, ordered_metrics):
        for run_name, epochs in runs.items():
            x = []
            y = []

            for epoch in sorted(epochs):
                if metric in epochs[epoch]:
                    x.append(epoch)
                    y.append(epochs[epoch][metric])

            if x:
                ax.plot(x, y, label=run_name)

        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if the number of metrics is odd.
    for ax in axes[len(ordered_metrics):]:
        ax.set_visible(False)

    # Create one shared legend for all subplots.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(labels), 4),
        fontsize=8
    )

    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
    fig.subplots_adjust(top=0.90)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output}")
    else:
        plt.show()

    plt.close(fig)


def load_runs(paths):
    """Load logs from individual run directories or parent directories."""
    runs = {}
    loaded_dirs = set()

    for path in paths:
        path = Path(path)

        if not path.is_dir():
            print(f"Warning: Directory not found: {path}")
            continue

        # If this directory contains log.txt, treat it as a single run.
        if (path / "log.txt").is_file():
            run_dirs = [path]
        else:
            # Otherwise, search recursively for training runs.
            run_dirs = [
                log.parent
                for log in path.rglob("log.txt")
            ]

        for run_dir in sorted(run_dirs):
            # Avoid loading the same run more than once.
            run_dir = run_dir.resolve()

            if run_dir in loaded_dirs:
                continue

            loaded_dirs.add(run_dir)

            log_path = run_dir / "log.txt"
            epochs = parse_log(log_path)

            if not epochs:
                print(f"Warning: No epoch data found in {log_path}")
                continue

            # Use the path relative to the current working directory as label.
            # try:
            #     name = run_dir.relative_to(Path.cwd()).as_posix()
            # except ValueError:
            #     name = run_dir.as_posix()
            name = run_dir.name

            runs[name] = epochs
            print(f"Loaded {len(epochs)} epochs from {log_path}")

    return runs

def main():
    parser = argparse.ArgumentParser(
        description="Compare training metrics from multiple runs."
    )

    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more training run directories."
    )

    parser.add_argument(
        "--out",
        help="Save plot to a PNG file instead of displaying it."
    )

    args = parser.parse_args()

    runs = load_runs(args.run_dirs)

    if not runs:
        parser.error("No valid training logs found.")

    plot_runs(runs, args.out)


if __name__ == "__main__":
    main()

