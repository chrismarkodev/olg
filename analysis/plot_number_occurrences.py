from collections import Counter
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "data_all_l649.csv"
    output_path = project_root / "results" / "number_occurrences_histogram.png"

    counts = Counter()
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in ("d1", "d2", "d3", "d4", "d5", "d6", "bonus"):
                counts[int(row[column])] += 1

    values = [counts.get(number, 0) for number in range(1, 50)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, 50), values, color="steelblue", edgecolor="black")
    ax.set_xticks(range(1, 50, 2))
    ax.set_xlim(0.5, 49.5)
    ax.set_xlabel("Ball number")
    ax.set_ylabel("Occurrences")
    ax.set_title("Ontario Lotto 6/49 number occurrences (1-49)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved histogram to {output_path}")
    print("Top 10 numbers:")
    for number, count in counts.most_common(10):
        print(f"{number}: {count}")


if __name__ == "__main__":
    main()
