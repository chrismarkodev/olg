from pathlib import Path
import csv
from collections import defaultdict


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "data_all_l649.csv"
    output_path = project_root / "results" / "cold_numbers.txt"

    draws = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            draw_numbers = [int(row[col]) for col in ("d1", "d2", "d3", "d4", "d5", "d6", "bonus")]
            draws.append(draw_numbers)

    last_seen = {number: 0 for number in range(1, 50)}
    coldness = {number: 0 for number in range(1, 50)}
    seen = {number: False for number in range(1, 50)}

    for idx, draw in enumerate(draws):
        for number in draw:
            if seen[number]:
                coldness[number] = idx - last_seen[number]
            else:
                seen[number] = True
            last_seen[number] = idx

    final_index = len(draws) - 1
    for number in range(1, 50):
        if seen[number]:
            coldness[number] = final_index - last_seen[number]

    ranked = sorted(
        ((value, number) for number, value in coldness.items() if seen[number]),
        reverse=True,
    )

    lines = ["Cold numbers (most draws since last appearance):"]
    for value, number in ranked:
        lines.append(f"{number}: {value} draws since last appearance")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved cold number report to {output_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
