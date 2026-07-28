import math
from datetime import datetime
from pathlib import Path
import csv


def calculate_weight(days_ago: int) -> float:
    if days_ago <= 1:
        return 0.0
    return 1 / math.log(days_ago)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "data_all_l649.csv"
    output_path = project_root / "results" / "cold_numbers_with_history.txt"

    draws = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            draw_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            draw_numbers = [int(row[col]) for col in ("d1", "d2", "d3", "d4", "d5", "d6", "bonus")]
            draws.append((draw_date, draw_numbers))

    reference_date = datetime.now().date()
    weighted_history = {number: 0.0 for number in range(1, 50)}
    seen = {number: False for number in range(1, 50)}

    for draw_date, draw_numbers in draws:
        days_ago = (reference_date - draw_date).days
        for number in draw_numbers:
            weight = calculate_weight(days_ago)
            weighted_history[number] += weight
            seen[number] = True

    trace_target = min(weighted_history.items(), key=lambda item: item[1])[0]
    trace_contributions = []

    for draw_date, draw_numbers in draws:
        days_ago = (reference_date - draw_date).days
        for number in draw_numbers:
            if number == trace_target:
                weight = calculate_weight(days_ago)
                trace_contributions.append((draw_date, days_ago, weight))

    ranked = sorted(
        ((value, number) for number, value in weighted_history.items() if seen[number]),
        reverse=True,
    )

    lines = ["Cold numbers (weighted history):"]
    for value, number in ranked:
        lines.append(f"{number}: {value:.4f} weighted history")

    trace_lines = [f"Trace for number {trace_target} (lowest weighted history):"]
    for draw_date, days_ago, weight in trace_contributions:
        trace_lines.append(f"draw {draw_date}: days ago={days_ago}, weight={weight:.6f}")
    trace_lines.append(f"total for number {trace_target}: {weighted_history[trace_target]:.6f}")
    lines.extend(trace_lines)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved cold number report to {output_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
