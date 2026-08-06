import math
from datetime import date, datetime
from pathlib import Path
import csv


def calculate_weight(days_ago: int) -> float:
    if days_ago <= 1:
        return 0.0
    return math.log(days_ago)


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
    last_seen: dict[int, date | None] = {number: None for number in range(1, 50)}

    for draw_date, draw_numbers in draws:
        days_ago = (reference_date - draw_date).days
        weight = calculate_weight(days_ago)
        for number in draw_numbers:
            weighted_history[number] += weight
            if last_seen[number] is None or draw_date > last_seen[number]:
                last_seen[number] = draw_date

    cold_scores = {}
    for number in range(1, 50):
        last_seen_date = last_seen[number]
        if last_seen_date is not None:
            days_since_last = (reference_date - last_seen_date).days
            cold_scores[number] = weighted_history[number] + 5 * days_since_last

    trace_target = min(cold_scores.items(), key=lambda item: item[1])[0]
    trace_contributions = []

    for draw_date, draw_numbers in draws:
        days_ago = (reference_date - draw_date).days
        for number in draw_numbers:
            if number == trace_target:
                weight = calculate_weight(days_ago)
                trace_contributions.append((draw_date, days_ago, weight))

    ranked = sorted(
        ((cold_scores[number], number) for number in cold_scores),
        reverse=True,
    )

    lines = ["Cold numbers (weighted history + last-seen age):"]
    for score, number in ranked:
        last_seen_date = last_seen[number]
        if last_seen_date is None:
            continue
        days_since_last = (reference_date - last_seen_date).days
        lines.append(
            f"{number}: {score:.4f} cold score (history={weighted_history[number]:.4f}, last_seen={last_seen_date}, days_since_last={days_since_last})"
        )

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
