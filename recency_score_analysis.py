import csv
import math
import statistics
from datetime import datetime, date
from pathlib import Path


def calculate_weight(days_ago: int) -> float:
    if days_ago <= 1:
        return 0.0
    return 1 / math.log(days_ago)


def load_draws(data_path: Path):
    draws = []
    with data_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            draw_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
            draw_numbers = [int(row[col]) for col in ('d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'bonus')]
            draws.append((draw_date, draw_numbers))
    return draws


def analyze_recency_vs_weight(draws, reference_date: date):
    weights = {number: 0.0 for number in range(1, 50)}
    last_seen = {number: None for number in range(1, 50)}
    counts = {number: 0 for number in range(1, 50)}

    for draw_date, draw_numbers in draws:
        days_ago = (reference_date - draw_date).days
        weight = calculate_weight(days_ago)
        for number in draw_numbers:
            counts[number] += 1
            weights[number] += weight
            if last_seen[number] is None or draw_date > last_seen[number]:
                last_seen[number] = draw_date

    items = [(number, weights[number], last_seen[number], counts[number]) for number in range(1, 50)]
    items.sort(key=lambda item: item[1], reverse=True)

    return items


def compute_inversion_stats(items, reference_date: date):
    older_first = sorted(items, key=lambda item: item[2])
    inversion_count = 0
    sample_pairs = []

    for i, (num_i, weight_i, last_i, count_i) in enumerate(older_first):
        for num_j, weight_j, last_j, count_j in older_first[i + 1:]:
            if weight_j > weight_i and last_j > last_i:
                inversion_count += 1
                if len(sample_pairs) < 20:
                    sample_pairs.append((num_i, num_j, last_i, last_j, weight_i, weight_j, count_i, count_j))

    ages = [(reference_date - last_seen).days for _, _, last_seen, _ in items]
    weights = [weight for _, weight, _, _ in items]
    mean_age = statistics.mean(ages)
    mean_weight = statistics.mean(weights)
    numerator = sum((age - mean_age) * (weight - mean_weight) for age, weight in zip(ages, weights))
    denominator = math.sqrt(sum((age - mean_age) ** 2 for age in ages) * sum((weight - mean_weight) ** 2 for weight in weights))
    correlation = numerator / denominator if denominator else float('nan')

    return {
        'inversion_count': inversion_count,
        'pair_total': len(items) * (len(items) - 1) // 2,
        'correlation_age_weight': correlation,
        'sample_pairs': sample_pairs,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / 'data' / 'data_all_l649.csv'
    reference_date = date.today()

    draws = load_draws(data_path)
    items = analyze_recency_vs_weight(draws, reference_date)
    stats = compute_inversion_stats(items, reference_date)

    print('Correlation age->weight:', stats['correlation_age_weight'])
    print('Inversion count (more recent higher weight):', stats['inversion_count'], 'out of', stats['pair_total'])
    print('\nTop 10 numbers by weight:')
    for number, weight, last_seen, count in items[:10]:
        print(number, weight, (reference_date - last_seen).days, last_seen, count)

    print('\nBottom 10 numbers by weight:')
    for number, weight, last_seen, count in items[-10:]:
        print(number, weight, (reference_date - last_seen).days, last_seen, count)

    print('\nSample inversions:')
    for sample in stats['sample_pairs']:
        print(sample)


if __name__ == '__main__':
    main()
