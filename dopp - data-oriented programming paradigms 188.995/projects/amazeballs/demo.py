from collections import defaultdict
from statistics import median


def summarize(rows):
    scores = defaultdict(list)
    for category, score in rows:
        if score is not None:
            scores[category].append(score)
    return [(key, len(values), median(values)) for key, values in sorted(scores.items())]


if __name__ == "__main__":
    rows = [
        ("books", -0.2), ("books", 0.6), ("books", 0.8),
        ("garden", -0.4), ("garden", 0.2), ("garden", None),
    ]
    result = summarize(rows)
    assert result == [("books", 3, .6), ("garden", 2, -.1)]
    for category, count, middle in result:
        print(f"{category}: n={count}, median={middle:.2f}")
