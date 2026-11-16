from collections import Counter


def map_counts(rows, term, category):
    return Counter((term in terms, label == category) for label, terms in rows)


def chi_square(counts):
    a, b, c, d = (counts[key] for key in (
        (True, True), (True, False), (False, True), (False, False)
    ))
    denominator = (a + b) * (a + c) * (b + d) * (c + d)
    if not denominator:
        return 0.0
    return (a + b + c + d) * (a * d - b * c) ** 2 / denominator


if __name__ == "__main__":
    shards = (
        [("garden", ["soil", "soil"]), ("books", ["novel"])],
        [("garden", ["soil"]), ("books", ["essay"])],
    )
    counts = sum((map_counts(s, "soil", "garden") for s in shards), Counter())
    assert sum(counts.values()) == 4
    assert chi_square(counts) == 4
    assert chi_square(map_counts(shards[0], "absent", "garden")) == 0
    print("Pearson chi-square:", chi_square(counts))
