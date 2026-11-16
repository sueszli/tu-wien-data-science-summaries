from math import log2
from statistics import mean


def ndcg(candidates, k=5):
    def dcg(labels):
        return sum(label / log2(i + 2) for i, label in enumerate(labels[:k]))

    labels = [label for _, score, label in sorted(candidates, key=lambda c: (-c[1], c[0]))]
    ideal = dcg(sorted(labels, reverse=True))
    return dcg(labels) / ideal if ideal else 0.0


if __name__ == "__main__":
    a = [("a", .8, 1), ("b", .4, 0), ("c", .1, 0)]
    b = [("d", .8, 0), ("e", .5, 1), ("f", .2, 0)]
    assert ndcg(a) == 1
    assert 0 < ndcg(b) < 1
    assert ndcg([("g", .8, 0)]) == 0
    print(f"Mean nDCG@5: {mean(map(ndcg, [a, b])):.3f}")
