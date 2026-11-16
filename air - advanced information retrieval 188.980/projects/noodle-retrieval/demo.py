from statistics import mean


def reciprocal_rank(ranking, relevant, k=10):
    return next((1 / rank for rank, item in enumerate(ranking[:k], 1)
                 if item in relevant), 0.0)


def mrr(cases, k=10):
    return mean(reciprocal_rank(ranking, relevant, k) for ranking, relevant in cases)


if __name__ == "__main__":
    before = [(["d2", "d1"], {"d1"}), (["d3", "d4"], {"d4"})]
    after = [(["d1", "d2"], {"d1"}), (["d4", "d3"], {"d4"})]
    assert mrr(before) == .5 and mrr(after) == 1
    assert reciprocal_rank(["a", "b"], {"b"}, k=1) == 0
    print(f"MRR@10: {mrr(before):.2f} to {mrr(after):.2f} on synthetic data")
