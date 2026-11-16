from functools import cache
from itertools import product


def min_nodes(rows):
    width = len(rows[0][0])

    @cache
    def solve(indices):
        if len({rows[i][1] for i in indices}) == 1:
            return 0
        best = float("inf")
        for feature in range(width):
            left = tuple(i for i in indices if not rows[i][0][feature])
            right = tuple(i for i in indices if rows[i][0][feature])
            if left and right:
                best = min(best, 1 + solve(left) + solve(right))
        return best

    answer = solve(tuple(range(len(rows))))
    return None if answer == float("inf") else answer


if __name__ == "__main__":
    xor = [((x, y), x ^ y) for x, y in product((0, 1), repeat=2)]
    assert min_nodes(xor) == 3
    assert min_nodes([((0,), 0), ((1,), 1)]) == 1
    assert min_nodes([((0,), 0), ((0,), 1)]) is None
    print("XOR: 3 internal nodes")
