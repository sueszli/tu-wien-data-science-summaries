# Find a minimum decision tree on a tiny dataset


A companion example for Fixed-Parameter Algorithms and Complexity (192.135, winter 2024). The original project studied [Parameterized Complexity of Small Decision Tree Learning](https://doi.org/10.1609/aaai.v35i7.16800) by Ordyniak and Szeider.

Finding a small consistent tree is different from greedily selecting the best next split. This stand-alone example tries every Boolean split and memoizes subproblems to minimize the number of internal nodes. The XOR fixture needs three internal nodes. The example is an exhaustive baseline, not the paper's parameterized algorithm or the original course submission.

    python3 demo.py
