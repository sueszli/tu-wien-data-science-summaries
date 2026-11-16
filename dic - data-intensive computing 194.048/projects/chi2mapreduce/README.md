# Score term-category association with chi-square


A companion to the MapReduce part of Data-intensive Computing (194.048). The original job emitted document-frequency counts by term and category and used the merged counts to rank terms by Pearson chi-square.

This example evaluates one contingency table from two synthetic shards. It counts a term at most once per document and merges cell counts before calculating the statistic. It does not require Hadoop and is not a copy of the original job or review dataset.

    python3 demo.py
