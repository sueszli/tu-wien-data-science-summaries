# Evaluate rankings within a news impression


A Recommender Systems group project (194.035, summer 2024) compared NRMS, LSTUR and GRU approaches for the [ACM RecSys Challenge 2024](https://recsys.acm.org/recsys24/challenge/).

The original models scored competing articles in the same impression. The example below illustrates that evaluation unit: it ranks candidates for two synthetic impressions and reports mean nDCG@5, with binary relevance. The scores are made up, not challenge results.

    python3 demo.py

The challenge dataset, vendor code, trained models, team submission, and reports are not included.
