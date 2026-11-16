# Evaluate a retrieval reranker with reciprocal rank


A companion to Advanced Information Retrieval (188.980, summer 2024). The original project combined relevance judgment aggregation, neural reranking and extractive question answering. Its reranking report used MRR@10, the mean reciprocal rank of the first relevant result within the first ten positions.

This example evaluates toy before-and-after rankings. It tests the metric only, not KNRM, TK or question answering. It includes no documents, labels from the original datasets, model files, or assessment materials.

    python3 demo.py
