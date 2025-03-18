> Ma, Xueguang, Tommaso Teofili, and Jimmy Lin. "Anserini Gets Dense Retrieval: Integration of Lucene's HNSW Indexes." Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 2023.

anserini wraps lucene.

pyserini wraps lucene and faiss, so it provides both vector search and inverted indexes.

but it turns out that lucene (since a recent release) can (additionally to inverted indexes) also support hnsw indexes for dense vector search. this while being reasonably effective. but there’s a catch: due to weird design choices, to do hnsw indexes requires training models based on cosine similarity in lucene.

this paper:

1. demonstrates that lucene can be used for both dense and sparse vectors
2. compares lucene and faiss for dense vectors
