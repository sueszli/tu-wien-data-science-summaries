> Xian, Jasper, et al. "Vector search with OpenAI embeddings: Lucene is all you need." Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 2024.
> 
> critique: https://news.ycombinator.com/item?id=37373635

**search problem in vector space:** given the query embedding, the system’s task is to rapidly retrieve the top-k passage embeddings with the largest dot products.

**assumption:** top-k retrieval on sparse vectors and dense vectors / embeddings require seperate and dedicated vector stores for operations around HNSW (for k-nearest neighbor search in vector space) for generative AI applications.

- there have been many recent vector stores (pinecone, weaviate, chroma, milvus, qdrant,…).
- these systems are very convenient because additionally to crud operations, they also handle nearest neighbor search.

**observation:** this is wrong. state of the art vector search using generative AI does not require any AI-specific implementations. providing operations around HNSW indexes does not require a separate vector store.

- we should build upon existing infrastructure. companies have already invested a lot of money into the lucene ecosystem (elasticsearch, opensearch, and solr) for sparse retrieval models.
- lucene already has built in HNSW. it has the same feature set, but it’s just less performant and less convenient.
- embeddings can be computed with simple API calls (encoding as a service).
- indexing and searching dense vectors is conceptually identical to indexing and searching text with bag-of-words models that have been available for decades.

**but there’s a catch:** the more mature software hasn’t quite caught up yet.

- it’s janky: lucene doesn’t officially support it yet. it was a hack.
- it’s slow: lucene achieves only around half the query throughput of faiss under comparable settings.

---

also interesting: postgres support

- https://www.crunchydata.com/blog/hnsw-indexes-with-postgres-and-pgvector
- https://neon.tech/blog/pg-embedding-extension-for-vector-search
- https://github.com/neondatabase/pg_embedding
