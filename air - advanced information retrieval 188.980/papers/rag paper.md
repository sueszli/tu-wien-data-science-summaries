> Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in Retrieval-Augmented Generation. In Proceedings of the 47th Int’l ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 6 pages. [arXiv:2404.13781](https://arxiv.org/pdf/2404.13781)
> 
> code: https://github.com/alirezasalemi7/eRAG

goal: evaluating the retrieval-system in rag pipelines

*rag*

- retrieval augmented generation
- benefits: decreasing hallucination, knowledge-grounding, personalization

*traditional end-to-end evaluation*

- comparing generated output with expected output
- types:
	- a) human annotators – human preference doesn't judge utility for llm in rag
	- b) binary relevance of output based on query (both by human and llm) – impractical for long text, classification
- weaknesses:
	- lack of transparency:
		- no doc-level annotations – we don't know which retrieved document contributed to generated output
		- no implicit feedback (ie. through clicking) to optimize ranking system
		- ranking isn't reproducible because retrieval system often merges multiple parallel rankings (interleaving)
	- expensive:
		- running llm on all docs at once needs a lot of resources
		- human annotators are expensive
	- inaccurate:
		- query-doc-labels show small correlation with rag system performance
		- human annotation / llm annotation relies on human preferences, not utility for llm in rag

*erag*

- = evaluating retrievers in rag systems
- each doc is individually processed by same llm used in rag system - then those outputs are compared with expected results and aggregated with set or ranking metrics
- benefits:
	- kenadall correlation $\tau$ = 0.168;0.494
	- 50x less memory
	- runtime goes from $O(k^2n^2)$ to $O(kn^2)$ for $k$ documents
- steps:
	- i. we retrieve $k$ documents $\mathbf R_k$ from ranking-system $\mathcal R$
	- ii. for the query $q$ and each document $d$ we let the llm $\mathcal M$ generate an answer $\bar y$ (= downstream output)
	- iii. we evaluate each answer to determine the usefulness of the document based on the expected output $y$
	- iv. we use the usefulness of all documents to the llm $\mathcal G_q$ as a metric to evaluate ranking-system $\mathcal R$
- notation:
	- $\forall d \in \mathbf R_k: \mathcal G_q[d] = \mathcal E_{\mathcal M}(\mathcal M(q, \{d\}), y)$
	- $\mathcal E_{\mathcal R}(\mathbf R_k, \mathcal G_q) \mapsto [0;1]$ as score for ranking system based on some metric
	- where:
		- $\mathcal G_q[d]$ = relevance score for doc, used to evaluate ranking system $\mathcal R$
		- $\mathbf R_k$ = $k$ retrieved documents, each $d$ individually processed
		- $q$ = query
		- $\mathcal M$ = llm
		- $\mathcal E_{.}$ = evaluator
		- $y$ = expected classification
		- $\bar y$ = actual classification

*experiment*

- test environment:
	- KILT benchmark
	- has a bunch of datasets with relevance-labels called "provenance"
	- 100 word passages
	- evaluation on whether documents are relevant to query with MISTRAL
	- T5-small for llm in rag

*findings*

- eRAG-results correlate with downstream-performance the strongest
	- this means that the rag-llm is the best judge for retriever performance
- increasing the number of retrieved documents decreases the eRAG correlation, but it still has the highest correlation overall
	- this is because the rag-llm processes all docs at once while all retrieval-systems evaluate docs individually
- some smaller llm-models have a higher correlation for some specific datasets
	- but it isn't a significant difference
- eRAG correlates strongest with fusion-in-decoder FiD llms
	- maybe because it also processes documents individually
- eRAG is more efficient than end-to-end evaluators
	- ~2500 times faster
	- 1200-2300x speedup
	- ~7-15x more memory-efficient in query-level config
	- ~40-50x more memory-efficient in doc-level config
