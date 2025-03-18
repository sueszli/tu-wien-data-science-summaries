> Gao, Yunfan, et al. "Retrieval-augmented generation for large language models: A survey." [arXiv preprint arXiv:2312.10997](https://arxiv.org/pdf/2312.10997) (2023)

survey based on 100 rag studies

# definition

*large language models llm*

- very large models doing next-word-prediction using transformers
- we fine tune llms through queries
- weaknesses: hallucination, outdated knowledge, non-transparent and untraceable reasoning process

*retrieval augmented generation rag*

- llms retrieve relevant data when necessary from external knowledge base
- 3 research paradigms: naive, advanced, modular rag
- llm augmentation stages: pre-training, fine-tuning, inference

*fine tuning vs. rag*

- fine-tuning can be understood as a type of rag
- fine tuning = further training of model
- rag = adding external knowledge, prompt engineering

*paradigm: naive rag*

- earlierst paradigm
- retrieve-read pattern
- steps:
	- i. indexing = documents are converted to plaintext, chunked, encoded, stored in vector database
	- ii. retrieval = we retrieve top-k most relevant documents based on user query and their cosine-similarity
	- iii. generation = llm processes user query + retrieved documents + conversation history
- weaknesses: retrieval has precision-recall-tradeoff, model still halucinates, model often doesn't add any value to retrieved docs and just echoes them

*paradigm: advanced rag*

- rewrite-retrieve-rerank-read pattern
- improves retrieval:
	- i. pre-retrieval:
		- optimize query for indexing, adding metadata, …
		- query-rewriting, query-routing, query-transformation, query-expansion, …
	- ii. retrieval: same as before
	- iii. post-retrieval:
		- optimize query for llm, integrate retrieved data into query, expand query, …
		- reranking chunks, compressing context, summarizing, fusion, …
		- putting most relevant data on edges, emphasizing important information, reducing context size, …

*paradigm: modular rag*

- like a hybrid model
- search module, fine-tuning retriever, different rag pipelines, …
- new modules:
	- search module: allows user to select data sources either through prompts or query languages
	- memory module: iterative self-enhancement, learning from user
	- predict module: reduce noise and redundancy in context
	- task-adapter module: automated prompt-retrieval
- new patterns:
	- feedback loop, fine-tuning, integrating reinforcement learning, …
	- generate-read pattern
	- dsp rag = demonstrate-search-predict pattern
	- iter-retgen rag = advanced rag but with iterative retrieval process

# improvements

*improving retrieval*

how documents are retrieved from the data source.

- **retrieval source**:
	- data structure:
		- structured - knowledge-graph, …
		- semi-structured - pdf, …
		- unstructured - wiki dump, cross lingual text, domain specific data, …
		- llm generated content - in case no external knowledge required
	- retrieval granularity:
		- coarse vs. fine grained - retrieval of tokens, passages, …, chunks, documents
- **preprocessing of retrieval**:
	- indexing optimization:
		- chunking strategy - in how many parts the document is chunked
		- metadata attachments - which metadata is added to chunks (ie. generated summaries, descriptions)
		- structural index - hierarchical file structure, knowledge graph structure, …
	- query optimization:
		- multi-query - prompt engineering for query expansion, parallel execution
		- sub-quert - split task up in sub-tasks
		- chain-of-verification cove - validate expanded queries to reduce hallucination 
		- query rewrite - optimize / transform / generate query
		- query routing - routing to different retrieval pipelines based on query needs (query semantics and metadata)
- **selection of embedding model**:
	- mix/hyprid retrieval - combining sparse and dense rerieval
	- finde-tuning embedding model - for domain specific tasks
- **external adapter**:
	- adapting the retrieved content into a format that llms can work with well

*improving generation*

adjusting the retrieved content, adjusting the LLM

- **content curation**:
	- reranking - put most important results first
	- content selection / compression - compress to a form that's interpretable by llms, reduce noise
- **llm fine tuning**:
	- reinforcement learning, manually annotating answers and learning from them 
	- knowledge distillation

*augmentation process in rag*

multi-step reasoning through multi-step retrieval

full loop: query → retrieve → generate → judge → response

- **iterative**:
	- multiple iterations between judge and retrieve-step
	- alternating between retrieval and generation
- **recursive**:
	- same as iterative but on each iteration the query-tasks get split up in sub-tasks
	- uses chain-of-thought cot
	- can also multi-hop through knowledge-graph
- **adaptive**:
	- same as recursive but there is another judgement step before retrieval that lets system decide by itself whether external knowledge retrieval is necessary or not and when to break out the loop

*rag evaluation*

- measuring retrieval and generation quality
- quality scores - context relevance, answer faithfulness, answer relevance
- abilities - noise robustness, negative rejection, information integration, counterfactual robustness, context relevance

*outlook*

- use cases: dialogue, summarization, question answering, fact verification
- llms don't seem to be limited by context size anymore with 200k+ tokens
- noise and misinformation greatly reduce output quality - retrieval output should also be validated by an additional model
- rag and fine-tuning are blending into eachother
- rag isn't quite production ready yet - it has low recall, misinforms, leaks confidential data
- multimodal rags that can deal with any kind of data and any domain
