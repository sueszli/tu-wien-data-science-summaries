<!-- most of these are from mattermost -->

# 2024 (first exam)

**question**: The M in MAP stands for the mean over all Queries

answer (boolean): False

- in the context of Mean Average Precision (MAP), the "M" stands for the "Mean", which refers to the mean of the Average Precision (AP) values calculated over all queries
- it calculates the average precision across all queries, not the "mean over all queries"

---

**question**: MRR only considers the first relevant document

answer (boolean): True

- it takes the inverse of the position of the first relevant doc

---

**question**: BERT does not transform Tokens

answer (boolean): False

- bert = bidirectional encoder representations from transformers
- it generates contextual representations from tokens

---

**question**: CNNs can form nGrams trough the sliding window

answer (boolean): True

- 1D-CNNs can only be used in NLP with a sliding window
- use cases:
	- n-gram representation learning = generating word embeddings as char-n-grams
	- dimensionality reduction = capturing an embedding-n-gram

---

**question**: IR Models made \[…] totally obsolete

answer (fill in): 

- manual searching

---

*paper specific questions*

based on: https://arxiv.org/pdf/2405.07767

- the authors disclosed the exact promts they used
- The Paper follows the Cranfield paradigm
- the Authors used a specific TREC Setting to evaluate
- \[insert random collection from paper] is a Test collection
- This paper is the first to use LLMs in Test Collection creation
- how were the different methods correlated? what did the $\tau$ values mean? what metrics were used?
- ChatGPT’s Queries were on average significantly longer than those of T5
- ChatGPT’s relevant judgements address the information need with way less documents
- Further experiments showed BIAS in using LLMs for the generation of Test collections
- Synthetic Testcollections are used for documents that are never used by humans
- Synthetic Queries resulted on average on more relevant documents per Query

# 2023

**question**: Recall and nDCG are typically measured at a lower cutoff than MAP and MRR (you don't have to know the exact formula)

answer (boolean): False

- the MAP is a more general metric and captures the area under the precision-recall-curve → ~@100-1000
- but the MRR, DCG, nDCG measure how far-up in the search results the rel-docs are positioned → ~@5-20
- "… typically, we measure all those metrics at a certain cutoff at "k" of the top retrieved documents. And for MAP and recall this is typically done at 100 and at 1000, whereas for position MRR and nDCG we have a much lower cutoff, so at 5, at 10 or at 20 to kind of get the same experience as users would do." [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%202%20-%20Closed%20Captions.md#14-ranking-list-evaluation-metrics)

---

**question**: Judgement pairs should use pooling of many diverse system results

answer (boolean): True

- this question is referring to the pooling-process in labeling ie. with mechanical turk where you're creating a cutoff-set to reduce the labor required to label your data.
- "… if we use a diverse pool of different systems, we can then even reuse those pool candidates and this gives us confidence that we have at least some of the relevant results in those pooling results. It allows us to drastically reduce the annotation time compared to conducting millions of annotations by hand." [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%203%20-%20Closed%20Captions.md#26-pooling-in-information-retrieval)

# 2022

**question**: Test collections should be statistically significant

answer (boolean): False

- the systems/models we build with the test-collections should be when we compare them, but not the test-collections themselves.
- statistical significance tests are used to verify that the observed differences between systems/models are not due to chance.
- "… we test whether two systems produce different rankings that are not different just by chance \[…]. Our hypothesis is that those systems are the same and now we test via a statistical significance test on a per-query basis" [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%202%20-%20Closed%20Captions.md#29-statistical-significance-i)

---

**question**: The quality of a test collection is measured with the inter-annotator agreement

answer (boolean): False

- the degree of agreement among raters ≠ test-collection quality
- "We can measure the label quality of annotators based on their inter-annotation agreement" [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%203%20-%20Closed%20Captions.md#25-evaluate-annotation-quality)
- see: https://en.wikipedia.org/wiki/Inter-rater_reliability

---

**question**: A word-1-gram that we use when training Word2Vec is also considered as a word-n-gram

answer (boolean): False

- 1-gram ≠ n-gram
- word2vec generates a single embedding for each word by learning to either guess the word from its surroundings or the other way around.
- you do have to train it with more than a single word and pass in a window size, but the word it's being trained to reconstruct is always a 1-gram / unigram.
- don't confuse this with CNNs that generate n-gram representations (a single embedding for n words)
- "1-Word-1-Vector type of class which includes Word2Vec" [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%204%20-%20Closed%20Captions.md#10-word-embeddings)
- see: https://radimrehurek.com/gensim/models/word2vec.html#usage-examples

---

**question**: ColBERTer achieves state-of-the-art performance on the MS Marco dev set

answer (boolean): True

- "When trained on MS MARCO Passage Ranking, ColBERTv2 achieves the highest MRR@10 of any standalone retriever." [(source: paper)](https://arxiv.org/abs/2112.01488)

---

**question**: "Assign each retrieval model its advantage of desirable properties for a retrieval model compared to the other models."

models:

- bert-cat
- tk
- colbert
- bert-dot

attributes:

- effectivity
- memory hog
- effort moved to indexing
- transformers combined with kernel-pooling

answer (assign attributes to sentences):

| model name | effectivity | latency | memory footprint | note                                                                         |
| ---------- | ----------- | ------- | ---------------- | ---------------------------------------------------------------------------- |
| bert-cat   | 1           | 950ms   | 10.4GB           | vanilla-bert and t5 are the slowest and most accurate                        |
| preTTR     | 0.97        | 445ms   | 10.9GB           | precomputes $n$ layers of bert for each doc                                  |
| colBERT    | 0.97        | 28ms    | 3.4GB            | precomputes representations for each doc                                     |
| bert-dot   | 0.87        | 23ms    | 3.6GB            | uses cosine similarity instead of linear layer                               |
| tk         | 0.89        | 14ms    | 1.8GB            | limits number of transformer layers and context, then applies kernel pooling |

above is a table from the lecture slides with some notes, that the following answers are based on:

- bert-cat ← effectivity
	- vanilla-bert and T5 are the most effective models against which all others are benchmarked
- tk ← transformers combined with kernel-pooling
	- can have a bunch of other optimizations as tkl or tk-sparse
- colbert ← memory hog
	- uses most memory in practice (i don't know why this doesn't align with the table above), because we're storing entire contextualized representations for each document
- bert-dot ← effort moved to indexing
	- used in combination with a nearest-neighbor-index so we can just use the cosine similarity instead of a linear layer or max-pooling

---

*course papers from that year:*

- colberter was particularly popular: https://arxiv.org/pdf/2203.13088
- https://dl.acm.org/doi/pdf/10.1145/3269206.3271719
- https://discovery.ucl.ac.uk/id/eprint/10119400/1/Mitigating_the_Position_Bias_of_Transformer_Based_Contextualization_for_Passage_Re_Ranking.pdf
- https://ecai2020.eu/papers/82_paper.pdf

# 2019

**question**: What are the differences between Matchpyramid and KNRM?

answer (open question):

- both are neural re-ranking models
- interpretability:
	- kernel-based models can be less interpretable, as the learned ranking function operates in a high-dimensional feature space
	- hierarchical convolutional layers provide some interpretability, as the model learns to capture matching patterns at different levels of hierarchy
- efficiency:
	- both scale linearly to the document length but knrm is faster because it's simpler [(source: paper)](https://www.ijcai.org/proceedings/2019/0758.pdf)
	- "the KNRM model is very fast, so it's definitely by far the fastest model we're talking about today, and in this course as a whole. And on its own, it has roughly the same effectiveness as MatchPyramid." [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%207%20-%20Closed%20Captions.md)
- robustness against small vocabularies:
	- "MatchPyramid and KNRM suffer if they have small vocabularies \[… but.] if you then use FastText, you get better results overall, for all models except for KNRM were the results are quite on par." [(source: lectures)](https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/Lecture%207%20-%20Closed%20Captions.md#44-effect-of-the-fixed-vocabulary-size)
- architectures:
	- matchPyramid – hierarchical pattern extraction
		- i. compute the 2D match-matrix of all query-doc cosine-similarities
		- ii. apply a series of convolutional kernels and pooling layers on match-matrix
		- iii. compute final score with a neural net
	- knrm – kernel-based approach to count the amount of different similarities between query and doc
		- i. (if it's a convolutional-knrm) use CNNs to generate word-n-gram embeddings
		- ii. compute the 2D match-matrix of all query-doc cosine-similarities
		- iii. apply the radial-basis-function kernel, summed along document dimension
		- iv. compute final score with a neural net

---

**question**: What would the precision-recall-curve of an ideal re-ranker look like?

answer (open question):

- "often, there is an inverse relationship between precision and recall" [(source: wikipedia)](https://en.wikipedia.org/wiki/Precision_and_recall#:~:text=Often%2C%20there%20is%20an%20inverse,illustrative%20example%20of%20the%20tradeoff)
- improving recall (complreteness) typically comes at the cost of reduced precision (correctness), because you're likelier to make more mistakes as you retrieve more data.
- usually we see high precision at low recalls, gradually decreasing as recall increases. and after all relevant documents have been retrieved we have diminishing returns and a sharp drop in precision.
- so ideally we'd like to have perfect precision until all relevant documents have been retrieved and vertically drop to 0 precision.

---

**question**: Why are low-frequency words an issue for information retrieval but not so much for other tasks like information categorization?

answer (open question):

- information-retrieval = deciding relevance of docs to a query
	- in neural IR: "(1) The model performs poorly on previously unseen terms appearing at retrieval time (OOV terms). (2) Due to the lack of training data for low-frequency terms, the learned vectors may not be semantically robust." [(source: hofstätter's paper)](https://arxiv.org/pdf/1904.12683)
	- but traditional IR models use TF-IDF
- classification = deciding whether a word belongs to a set of predefined categories
	- we're capturing the general theme or topic of a document, so presence / absence of rare words may have less influence on the overall decision.

