> neo4j offers kg functionality right out the box: https://neo4j.com/docs/graph-data-science/current/introduction/

# introduction

*knowledge graphs*

simplified: a graph database with reasoning functionality.

- $KG = (D, K; g, l, r)$
- where:
	- $D$ = data
	- $K$ = knowledge
	- $g$ = graph function (get graph from data)
		- $g: \mathbb D \mapsto \mathbb G$
		- $g: \mathbb D \mapsto (\mathbb V \times \mathbb E)$
	- $l$ = learning function (get knowledge from data)
		- $l_{\mathbb K}: \mathbb D \mapsto \mathbb K$
	- $r$ = reasoning function (infer more data)
		- $r: \mathbb D \mapsto \mathbb D$

# embeddings and neural nets

*graph embeddings*

- $e: (\mathbb V \cup \mathbb E \mapsto \mathbb{R}^n)$
- $l: \mathbb G \mapsto \mathbb e = \mathbb K$
- embedding function maps graph to vectors
- latent knowledge representation
- we can infer more data by assigning likelyhoods to vetrices and edges

*trans-e algorithm*

- $\mathcal{L}=\sum_{(h,r,t)\in E}\sum_{(h^{\prime},r,t^{\prime})\in E^{\prime}}[\gamma+f(h,r,t)-f(h^{\prime},r,t)]_{+}$
- where:
	- $f(h, r, t) = ||h + r - t||_1$ = score function using L1 distance
	- $\gamma$ = constant
- https://gist.github.com/mommi84/07f7c044fa18aaaa7b5133230207d8d4
- get embeddings of objects and relations
- make sure that when adding relation vector to head vector you end at the tail vector: `head + relation ≈ tail`
- generate negative examples by using false node as head or tail (but not both at once)
- in the loss function we want the first part to be small, the second part to be large (which is why it's negative)
- weights are updated with stochastic gradient descent

*graph neural nets*

- $H^{t+1} = F_W(H^t, X)$
- where:
	- $H_v$ = hidden state of node, as embedding
	- $X_v$ = features of node
	- $X_{v,w}$ = features of edge
	- $F_W$ = propagation step, with weights
- https://arxiv.org/ftp/arxiv/papers/1812/1812.08434.pdf
- learns change in network
- inductive (opposite to transductive): can be applied to unseen nodes, knowledge is latent, we don't have experts

*message passing neural nets*

- $H_v^{t+1}=u(H_v^t,a_{u\in N(v)}(m(H_v^t,H_u^t,X_{u,v}))$
- where:
	- $a_{u\in N(v)}(m(H_v^t,H_u^t,X_{u,v}))$ = aggregation function $a$ applied on all neighbor messages $m$
	- $u(H_v^t,a(\ldots)$ = update of hidden state using previous state and message aggregation ie. by using ReLu
- https://snap.stanford.edu/graphsage/
- messages from neighbours get aggregated to update the current state to generate embeddings
- used to predict links to unseen entities, classify entities, generate new entitites

# reasoning

*logic*

- $\models:\mathbb{F}\mapsto\mathbb{F}$
- logic = logical consequence function between formula
- can be expressed in many languages (math notation, datalog, sql, relational calculus)

*reasoning*

- $r = \models$
- $l: \mathbb D \mapsto \mathbb F = \mathbb K$
- types:
	- recursive reasoning = full recursion over graphs
	- ontological reasoning = object creation
	- numerical reasoning = numeric computation and aggregation
	- probabilistic reasoning = uncertain information
	- subsymbolic reasoning = low-dimensional spaces
	- temporal reasoning = reasoning over time
	- scalable reasoning = coping with large datasets
- https://www.postgresql.org/docs/current/queries-with.html
- used to infer more data, based on expert knowledge
- graph exploration needs recursive queries (recursive resoning) / object creation with an existence quantifier (ontological reasoning) but both are undecidable in datalog and need special techniques like "warding" heuristics to make them deterministic

*tuple generating dependencies*

- $\forall x\left(\varphi(\vec{x})\to\exists\vec{y}\,\psi(\vec{x},\vec{y})\right)$
- inferring data
- where:
	- $\forall x$ = quantifier
	- $\varphi(\vec{x})$ = premise
	- $\psi(\vec{x},\vec{y})$ = conclusion

*equality generating dependencies*

- $\forall x\left(\varphi(\vec{x})\to x_i=x_j\right)$
- joining tables
- where:
	- $\forall x$ = quantifier
	- $\varphi(\vec{x})$ = premise
	- $x_i=x_j$ = equality
