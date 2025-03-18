*ml*

ml = algorithms that fit data and make predictions

- supervised = labeled data (clustering, anomaly detection, classification, regression, …)
- unsupervised = unlabeled data (data-mining, …)
- reinforcement learning = reward function

*prediction types*

- regression (continuous)
- classification (discrete)
	- binary: 1/2
	- multi-class: 1/n
	- multi-label: m/n

*data types*

- qualitative / categorical data:
	- nominal = unique names
	- ordinal = also have order (sorting)
- quantitative / numerical data:
	- interval = also can be measured and compared on a scale (addition, subtraction)
	- ratio = also have an absolute zero point (multiplication, division)

*normalization*

- categorical:
	- 1-hot-encoding = map to 0 array with one flag bit
	- distance encoding = map ordinal-data to integers
- numerical:
	- min-max = map to 0;1
		- $z_i = (x_i - \min(X)) / (\max(X) - \min(X))$
	- z-score = map distribution to mean 0, std dev 1
		- $z_i = \frac{x_i - \mu}{\sigma}$
	- binning = map value ranges to discrete numbers

*missing values*

- a) deletion:
	- remove attribute
	- remove row (only in train-set)
- b) imputation:
	- categorical: `NA` as label, regression, clustering, knn
	- numerical: mean, median, regression, clustering, knn
- not allowed to influence results
- don't leak data from test to train-set when reconstructing values

*sampling*

- randomize data first
- stratification = make sure each class is equally represented in all sets
- data-leakage = data from train-set influencing test-set
- validation-set = subset of train-set to tune hyperparameters

holdout:

- 80/20 train and test split

k-fold cross val:

- split data in $k$ same-sized parts
- 1 part for test-set, remaining parts for train-set
- repeat $k$ times, make sure std. dev is low
- $k$ should be lower on datasets that are large (too expensive), and small (test-set too small)

leave-p-out cross val:

- choose unique subset of size $p$ (must differ from seen subsets by min. 1 elem)
- use this subset for test-set, remaining data for train-set
- repeat (too expensive!)

bootstrapping:

- assume that dataset has the same distribution as real data
- choose non-unique random subset
- use this subset (= bootstrap set) for test-set, remaining data (= out-of-bag set) for train-set

*feature extraction*

- get a representation of data (text, image, audio, …) that we can process
- representation learning

*feature selection*

- dimensionality reduction = select subset of features to reduce redundance, noise

comparison:

- supervised:
	- how well we can predict the target variable with the input features
- unsupervised:
	- dimensionality reduction
	- hide labels during unsupervised selection
	- might learn unwanted feature dependencies - run correlation analysis to prevent

types:

- supervised types:
	- a) wrapper
		- = evaluate models trained with feature subsets – search through all possible subsets.
		- forward selection – greedy search, starting with no features and progressively adding features
		- backward elimination – greedy search, starting with all features and progressively removing features
	- b) filter
		- = use statistical metrics directly on features – efficient, independent of the learning algorithm, less prone to overfitting.
		- information gain
		- chi-square test
		- mutual information
		- f-score
		- fisher's score
	- c) embedded
		- = evaluate features during training
		- lasso and ridge regulation
		- decision tree
- unsupervised types:
	- frequency count
	- principal component analysis – pca, project to lower dimension through single value decomposition
	- autoencoder – generating vector representations in hidden layer

*data augmentation*

- extend train-set by slightly modifying data (ie. flip, rotate, scale images, …)

# evaluation

*performance*

- efficiency = resource consumption during training or prediction
- effectivity = quality of prediction
	- we don't know how well the model really generalizes because we don't know the true mapping of the function we want to learn
	- error types:
		- irreducible - wrong framing of problem
		- bias - underfits, inaccurate, wrong assumptions about target function
		- variance - overfits, too noisy, doesn't generalize

*contingency table*

|                     | predicted positive PP       | predicted negative PN        |
| ------------------- | --------------------------- | ---------------------------- |
| actually positive P | true positive TP            | false negative FN (error II) |
| actually negative N | false positive FP (error I) | true negative TN             |

*confusion matrix*

- table of predicted vs. actual for all classes

*metrics for classification*

- accuracy
	- $\frac{TP + TN}{TP + FP + TN + FN}$
	- correctness of both positives and negatives
- precision
	- $\frac{TP}{TP + FP}$
	- correctness of positives
- specificity
	- $\frac{TN}{TN + FP}$
	- correctness of negatives
- recall, sensitivity
	- $\frac{TP}{TP + FN}$
	- completeness
- balanced accuracy
	- $\frac{\frac{TP}{TP + FN} + \frac{TN}{TN + FP}}{2}$
	- average of precision and specificity
- f1 score
	- $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$

*metrics for regression*

- $p_i$ = predicted
- $a_i$ = actual
- mean squared error MSE
	- ${\sum_i (p_i - a_i)^2} ~/~ n$
- root mean squared error RMSE
	- $\sqrt{\sum_i (p_i - a_i)^2 ~/~ n}$
- mean absolute error MAE
	- ${\sum_i |p_i - a_i|} ~/~ n$
- relative squared error RSE
	- ${\sum_i (p_i - a_i)^2} ~/~ {\sum_i (a_i - \bar{a})^2}$
- root relative squared error RRSE
	- $\sqrt{{\sum_i (p_i - a_i)^2} ~/~ {\sum_i (a_i - \bar{a})^2}}$
- relative absolute error RAE
	- ${\sum_i |p_i - a_i|} ~/~ {\sum_i |a_i - \bar{a}|}$
- statistical correlation coefficient:
	- ${S_{PA}} ~/~ {\sqrt{S_P \cdot S_A}}$
	- where:
		- $S_{PA} = {\sum_i(p_i-\overline{p})(a_i-\overline{a})} ~/~ {n-1}$
		- $S_P = {\sum_i(p_i-\overline{p})^2} ~/~ {n-1}$
		- $S_A = {\sum_i(a_i-\overline{a})^2} ~/~ {n-1}$

*macro averaging*

- $\frac{1}{|C|} \sum_{i=1}^{|C|} w_i \cdot \text{metric}$
- weighted average of metrics for each class
- weight based on cost of missclassification

*statistical significance testing*

- null hypothesis = difference between two systems isn't by chance (like variations in data or randomness in algorithm)
- level of significance $\alpha$ = probability of false negative – ie. 5% p-level means there is a 5% chance that the result is just by chance
- false positives in significance testing likelier if sample is too small
- mcnemars test:
	- not the same as chi-squared test
	- $\chi^2 = \frac{(N_{01} - N_{10}) ^ 2}{N_{01} + N_{10}}$
	- null-hypothesis example: we assume two models are identical $N_{01} = N_{10} = \frac {N_{01} + N_{10}} 2$
	- look up percentage points in chi-distribution (1 degree of freedom, your p-value) to see whether to reject
- paired t-tests:
	- for paired samples - ie. train-set vs. data-set in cross-validation, where $k$ are the degrees-of-freedom
	- $\begin{gathered}t=\frac{\bar{X}_{D}-{\mu}_{0}}{\frac{S_D}{\sqrt{n}}}\end{gathered}$
	- where:
		- $n$ = number of samples
		- $\bar X_D$ = difference of performance in each run (ie. accuracy)
		- $S_D$ = std dev of difference
		- $\mu_0$ = mean of difference (usually set to 0)

# knn - k nearest neighbour

*knn*

- lazy learner = no training, computation at prediction-step $O(Nd)$
- $d$ – feature dimensions
- $k$ – increase to reduce noise
- $w$ – weights can be inverse of neighbor rank or distance
- distance functions: euclidian, manhattan, linfinity, cosine, …

algorithm:

- $y \leftarrow$ mean / weighted majority of $k$ closest training-examples $x$

*k-d tree*

- data structure to improve knn efficiency
- only effective for low-dimensional data

algorithm:

- partition space evenly until there are just $n$ data-points per subspace

*k-means*

- clustering algorithm to reduce knn search space

algorithm:

- place centroids, assign each data-point to the closest centroid
- update centroid positions based on the average position of all assigned data points
- converges when centroids stop moving

# dt - decision tree

*decision tree*

- other versions: 1R (one rule), 0R (no rule, return majority)
- num of features are the space dimensions
- binary trees

algorithm:

- partition space with least label-uncertainty in each subspace - until there is just one label per leaf node

*measuring label-uncertainty*

- relative error rate = 1 - accuracy
- aboslute error rate = $FP + FN$
- information gain = $IG(X_1, \dots, X_m) = H(X) - \sum_{i=1}^m p(X_i) \cdot H(X_i)$
	- where:
		- $X_.$ = $m$ subspaces created through split, each containing different labels $C$
		- $0 \leq IG \leq H(X) = \log_2(|C|)$
		- $H(X) = E(I(X)) = \sum_{i = 1}^n p(x_i) \cdot I(x_i)  = - \sum_{i = 1}^n p(x_i) \cdot \log_{2}(p(x_i))$ = label entropy in subspace
		- entropy is redundancy as number of bits needed for lossless compression
- ratio = $R(X) = IG(X) / V(X)$
	- normalizes ig, penalizes too many subspaces
	- where:
		- $V(X) = \sum_{i=1}^m \frac{|T_i|}{|T|} \cdot \log(\frac{|T_i|}{|T|})$ = value
- gini impurity =  $I_G(p)=\sum_{i=1}^{|C|}p_i(1-p_i)= \dots =1-\sum_{i=1}^{|C|}p_i^2$
	- how often we would mislabel, if we would re-label everything based on current distribution (almost the same as information-gain)

*pruning tree*

- improve generalizability
- during training (prepruning):
	- based on threshold for: samples in leaf, tree depth, information gain
- during evaluation (pruning):
	- reduced error pruning = replace subtree with majority-vote class, keep reasonable accuracy
	- cost complexity pruning = generate competing trees, compare by: error divided by num of leafs
	- pessimistic error pruning = traverse tree, remove redundancy

*random forest*

- i. train multiple classifiers – through bootstrap sampling
- ii. combine classifiers – through majority-voting of each classifier
- iii. evaluate combined classifier – out-of-bag elements for each tree as test-set

*covering algorithm*

- alternative representation of decision-tree through logical expressions
- assumption: classes have non overlapping rules

algorithm (prism):

- for each class find an expression that covers all training-examples
	- for each expression find 'tests' (concatenated by $\wedge$) that have the least missclassifications

# nb - naive bayes

*statistical dependence vs. independence*

- independence:
	- $p(A \cap B) = p(A) \cdot p(B)$
	- $p(A\mid B)= \frac{p(A~\cap~ B)}{p(B)} = \normalsize p(A)$
- dependence (bayes theorem):
	- $p(A \cap B) = p(B \mid A) \cdot p(A)$
	- ${p}(A\mid B)= \frac{p(A ~\cap~ B)}{p(B)} = \frac{p(B \mid A) ~\cdot~ p(A)}{p(A) ~\cdot~ p(B \mid A) ~~+~~ p(\neg A) ~\cdot~ p(B \mid \neg  A)} = \frac{p(B \mid A) ~\cdot~ p(A)}{p(B)}$
	- $p(A\mid BCD)=\frac{p(BCD\mid A) ~\cdot~  p(A)}{p(BCD)}=\frac{p(B \mid A) ~\cdot~ p(C \mid A) ~\cdot~ p(D \mid A) ~\cdot~ p(A)}{p(BCD)}$
	- the proportion of all cases where $B$ is true, in which $A$ is also true

*naive bayes*

- leave out missing values
- assumption: statistical dependence of attributes, but not classes

algorithm classification:

- for each class, get probability of classification under given train-examples: $P(H \mid E_1 E_2 \dots E_n)$
- laplace estimator = add 1 or a weight $\mu$ to all fequencies, to avoid multiplying by 0

algorithm regression:

- assume normal distribution
- use probability-density-function of normal distribution $f(x)$ for each value

*bayes optimal classifier*

- same as naive-bayes but computes distribution of all classes together, not independently

algorithm:

- estimate probability distribution based on samples
- classify most likely output for each input

# bn - bayesian network

- https://ocw.mit.edu/courses/6-825-techniques-in-artificial-intelligence-sma-5504-fall-2002/pages/lecture-notes/
- https://cs.uwaterloo.ca/~a23gao/cs486686_f21/lecture_notes/Lecture_13_on_Variable_Elimination_Algorithm.pdf
- https://ermongroup.github.io/cs228-notes/

*bayesian network*

- directed acyclic graph
- assumption: statistical dependence of attributes, but not classes
- edges = conditional dependencies between attributes
- nodes = attributes
	- nodes have truth-tables (conditional-probability-tables cpt)
	- columns (inputs) are incoming edges (ancestors being true/false)
	- rows (outputs) are the probability of this node being true/false (given the priors)

algorithm:

- given our knowledge of which nodes are true/false (evidence $E$) get probability distribution for variables were interested in (query variables $Q$)
- $\text{argmax}_q ~~ {p}({Q}_1{=}{q}_1,{Q}_2{=}{q}_2, \ldots \mid {E}_1{=}{e}_1,{E}_2{=}{e}_2,\ldots)$

example:

- graph: $A \rightarrow B \rightarrow C \rightarrow D$
- query: $D=t$
- evidence: none
- $P(D=t)=\sum_a \sum_b \sum_c P(D=t|C)\cdot P(C|B) \cdot P(B|A) \cdot P(A)$
- sum over all combinations in all truth-tables - cache intermediary results, optimize execution order (variable-elimination-algorithm)

example:

- graph: $\{B, E\} \rightarrow A \rightarrow \{J, M\}$
- query: $B=t$
- evidence: $J=t$
- $P(B=t|J=t) = P(B=t,J=t) / P(J=t)$
	- $P(B=t,J=t) = \sum_a \sum_e \sum_m \sum_b P(J=t|A) \cdot P(M|A) \cdot P(A|B=t, E) \cdot P(B=t) \cdot P(E)$
	- $P(J=t) = \sum_a \sum_e \sum_m \sum_b   P(J=t|A) \cdot P(M|A) \cdot P(A|BE) \cdot P(B) \cdot P(E)$

*learning network structure*

- i. generate inital network
	- start with an initial network structure – this could be an empty network (no edges), a naive Bayes structure (all nodes connected to a single parent node), a randomly generated one, one based on prior knowledge, etc.
- ii. compute probabilities
	- count occurences in all truth tables
- iii. evaluate network
	- $D$ data, $M$ model
	- goodness-of-fit:
		- $p(D|M)=\Pi_jP(s^j|M)=\Pi_j\Pi_i p(N_i=v_i^j\mid Parents(N_i),M)$
	- log-probability:
		- $\log (p(D|M)) - \alpha =\sum_j\sum_i p(N_i=v_i^j \mid Parents(N_i),M)$
		- $\alpha$ - hyperparam to penalize network complexity
- iv. search for related networks, repeat
	- generate neighbors from the current network through local modifications like edge addition, deletion or reversal.

search algorithms:

- hill climbing = choose best in neighborhood, until local optimum is reached
- tabu search = hill-climbing but some some neighbors are hidden
- simulated annealing = based on cooldown parameters
- genetic algorithm = initialize population, mutate, reproduce, evaluatie population.

*connection types*

- transmission of evidence:
	- d-seperated / blocked = we can't transmit evidence (conditional independence)
	- d-connected / connected = we can transmit evidence
	- the rules below must apply to all possible paths between A, C
	- there must be at least one intermediate node
	- in the serial case we can differentiate between forward-serial and backward-serial
	- in the converging case, if B or any of its descendants are observed we can "explain away"
- connection types:
	- serial: A → B → C
		- A → C transmission blocked if we know B
	- diverging: A ← B → C
		- A ←→ C transmission blocked if we know B
	- converging: A → B ← C
		- A ←→ C transmission blocked if we don't know B or any of it's descendants

# svm - support vector machine

- https://en.wikipedia.org/wiki/Support_vector_machine

*kernel matrices*

- kernel trick = increase dimensionality of feature space, to make them linearly seperable by high-dimensional hyperplane
- $K\in\mathbb{R}^{n\times n}$
- polynomial kernel of degree $d$:
	- $K(x,y) = (x^\intercal y + c)^d$
- requirements:
	- kernel must be positive semi-definite (psd): $\forall \mathbf x\in\mathbb{R}^n:\mathbf x^tK\mathbf x\geq0$
	- kernel must be factorizable: $K = F^{t}F$
	- kernel must only have non-negative eigenvalues: $UDU^t=K,~U^tU=1$
- can be also applied to kernel perceptron

kernel examples:

- linear: $K(x, y) = x^\intercal y$
- polynomial: $K(x, y) = (x^\intercal y + c)^d$
- radial basis function rbg: $K(x, y) = exp(-γ \cdot ||x - y||^2)$
- sigmoid: $K(x, y) = tanh(\alpha \cdot x^\intercal y + c)$

*linear svm*

- the objective is a convex, quadratic function
- the constraints are linear inequality constraints
- minimum can be found with convex optimisation algorithm (ie. quadratic program QP solver)
- alternatively an unconstrained optimization solver
	- $\arg\min_{w\in\mathbb{R}^d}\sum_{i=1}^n\max(0,1-y_i \cdot x_i \cdot w) + \nu\|w\|^2$
- the outermost $\mathbf{x}_i$ of each class determine the hyperplane, so they're called support-vectors

algorithm:

- assuming two dimensional space
- data:
	- $D = \{(x_i, y_i)\}_{i=1}^n \subset \mathbb{R}^2 \times \{-1, +1\}$
	- features = $\{(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)\}$
	- labels = $\{-1, +1\}$
- hyperplane:
	- $\{x\in \mathbb{R}^2: x \cdot w = 0\}$
	- $\mathbf w^\intercal  \cdot \mathbf x = 0$
	- $\mathbf w$ = normal vector to hyperplane (not normalized)
	- the hyperplane can also have a bias $b$
- two subspaces:
	- $P_{w, \gamma} = \{x\in \mathbb{R}^2: x \cdot \frac{w}{||w||} \geq +\gamma\}$
	- $N_{w, \gamma} = \{x\in \mathbb{R}^2: x \cdot \frac{w}{||w||} \leq -\gamma\}$
	- $w$ is be normalized with $\frac{w}{||w||}$
- we're looking for weights that predict labels correctly:
	- $\forall i: y_i = + 1 \Rightarrow x_i \in P_{w, \gamma}$
	- $\forall i: y_i = - 1 \Rightarrow x_i \in N_{w, \gamma}$
	- can be rewritten as
	- $\forall i: y_i (\mathbf{w}^\intercal \mathbf{x}_i) \geq 1$
	- $\forall i: (\mathbf{w}^\intercal \mathbf{x}_i) \geq 1$

*hard-margin svm*

- maximizes margin between outliers to place hyperplane and accuracy
- overfits, doesn't generalize
- idea:
	- the distance between the two hyperplanes is $\frac{2}{||w||}$
	- so to maximize it we need to minimize $||w||$
- linear case:
	- $\underset{w\in\mathbb{R}^D}{\text{argmin}}\quad\|w\|^2_2$
	- while also $\forall i: (\mathbf{w}^\intercal \mathbf{x}_i) \geq 1$
- non linear case:
	- $\underset{w\in\mathbb{R}^D}{\text{argmin}}\quad c^\intercal K c$

*soft-margin svm*

- has a tolerance of $\xi$ to missclassifications
- generalizes better
- linear case:
	- $\underset{w\in\mathbb{R}^D,\xi\in\mathbb{R}^n}{\text{argmin}}\quad\sum_{i=1}^n\xi_i + \nu\cdot\|w\|^2$
	- while also $\forall i: (\mathbf{w}^\intercal \mathbf{x}_i) \geq 1 - \xi_i$
	- where $\xi_i \geq 0$
- non linear case:
	- $\underset{w\in\mathbb{R}^D,\xi\in\mathbb{R}^n}{\text{argmin}}\quad\sum_{i=1}^n\xi_i + \nu\cdot\ c^\intercal K c$

# mlp - multi layer perceptron

*linear regression*

- $w_0, w_1$ = bias, slope
- analytical solution (not faster than gradient descent): ${\mathbf w = (\mathbf X^\intercal ~ \mathbf X)^{-1} ~ \mathbf X^\intercal ~ \mathbf y}$

algorithm:

- initialize $w_0, w_1$
- until convergence of error, for each training example $(x, y)$ do:
	- ${w}_0 \leftarrow {w}_0-\alpha \cdot \frac{\partial L(w_0,w_1)}{\partial w_0}$
	- ${w}_1 \leftarrow {w}_1-\alpha \cdot \frac{\partial L(w_0,w_1)}{\partial w_1}$
	- where:
		- $L$ = loss function, sum of squares RSS: $\mathrm{L}({w}_0,{w}_1)=\sum_{i=1}^N\left(y_i-(w_0+w_1*x_i)\right)^2$ 

*polynomial regression*

- polynomial with one weight per coefficient
- loss function:
	- $L$ = rss + $\lambda$ · regularization
	- $L = \mathrm{RSS}({w}_0,\cdots,{w}_n) + \lambda \cdot L_{1 ~ or ~2}(w)$
		- residual sum squared rss: $\text{RSS} = \sum_i e_i^2$
		- lasso regression: $L_1 = ||\mathrm{w}||_{1} = |w_0| + \dots + |w_n|$
		- ridge regression: $L_2 = ||\mathrm{w}||_{2}^2 =  w_0^2 + \dots + w_n^2$

---

*perceptron*

- binary classifier
- linear decision boundary - doesn't converge for xor function
- geometric intuition: weight vector is perpendicular to the decision boundary, adding the input vector to it corrects its angle

algorithm:

- initialize weights and biases
- until convergence of error, for each training example $(x, y)$ do:
	- $\hat y \leftarrow f(x) = \sigma(\mathbf w^\intercal \cdot \mathbf x + b)$  –  predict output
	- $w \leftarrow w  + \alpha \cdot (y- \hat y) \cdot x$  –  update weights
	- $b \leftarrow b + \alpha \cdot (y - \hat y)$  –  update bias
	- where:
		- $\sigma$  = activation function (ie. linear, sigmoid, relu, softmax, …)
		- $b$ = bias, can be another $w_i$ with constant $x_i$
		- $\alpha$ = learning rate

*mlp - multi layer perceptron*

- can approximate functions by composing different parts of the activation function in each layer
- 1 hidden layer with enough nodes can approximate any continuous function
- deep learning = stacking mlps
- backprop + gradient descent = trace error back to first layer with chain rule $w_i \leftarrow w_i - \frac{\partial \text L}{\partial \mathbf W}$

algorithm:

- initialize weights and biases
- until convergence of error, for each training example $(x, y)$ and layer $i$ do:
	- feed forward:
		- $\hat y_i = \sigma(\mathbf W^\intercal_i \cdot \mathbf x + b)$
	- backprop:
		- $w_i \leftarrow w_i - \alpha \cdot \frac{\partial e(n)}{\partial x(n)} \cdot y_i(n)$
		- where:
			- $x(n)$ = node input for sample $n$
			- $e(n)$ = node error for sample $n$ based on loss function
				- cross entropy: $\text{L}(\hat{y},y,W) = -y \cdot \ln(\hat{y})-(1-y) \cdot \ln(1-\hat{y})$
			- $y_i$ = output of ancestor node
			- $\alpha$ = learning rate

*mlp hyperparameters*

- **architecture**:
	- num hidden layers = <2x neurons in input layer
	- num neurons per layer = ~2/3 of input dims, >log2(num of classes)
	- dropout = randomly freeze units for some samples, improves robustness and units becoming interdependent
- **gradient descent**:
	- num of samples per weight-update
	- iterative = 1
		- stochastic gradient descent
		- momentum = increases speed of convergence if all gradients point to same direction
		- adaptive moment estimation (adam) = moving average over past gradient magnitudes
	- batch = all
		- gradient descent
	- mini-batch = $2^x$
		- batch normalization = updating weights with average gradient across all samples in batch
- **learning rate**:
	- learning rate sets decision boundaries, overfits if high, underfits if low
	- random value
	- time decay: $\alpha_t=\alpha_0 / t$
	- exponential decay: $\alpha_t=\alpha_0\cdot\exp(-t \cdot k)$
	- L2 regularization: $\alpha \cdot ||W||^2$
- **initial weights**:
	- just right to avoid exploding / vanishing gradients
	- ReLu avoids vanishing gradients, no saturation, cheap to compute

*cnn - convolutional neural networks*

- i. convolutional layer = learn and apply convolutional kernels, extract features
	- use same kernel per layer
	- multiple small kernels and deeper network prefered to large kernel
- ii. subsampling layer = max/avg pooling of patches, reduce data
	- can also be done by 1x1 kernel
- iii. fully connected layer = combine results with mlp

output size:

- $n\times n \circledast f\times f \Rightarrow \left\lfloor\frac{n+2p-f}{s}+1\right\rfloor\times\left\lfloor\frac{n+2p-f}{s}+1\right\rfloor$
- where:
	- $n$ = input
	- $f$ = kernel filter
	- $p$ = padding
	- $s$ = stride

# rnn - recurrent neural network

*rnn - recurrent neural network*

- a function applied on every element in input seequence, while updating an inner state
- input and output can have arbitrary length
- backprop done by unrolling network
- exploding/vanishing gradients can be solved through gradient-clipping (max limit) or lstm, gru

algorithm:

- $a_h(t) = U \cdot x(t) + W \cdot s(t\text–1)$  ⟶  network input
- $A_0(t) = V_s(t)$  ⟶  unit output
- $O(t) = f_0(a_0(t))$  ⟶  network output
- $s(t) = f_h(a_h(t))$  ⟶  memory output of this iteration
- where:
	- $x$ = input
	- $t$ = time
	- $s$ = state
	- $U, W$ = input and memory weights (constant)

*lstm - long short term memory*

- long term memory (ltm) = scalar, not weighted
- short term memory (stm) = weighted
- forget gate = limit how much is remembered
- output gate = decide stm for next iteration

*gru - gated recurrent units*

- similar performance to lstm but fewer parameters

*auto-encoder*

- reconstruct input, generate representation in bottleneck
- used for representaiton learning, dimensionality reduction

# rl - reinforcement learning

- http://incompleteideas.net/book/RLbook2020.pdf
- https://www.youtube.com/playlist?list=PLnn6VZp3hqNvRrdnMOVtgV64F_O-61C1D

*rl - reinforcement learning*

- environment = has state (is just called state)
	- reward signal = immediate or delayed signal sent to agent
- agent = can read / update state through actions
	- policy = state-action matrix (behavior)
	- value estimate = action-reward matrix (estimated rewards for actions)
	- has internal model of environment, learns based on own experience
- tabular solution method = state-action-matrix is small enough, optimal policy and value can be found

example: updating value matrix

- $V(S_{t-1})\leftarrow V(S_{t-1})+\alpha \cdot \left[V(S_{t})-V(S_{t-1})\right]$
- new estimate ← old estimate + step size · (error in estimate)

*exploration-exploitation tradeoff*

- exploitation = choose actions based on rewards (greedy)
	- $A_t\doteq\arg\max_aQ_t(a)$
- exploration = but occasionally also try new actions
	- $\varepsilon$-greedy – try non optimal action with probability $\varepsilon$
	- optimistic greedy – larger initial values (for stationary problems)
	- upper confidence bound – adapt based on closeness to a max value:
		- $A_t\doteq\underset{a}{\arg\max}\left[Q_t(a)+c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]$
		- where:
			- $\ln(t)$ = natural logarithm of $t$
			- $N_t(a)$ = number of times action was selected prior to $t$
			- $c$ = degree of expliration

*stationary reward probability distribution*

- reward for action, based on probabilities:
	- $q_*(a)\doteq\mathbb{E}[R_t \mid A_t=a]$
	- where:
		- $A_t$ = action
		- $R_t$ = reward for action
		- $\mathbb{E}$ = expectation, sum of rewards times probability
- reward for action, based on experience:
	- $Q_t(a) = \large \frac{\sum_{i=1}^{t-1}R_i\cdot\mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}}$
	- = sum of rewards from $a$, divided by how often $a$ was chosen so far
	- should ideally converge to $q_*(a)$ 
	- if repeated $n \texttt{-}1$ times this results in:
		- $Q_{n+1}=\frac{1}{n}\sum_{i=1}^nR_i = \cdots = Q_n+\frac{1}{n} \cdot \Big[R_n-Q_n\Big]$

algorithm:

- for all actions $a\in A$ initialize:
	- $Q(a) \leftarrow 0$
	- $N(a) \leftarrow 0$
- loop forever:
	- $A_t\doteq\arg\max_aQ_t(a)$
	- $R \leftarrow \text{execute}(A)$
	- $N(A) \leftarrow N(A) + 1$
	- $Q(A) \leftarrow Q(A) + \frac 1 {N(A)} \cdot \Big[R-Q(A)\Big]$

*non-stationary reward probability distribution*

- rewards change over time, we can't just take the average $\frac 1 n$ for updates
- same algorithm but add more weight to more recent observations
- $Q_{n+1}\doteq Q_n+\alpha \cdot \Big[R_n-Q_n\Big] = \dots = (1-\alpha)^n \cdot Q_1+\sum_{i=1}^n\alpha \cdot (1-\alpha)^{n-i} \cdot R_i$

*mdp - markov decision processes*

- reward for action not only depends on action but also state: $q_*(s,a)$

environment state probabilities:

- $p:\mathcal{S}\times\mathcal{R}\times\mathcal{S}\times\mathcal{A}\mapsto[0,1]$   ⟶  next (state, reward) based on previous (state, reward)
- $p(s',r \mid s,a) \doteq \Pr[S_t=s',~R_t=r \mid S_{t-1}=s,~A_{t-1}=a]$
- $p(s'\mid s,a) \doteq \Pr[S_t=s'\mid S_{t-1}=s,A_{t-1}=a] = \sum_{r\in\mathbb{R}}p(s',r \mid s,a)$
- $\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{R}}p(s',r \mid s,a)=1$  ⟶  sum of probabilities
- for all:
	- $s, s' \in \mathcal S$ = variable with a known distribution
	- $r \in \mathcal R \subset \mathbb R$ = variable with a known distribution
	- $a \in \mathcal A(s)$

reward for action, based on experience:

- $r(s,a) \doteq \mathbb{E}[R_t\mid S_{t-1}=s,A_{t-1}=a] = \sum_{r\in\mathbb{R}}r\sum_{s'\in\mathcal{S}}p(s',r \mid s,a)$
- $r(s,a,s') \doteq \mathbb{E}[R_t\mid S_{t-1}=s,A_{t-1}=a,S_t=s'] = \sum_{r\in\mathbb{R}}r \cdot \frac{p(s',r | s,a)}{p(s' | s,a)}$

policy function:

- $\pi(a \mid s)$ = probability of selecting action $a$ in state $s$

state-value function:

- $v_\pi(s) \doteq \mathbb{E}_\pi[G_t\mid S_t=s] = \mathbb{E}_\pi\Big[\sum_{k=0}^\infty\gamma^kR_{t+k+1} ~\Big|~ S_t=s\Big] = \dots \quad \sum_{a}\pi(a|s) \cdot \sum_{s',r}p(s',r \mid s,a) \cdot \Big[r+\gamma \cdot v_{\pi}(s')\Big]$
	- bellman equation
	- where:
		- $G_t \doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots = \sum_{k=0}^{\infty}\gamma^{k} \cdot R_{t+k+1}$ = reward sequence weighted by recency
		- $\sum_{a}\pi(a|s)$ = probability of taking action
		- $\sum_{s',r}p(s',r \mid s,a)$ = probability of this state-reward pair to occur on the given action
		- $\Big[r+\gamma \cdot v_{\pi}(s')\Big]$ = recursive call of this function to get previous reward (use dynamic programming)
- finding the optimum:
	- $v_*(s)\doteq\max_\pi v_\pi(s)$

action value function:

- $q_{\pi}(s,a) \doteq \mathbb{E}_{\pi}[G_{t}\mid S_{t}=s,A_{t}=a] = \mathbb{E}_{\pi}\Big[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1} ~\Big|~ S_{t}=s,A_{t}=a\Big]$
- finding the optimum:
	- $q_*(s, a)\doteq\max_\pi q_\pi(s, a) =\mathbb{E}\Big[R_{t+1}+\gamma \cdot v_*(S_{t+1})\mid S_t=s,A_t=a\Big]$

*first-visit monte-carlo*

- estimates state-value function $V \approx v_\pi$ for a given policy $\pi$
- based on sample sequences of states, actions, rewards from experiences made so far
- converges as the number of state visits increases
- how it works: by taking the the average of the returns from the first visits to state $s$

algorithm:

- initialize:
	- $\forall s: V(s) \leftarrow$ random value
	- $\text{Returns} \leftarrow [\varnothing]$
- loop forever:
	- $\text{episode} \leftarrow (S_0, A_0, ~ R_1, S_1, A, ~\dots, S_{T-1}, A_{T-1}, R_T)$ generated with $\pi$ 
	- $G \leftarrow 0$
	- for each $t$ in $\text{episode}$:
		- $G \leftarrow \gamma \cdot G + R_{t+1}$
		- if $S_t \not \in \text{episode}$:
			- $\text{Returns}[S_t] \leftarrow \text{Returns}[S_t] \cup G$
			- $V(S_t) \leftarrow \text{average}(\text{Returns}(S_t))$

*exploring-starts monte-carlo*

- improves policy $\pi \approx \pi_*$
- the idea behind "monte-carlo control" is that $\pi$ and $q_\pi$ mutually improve eachother

algorithm:

- initialize:
	- $\forall s: \pi(s) \leftarrow$ random value
	- $\forall s,a: Q(s,a) \leftarrow$ random value
	- $\text{Returns}(s, a) \leftarrow [(\varnothing, \varnothing)]$
- loop forever:
	- $\text{episode} \leftarrow (S_0, A_0, ~ R_1, S_1, A, ~\dots, S_{T-1}, A_{T-1}, R_T)$ generated with $\pi(S_0, A_0)$
	- $G \leftarrow 0$
	- for each $t$ in $\text{episode}$:
		- $G \leftarrow \gamma \cdot G + R_{t+1}$
		- $\text{Returns}[S_t] \leftarrow \text{Returns}[S_t] \cup G$
		- $Q(S_t, A_t) \leftarrow \text{average}(\text{Returns}(S_t))$
		- $\pi(S_t) \leftarrow \arg\max_a Q(S_t, a)$

# combining models

*transfer learning*

- build up on the work of someone else / on a different dataset
- a) off-the-shelf = discard model, reuse layers in your own
- b) supervised task adaptation = use model, replace layers with your own
	- freeze / fine tune selected layers (fine tuning always helps)
	- first layers (feature detection) generalize better last layers (classification)

*model selection*

- train multiple models, pick best one

*ensemble learning*

- train multiple models, combine predictions
- compounded accuracy:
	- this estimation doesn't hold up in practice
	- assumption: majority voting, independence
	- $p_{maj}=\sum_{m=\lfloor L/2\rfloor+1}^L \binom{L}{m} \cdot p^m \cdot (1-p)^{L-m}$
	- where:
		- $p$ = accuracy
		- $L$ = num classifiers

bagging (bootstrap aggregating):

- = models trained on bootstrap samples
- = parallel evaluation of independent models
- works best when models are non-deterministic

boosting:

- = stacked layers of models, each trying to fix shortcomings of previous layer
- = sequential evaluation of models
- **adaBoost**:
	- linear combination of models
	- model-weights adjusted based on loss
	- data-weights adjusted based on missclassifications
	- $H ( x ) = \text{sign} (\sum_{t = 1}^{T} \alpha_t \cdot h_t( x ) )$
	- where:
		- $h$ = model
		- $T$ = num iterations
		- $\alpha$ = model weights - initially 1/n
		- $D_t$ = train-set weights - initially 1/n then multiplied by $e^{\pm \alpha t}$
		- sampling probabilities can also be weighted
- **gradient boost**:
	- gradient descent + boosting
	- model-weights are constant
	- $F ( x ) = \sum_t \rho_t \cdot h_t( x )$
	- add models during training:
		- i. train initial zero-rule model and put into set $F(x)$
			- compute gradient over train-set
		- ii. compare models $h(x)$ to add:
			- $\text{residual}_i= y_i - F(x_i) = ({F(x_i) + h(x_i)}) - F(x_i)$
			- residual is equivalent to negative gradient of ensemble
		- iii. predict:
			- regression: mean
			- classification: log(odds) of majority class

# automl

meta-learning = learning how to train learning algorithms

*no-free-lunch theorem*

- no model can generalize to all kinds of tasks
- when averaged across all tasks, all models are equally accurate
- but we assume: ccwa
	- ocwa (open classification world assumption) = models aren't comparable
	- ccwa (closed classification world assumption) = models are comparable - because we trust our experiments to reflect generalizability

*rice's framework*

- framework for finding the best model
- for a given problem instance $x \in P$ with features $f(x) \in F$ find the selection mapping $S(f(x))$ into algorithm space, such that the selected algorithm $a \in A$ maximizes the performance mapping $y(a(x)) \in Y$
- i. feature extraction:
	- $x \in P \mapsto f(x) \in F$
	- $x$ = dataset from problem-space $P$
	- $f$ = extracted features from feature-space $F$
	- we assume that we can access a subset $P'$ of the universe $P$
	- the dataset $x$ contains features $f(x)$, but we also have to add labels $t(x)$
- ii. algorithm selection:
	- $S: f(x) \mapsto \alpha \in A$
	- $\alpha$ = algorithms that can solve the problem in algorithm-space $A$
- iii. performance measurement:
	- $\alpha(x) \mapsto y \in Y$
	- $y$ = performance metrics in performance-metric space
	- next select another algorithm and repeat

*landmarking*

- landmarkers = simple and fast algorithms (ie. naive bayes, 1nn, 1r, …)
- landmarking features = performance of landmarkers
- landmark learner = selects best performing landmarker
- the best performing landmarker tells us something about the dataset
- ie. linear classifier does well on linearly seperable data

*hyperparameter optimization*

- loss function for hyperparams:
	- $f(\boldsymbol{\lambda})=\frac{1}{k}\sum_{i=1}^{k}{L}(A_{\boldsymbol{\lambda}},{D}_{\mathrm{train}}^{(i)},{D}_{\mathrm{valid}}^{(i)})$
	- where:
		- $A_\lambda$ = algorithm configured with hyperparam $\lambda$
		- $\mathbf \Lambda = \Lambda_1 \times \dots \times \Lambda_n$ = hyperparam-space from different domains
- search types:
	- grid search = test equi-distanced values on discrete scale
	- random search = sample from a distribution, can outperform grid-search
	- sequential model-based bayesian-optimization (smbo): probabilistic regression model with loss function

# mlsec

*mlsec*

- metrics = privacy, robustness, interpretability, fairness, …
- attack types:
	- integrity = cause unexpected behavior
	- confidentiality = breach sensitive data
	- availability = disrupt service, reduce service quality

*adversarial examples*

- minimal pertubation $t$ of input $X$ to achieve misclassification
- requires decision boundary search
- perturbation should be small enough to be invisible to humans
- search with access to model:
	- greedy search = search around classification boundary
		- inefficient
	- fast gradient sign (fgsm) = use loss-function the model was trained with
		- $x+\varepsilon\cdot \text{sign}(\nabla_{\boldsymbol{x}}J(\theta,x,y))$
		- backprop is non-deterministic, success not guaranteed but ~60-70%

*backdoor attacks*

- train on poisoned data so some inputs trigger misbehavior - without influencing performance
- works well if models overfit

*xai - explainable ai*

- white box models are trustworthy
- black-box models can be made more transparent
- types:
	- ex-ante = explain statistical workings, feature plots, bias, variable dependencies
	- intrinsic = explain weaknesses due to the nature of the model
	- post-hoc = explain model through it's weights – activation maps, textual explainations, negative examples (counterfactuals), anchors (deciding parts of input)
	- specific / agnostic = specific model config / general
	- local / global = per region / holistic

*data privacy*

- data breach = identity disclosure, re-identification through attribute disclosure
- meta-information = membership disclosure
- data sanitization = removing identifiers, k-anonymity, synthetic data
- differential privacy = adding noise/perturbations to data, without changing distribution
- federated learning = distributed training
- secure computation = multi party computation, homomorphic encryption

*inference / inversion attacks*

- membership inference attack = reconstruct model weights, training data (if model overfits) by querying it frequently
- model inversion = recreating training data by maximizing model confidence for specific labels
- model extraction = learning weights to ie. jailbreak models

# mlops

*mlops*

- ops = operation of data services and infrastructure
- in the future we will have smaller datasets, smaller highly-specialized models (micro specialists)
- models should be modular, composable, easy to orchestrate

*ml lifecycle*

- 1) data engineering
- 2) training
- 3) evaluation / validation
	- unit testing, integration testing, …
- 4) deployment
- 5) re-evaluation / experimentation
	- continuous learning
	- a/b testing, user studies, …

*infratstructure*

- memory:
	- data-lake = raw data for training, cheap storage (ie. s3)
	- feature store = must be accessible for predictions
	- model store = containerized models
- compute:
	- training systems (dev) = distributed training, k8s, spark clusters, …
	- inference systems (prod) = containerized services, …
