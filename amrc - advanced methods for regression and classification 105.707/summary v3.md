# feature selection

to decorrelate, handle $p > n$

*ordinary least squares OLS*

- $\hat y_i = \hat\beta_0+\sum_{j=1}^px_j \hat\beta_j$
- $\varepsilon_i = y_i- \hat y_i$
- $\text{RSS}(\hat\beta)=\sum_{i=1}^n(y_i- \hat y_i)^2$
- normal equation:
	- ${\hat y} =  X  \cdot {\hat \beta}$
	- ${\hat y} =  X \cdot ({X}^\top{X})^{-1}{X}^\top  \cdot {y}$
	- ${\hat y} = {H} \cdot {y}$ (hat matrix)
- correlated features = no unique solution ($X$ not invertible, $X^\top X$ not singular, unreliable inference due to $d_j$)
- inference assumption:
	- $\varepsilon_i \sim N_n(0,~\sigma^2)$
	- $\varepsilon_i$ is an i.i.d. independent, identically distributed random var (given fixed $X$)
	- $\hat{\sigma}^{2} = \frac{\text{RSS}}{n-p-1}$ = unbiased variance estimator
	- $\hat{{\beta}}\sim N_{p+1}({\beta},~\sigma^2({X}^\top{X})^{-1})$
- gauss-markov-theorem: BLUE = best (lowest variability), linear, unbiased ($\mathbb E(\hat{{\beta}})={\beta}$) estimator under inference assumption

*z or t-test*

- $H_0: \beta_j = 0$
- $H_1: \beta_j \not = 0$
- ${z_j=\frac{\hat{\beta}_j}{{\sigma}\sqrt{d_j}}} \sim N(0,1)$
- $d_j$ = a diagonal elem of $({X}^\top{X})^{-1}$
- use $\hat \sigma$ for t-test

*decomposition of variance*

- $\text{TSS = RegSS + RSS}$
- $\sum_{i} (y_i - \bar y)^2 = \sum_{i} (\hat y_i - \bar y)^2 + \sum_{i} (y_i - \hat y_i)^2$ 
- total variance = explained component + unexplained component

*anova f-test*

- $H_0: \forall i: \beta_i = 0$
- $H_1: \exists i: \beta_i \not=0$
- $F = \frac{\text{RegSS}/p}{\text{RSS}/(n-p-1)} \sim F_{p,n-p-1}$
- F = explained variance per df / unexplained variance per df

*extra-sum-of-squares f-test*

- $H_0$: smaller (nested/subset) model is good enough
- $H_1$: larger model is better
- $p_0 < p_1 \leq p$
- $F = \frac{(\text{RSS}_0 - \text{RSS}_1)/(p_1 - p_0)}{\text{RSS}_1/(n - p_1 - 1)} \sim F_{p_1 - p_0, n - p_1 - 1}$
- F = change in residuals per additional parameter, normalized by $\hat \sigma ^2$

*r-squared*

- $R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{RegSS}}{\text{TSS}} = \text{Cor}^2(y, \hat y)$ (ratio of explained variance)
- $\tilde R^2 = 1 - \frac{\text{RSS} / (n-p-1)}{\text{TSS} / (n-1)}$ (adjusted, also penalizes model size)

*information criteria*

- = performance metric, also penalizing model size
- akaike's information criterion:
	- if $\sigma$ known) $\text{AIC} = \text{RSS}/\sigma^2 + 2p$
	- if $\sigma$ unknown) $\text{AIC} = n \cdot \log(\text{RSS}/n) + 2p$
	- best for descriptive models, absolute value can't be interpreted, based on kullback-leibler KL
- bayes information criterion:
	- if $\sigma$ known) $\text{BIC} = \text{RSS}/\sigma^2 + \log(n) \cdot p$
	- if $\sigma$ unknown) $\text{BIC} = \text{-}2 \cdot \max \log L + \log(n) \cdot p$
	- best for predictive models, absolute value can't be interpreted, favors simpler models than AIC
- mallow's cp:
	- $C_p = \text{RSS}/ \sigma^2 + 2p - n$
	- estimate $\sigma^2$ if unknown
	- mainly used for stepwise regression (iteratively adding/removing features)

*evaluation*

- cross validation:
	- i) split full set in $k$ folds
	- ii) 1 fold for test set, remaining for train set
	- iii) repeat $k$ times, training $k$ models
	- $\text{Err} = \frac{1}{n} \sum_i \text{Loss}_i$
	- samples occur exactly once
	- leave-1-out-cv: set $k\text{=}n$, too compute intensive, high variance
- bootstrap:
	- i) bootstrap / train set = sample $n$ elems with replacement
	- ii) out-of-bag / test set = use leftovers $(1-\frac{1}{n})^n \approx \frac{1}{3}$
	- ii) repeat $k$ times, training $k$ models
	- $\text{Err} = \frac{1}{k} \sum_i \frac{1}{|\text{OOB}|} \sum_j \text{Loss}_{i,j}$
	- samples overrepresented, too optimistic due to data-leaks
	- leave-1-out-bootstrap = avoid data leaks. for each elem use models $C^{-i}$ that weren't trained on it $\text{Err}=\frac{1}{n} \sum_{i} \frac{1}{|C^{-1}|} \sum_{j} \text{Loss}_{i,j}$

*variable selection*

- search tree of size $2^p$, using f-statistic, information criteria
- backward stepwise selection = full model, recursively drop least useful params
- forward stepwise selection = empty model, recursively add most useful parameters
- best subset regression = use info criteria
- leaps and bounds algorithm = heuristic, drop entire search branches / coeff subsets

# dimensionality reduction

*bias vs. variance*

- reduce variance at the cost of increased bias
- $\text{MSE} = \mathbb{E}[(y - \hat y)^2] = {\sigma^2} + {\text{Var}(\hat y)} + {\left (\mathbb{E}[y - \hat y]\right )^2}$
	- $\sigma^2$ = irreducible error (independent of model)
	- $\text{Var}(\hat y)$ = variance of estimates (model instability depending on train set)
	- $\mathbb{E}[y - \hat y]^2$ = squared bias (= zero for OLS, because it's unbiased)

*principal component regression PCR*

- = expressing data with a simpler coordinate system, dimensionality reduction, decorrelation
- unsupervised method: only considers variance of $X$
- preprocess: scale (ie. normal dist with z-score), center (subtract the mean)
- $Z = XV$
	- $Z$ = principal components, uncorrelated, sorted by variance → orthogonal axes aligned with directions of largest spread, to minimize information loss at projection
	- $V$ = found next through spectral decomposition (expects input to be square, symmetric)
- $\text{Cov}(X) = V A V^{\top}$
	- $\text{Cov}(X)$ = variance between $x_i$ and $x_j$ at index $i,j$
	- $V$ = eigenvectors, normalized, orthogonal ($V^\top \text{=} V^{-1}$)
	- $A$ = eigenvalues/variance as diagonal elements, sorted in descending order
- $\text{Cov}(Z) = A$
	- validating that PCs are uncorrelated: their covariance matrix is a diagonal matrix
- use first $q <p$ vectors
	- $y = X \beta + \varepsilon$
	- $y = (X V) (V^\top \beta) + \varepsilon$ (because $VV^\top \text{=} I$)
	- $y = Z \cdot \theta + \varepsilon$
	- $y=Z_{1:q}\theta_{1:q}+ \tilde\varepsilon$
	- $\hat \theta = (Z^\top_{1:q} Z_{1:q})^{-1} Z^\top_{1:q} y$
	- $\tilde{\hat \beta} = V_{1:q} \hat \theta_{1:q}$ → apply $V$ (to undo $V^\top$) and interpret results in terms of original vars

*partial least squares PLS regression*

- supervised method: maximizing covariance between $X, y$
- preprocess data first: scale, center
- $T = XW$
	- $T$ = scores
	- $W$ = loadings
- ${w}_{k}=\text{argmax}_{{{w}}}\text{Cov}({y},{X}{w})$
	- $\|{w}_k\|=1$ (normalized)
	- $\text{Cov}(X{w}_k,X{w}_j)=0,\forall j<k$ (subsequent vectors orthogonal to previous ones)
- use first $q <p$ vectors
	- $y = T \gamma + \tilde{\varepsilon}$
	- $y = X W \gamma + \tilde{\varepsilon}$ (where $W \gamma \approx \beta$)

*continuum regression*

- $w_k = \text{argmax}_w \{[\text{Cov}(y, Xw)]^2 \cdot [\text{Var}(Xw)]^{\frac{{\delta}}{1-\delta}-1}\}$
- first term is PLS, second term is PCR, $\delta \text{=} 0$ means OLS

# regularization

smooth out coefficients to reduce variance

*ridge regression*

- L2-norm penalty, coefficients never reach zero
- preprocess: scale, center, set $\beta_0 = \bar y$ and drop it
- $\hat{\beta}_{\text{Ridge}} = {\text{argmin}}_\beta \{ \text{RSS} + {\lambda\sum_{j=1}^p \beta_j^2} \}$
- $\sum \beta_j^2 \leq s$
- $\hat\beta_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$
	- $\lambda I$ makes $X^\top X$ invertible
- $df = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$ sum of diagonal elems
	- increasing $\lambda$ drops effective df from $p$ to $0$

*lasso regression*

- L1-norm penalty, coefficients reach zero
- preprocess: scale, center, set $\beta_0 = \bar y$ and drop it
- $\hat{\beta}_{\text{Lasso}} = {\text{argmin}}_\beta \{\text{RSS} + {\lambda\sum_{j=1}^p |\beta_j|}\}$
- $\sum | \beta_j | \leq s$
- no closed-form solution like ridge

*pcr vs. ridge*

- singular value decomposition:
	- $X = U D V^\top$
	- $U, V$ = eigenvectors (left/right singular vectors), normalized, orthogonal, sorted by singular vals
	- $D$ = eigenvalue roots $\sqrt d_i$ (singular values $\sigma$) as diagonals, proportional to variance of PCs
- pcr:
	- $Z = X V = UD$
	- $\text{Cov}(X) = \frac{1}{n-1} X^\top X = \frac{1}{n-1} VD^2 V^\top$
	- $\text{Var}(Z) = D^2$
- ridge:
	- $\hat y = X \cdot \hat \beta_{\text{Ridge}} = UD (D^2 + \lambda I)^{-1} DU^\top y$ 
	- $\hat y = X \cdot \hat \beta_{\text{Ridge}} = \sum_{j=1}^p u_j {\frac{d_j^2}{d_j^2 + \lambda}} u_j^\top y$
	- $\frac{d_j^2}{d_j^2+\lambda}$ = shrinkage factor
    - $\lambda$ = regularization strength
	- smaller $d_j^2$ (low-variance PCs) → more shrinkage
- similarities:
	- both wrongly assume PCs with higher variance $d_j^2$ are always more relevant for predicting $y$
	- neither consider correlation with $y$ (unsupervised)
	- PCR discards low-variance PCs, ridge smoothly downweights them

# linear classification

*linear regression of indicator matrices*

- $\{ x: (\hat{\beta}_{k0} - \hat{\beta}_{l0}) + (\hat{\beta}_k - \hat{\beta}_l)^\top x = 0 \}$
	- linear decision boundary $\mathbb R^p$ where two classes share points (just like LDA)
- $\hat f_k(x) = \hat \beta_{k0} + \beta_{k}^\top x$
	- independent regressions compute scores (not probabilities) or each class $\hat y_k \in \mathbb R$
- $\hat f(x) = [(1, x^\top) \hat B]^\top$
	- $Y = [y_1, \ldots, y_K ]$ = indicator matrix (one-hot-encoded)
	- $\hat B = [\hat \beta_1, \ldots, \hat \beta_K] = (X^\top X)^{-1}X^\top Y$ = coefficients for each class
- $\hat G(x) = \text{argmax}_{k\in \mathcal G} \hat f_k (x)$

*linear discriminant analysis LDA*

- assumes: shared covariance matrix $\Sigma_k = \Sigma$, posterior follows multivariate normal distribution $\varphi_k(x) \sim \mathcal{N}(\mu_k, \Sigma_k)$
- $\begin{gather}P(G{=}k \mid {x})=\frac{\varphi_k({x}) \cdot \pi_k}{\sum_{l=1}^K \varphi_l({x}) \cdot \pi_l}\end{gather}$
	- $\pi_k$ = how common class $k$ is overall (prior) → where $\Sigma_{k=1}^K \pi_k = 1$
	- $\varphi_k(x)$ = how common $x$ is in $k$ (posterior)
	- $\sum_{l=1}^K \varphi_l(\text{x}) \cdot \pi_l$ = how common $x$ is overall
- $\log (\frac{P(G=k|x)}{P(G=l|x)} ) = \delta_k(x) - \delta_l(x)$
	- find decision boundary through shared points (logarithms of odds for numerical stability)
- $\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k +  \log\pi_k$ (linear discriminant function)
- $G(x) = \text{argmax}_k \delta_k(x)$
- $\hat G(x)$ needs to be estimated from training data in practice:
	- $\hat \pi_k = \frac{n_k}{n}$ (class proportion)
	- $\hat \mu_k = \sum_{g_i = k} \frac{x_i}{n_k}$ (class mean)
	- $\hat{{\Sigma}}=\frac{1}{n-K} \sum_{k=1}^K \sum_{g_i=k}({x}_i-\hat{{\mu}}_k)({x}_i-\hat{{\mu}}_k)^{\top}$ (pooled covariance)
	- $n_k$ = number of data points per class
	- $g_i$ = true group number of data point

*quadratic discriminant analysis QDA*

- $\delta_k({x})=-\frac{1}{2} \log |{\Sigma}_k|-\frac{1}{2}({x}-{\mu}_k)^{\top} {\Sigma}_k^{-1}({x}-{\mu}_k)+\log \pi_k$
- quadratic decision boundaries
- doesn't assume shared cov matrix (but multivariate normal distribution)
- needs more parameters ($\Sigma_k$ for each class) than LDA
- both LDA and QDA work well in practice, even if assumptions not met

*regularized discriminant analysis RDA*

- $\hat{{\Sigma}}_k(\alpha)=\alpha \hat{{\Sigma}}_k+(1-\alpha) \hat{{\Sigma}}$
- $\hat{{\Sigma}}=\frac{1}{\sum_{k=1}^K n_k}(\sum_{k=1}^K n_k \hat{{\Sigma}}_{{k}})$ (regularized pooled covariance)
- $\alpha \in [0;1]$ - ranging from LDA to QDA
- allows to shrink the separate covariances of QDA towards a common covariance as in LDA

*logistic regression*

- no assumptions, can be used for inference (ie. feature importance)
- $\log (\frac{P(G = k|x)}{P(G = K|x)}) = \beta_{k,0} + \beta_{k}^\top x$
	- linear regression models the log-odds
- $P(G \text{=} k | x) = \frac{\exp\{\beta_{k,0} + \beta_k^\top x\}}{1 + \sum_{l=1}^{K-1} \exp\{\beta_{l,0} + \beta_{l}^\top x\}}$ (other classes)
- $P(G \text{=} K | x) = \frac{1}{1 + \sum_{l=1}^{K-1} \exp\{\beta_{l,0} + \beta_{l}^\top x\}}$ (reference class $K \not = k$)
	- reference class choice is arbitrary
	- satisfies $\Sigma_k P(G \text{=} k | x) = 1$ by using logistic/softmax function

maximum likelihood estimation MLE (binary classification):

- = maximize log-likelihood $l(\beta)$ to estimate the coefficients $\beta$
- $l({\beta})=\sum_{i=1}^{n}\{ y_i \cdot {\beta}^{\top} {x}_{i}-\log [1+\exp ({\beta}^{\top} {x}_{i})]\}$
	- probability of observing the actual data under the model $\beta$
	- boolean flag ${y_i} \in \{0,1\}$ to enable/disable terms for $g_i \text{=} 2$ or $g_i \text{=} 1$
- first derivative (set to zero):
	- $\frac{\partial l({\beta})}{\partial {\beta}} = X^\top (y - p)$
	- intialize $\beta \text{=} 0$
	- $p$ = vector of estimates $p(x_i; \beta_{old})$
- second derivative:
	- $\frac{\partial^2 l({\beta})}{\partial {\beta} \partial {\beta}^{\top}} = -X^\top W X$
	- $W$ = diagonal matrix with weights in diagonals $p(x_i; \beta_{old}) \cdot (1 - p(x_i; \beta_{old}))$
	- $W$ isn't diagonal from $K \geq 3$ 
- newton-raphson / iteratively reweighted least squares IRLS algorithm:
	- ${\beta}_{\text {new }} \leftarrow {\left({X}^{\top} {W} {X}\right)^{-1} {X}^{\top} {W}} \cdot {z}$
		- first term: weighted OLS
		- second term: ${z} = {X}{\beta}_{\text{old}} + {W}^{-1}({y}-{p})$ = adjusted response
		- same as: ${\beta}_{\text {new }} \leftarrow {\text{argmin}}_{{\beta}}({z}-{X} {\beta})^{\top} \cdot {W} \cdot ({z}-{X} {\beta})$

# splines

*basis expansions*

- $h_m(x) : \mathbb R^p \mapsto \mathbb R$
- $f(x) = \sum_{m = 1}^M \beta_m h_m(x)$
- linear combination of basis functions
- $M \geq p$ (ie. polynomial terms, interactions $x_j \cdot x_k$, …)

*piecewise polynomials*

- = seperate polynom of order $M$ for each of $k\text{+} 1$ intervals: $(- \infty, \xi_1),~ [\xi_1, \xi_2),~ \ldots [\xi_{k}, \infty)$
- $M(k\text{+}1)$ coefficients
- order / num coefficients = degree + 1
- continuity constraint:
	- = no abrupt jumps between intervals until derivative $M\text{–}1$
	- reduces coefficients by 1, per derivative level, per interior knot
- naturality constraint
	- = continuity constraint + linearity beyond outermost knots (by setting first and second derivative to 0)
	- reduces coefficients by 4 (2 constraints per boundary · 2 boundaries)

*splines*

- = continuous piecewise polynomials
- $f(x) = \sum_{j=1}^M \beta_j x^{j-1} + \sum_{l=1}^k \beta_{M+l} (x - \xi_l)_+^{M-1}$
	- $h_j(x) = x^{j-1}$
	- $h_{M+l}(x) = (x - \xi_l)_+^{M-1} = \max(0,~ x - \xi_l)^{M–1}$
	- $j$ iterates over polynom degrees, $l$ iterates over knots
- $M\text{+}k$ coefficients
- hyperparams: order (usually 4), number of knots, placement of knots

*cubic smoothing splines*

- = regularized splines
- $f(x) = \sum^n_{j=1} N_j(x) \cdot \theta_j$
- $f(x) = N \cdot \theta$
	- $N_j$ = natural cubic splines → knots placed at all data points $x_i$ (later shrunken away)
	- $\theta_j$ = coefficients (of splines)
- $RSS(f, \lambda) = \text{RSS} + \lambda \int f''(t)^2 dt$
- $RSS(\theta, \lambda) = {(y - N \theta)^\top (y - N \theta)} + {\lambda \theta^\top \Omega_N \theta}$
	- $\Omega_N = \int N_j^{''}(t) \cdot  N_k^{''}(t) ~ dt$
	- lambda penalizes roughness: from interpolating all points, to a straight line
- $\hat f(x) = N \cdot \hat \theta$
	- $\hat \theta = (N^\top N + \lambda \Omega_N)^{-1} \cdot N^\top y$ (generalized ridge regression)

finding $\lambda$:

- $S_\lambda = N (N^\top N + \lambda \Omega_N)^{-1} \cdot N^\top$
	- computing smoother matrix is too expensive because $M \text{=} n$
- $H_\xi = B_\xi (B_\xi^\top B_\xi)^{-1} B_\xi^\top$
	- hat matrix approximates $S_\lambda$ with a subset of knots and basis functions $M \ll n$
- $df_\lambda = \text{trace}(S_\lambda) = \sum_i S_{ii}(\lambda) = M$
- $\text{MSE}_{\text{LOO}}(\lambda)  = \frac{1}{n} \sum_{i=1}^n ( \frac{y_i - \hat f_\lambda(x_i)}{1 - S_{ii}(\lambda)} )^2$
	- leave-1-out-cv
	- computing diagonals $S_{ii}(\lambda)$ is too expensive
- $\text{MSE}_{\text{GCV}}(\lambda) = \frac{1}{n} \sum_{i=1}^n ( \frac{y_i - \hat f_\lambda(x_i)}{1 - \text{trace}(S_{\lambda})/n} )^2$
	- generalized-cv
	- approximates $S_{ii}(\lambda)$ with the mean of its trace

# generalized additive models

*generalized linear models GLM*

- = linear regression + link function $g$ to wrap non-normal data
- $g(\mu) = X \beta$
- $\mathbb E (y | X) = \mu = g^{-1} (X \beta)$
- $\mathbb E (y|X)$ follows a straight line, assumption is met
- $y \sim$ distribution from the exponential family

*generalized additive models GAM*

- = GLM + basis expansions
- $g(\mu(x)) = \alpha + \sum_{j = 1}^p f_j(x_i)$
- $RSS(\alpha, f_1, \ldots, f_p) = \text{RSS} + { \sum_{j=1}^p \lambda_j \int f_j''(t_j)^2 dt_j}$
	- $f_j$ = cubic smoothing splines, knots placed at all data points $x_i$ 
	- seperate $\lambda_j$ defined for each $f_j$

backfitting algorithm:

- assume $X$ is non-singular
- initialize $\hat f_j \leftarrow  0$ and $\alpha \leftarrow  \bar y$
- iterate $j \in [1;p]$:
	- $r_{ij} \leftarrow y_i - \left(\hat{\alpha} + \sum_{k \neq j} \hat{f}_k(x_{ik})\right)$ for $i \in [1; n]$
		- compute residual without using $\hat f_j$ to isolate its contribution to the model
	- $\hat{f}_j \leftarrow S_j\left( r_{1j}, r_{2j}, \ldots, r_{nj}\right)$
		- fit a cubic smoothing spline $S_j$ with the residuals
	- $\hat{f}_j \leftarrow \hat{f}_j - \frac{1}{n}\sum_{i=1}^n \hat{f}_j(x_{ij})$
		- center $\hat f_j$ so it has a mean at zero and doesn't overlap with $\hat \alpha$
	- repeat until $\hat f_j$ convergence

# trees

decide tree structure, features, thresholds

*regression trees*

- $\hat f(x) = \sum_{m = 1}^M \hat c_m \cdot I(x \in R_m)$
	- $I(x \in R_m) \in \{0, 1\}$
	- $\hat c_m = \frac{1}{n_m} \sum_{x_i \in R_m} y_i$ (average of region)
- partitioning:
	- $\min_{j,s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \right]$
	- $R_1(j,s) = \{ x \mid x_j \leq s \} \quad R_2(j,s) = \{ x \mid x_j > s \}$
	- outer loop: find feature $j$, threshold $s$ to partition into regions $R_1, R_2$ (with the lowest RSS)
	- inner loop: use $\hat c_m$ for the prediction values $c_1, c_2$
- cost complexity criterion (for pruning):
	- $c_\alpha(T) = \sum_{m = 1}^{|T|} \text{RSS} + \alpha \cdot |T|$
	- $\alpha$ = tuning parameter: 0 results in full tree, always has a unique solution
	- $|T|$ = num of leaf nodes

*classification trees*

- $k(m) = \text{argmax}_k~ \hat p_{mk}$
- $\hat p_{mk} = \frac{1}{n_m} \sum_{x_i \in R_m} I (y_i = k)$
- label of region $m$ is the class $k$ with the highest ratio
- node impurity metrics $Q_m(T)$:
	- misclassification error = $\frac{1}{n_m} \sum_{x_i \in R_m} I(y_i \neq k(m)) = 1 - \hat{p}_{mk(m)}$
	- gini index = $\sum_{k \neq k'} \hat{p}_{mk}\hat{p}_{mk'} = \sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk})$
	- cross entropy = $\text{–}\sum_{k=1}^K \hat{p}_{mk} \log \hat{p}_{mk}$
	- measure how homogenous regions are. cross-entropy and gini are preferred because they are differentiable, more sensitive to changes

*random forests*

- ensemble of $b$ trees
- train with bootstrap set
- only use a random subset of features ($\sqrt p$ for classification, $\frac{p}{3}$ for regression)
- explained variance = $1 - \text{MSE} / \hat \sigma^2_y$
- variable importance = $\hat \theta = \frac{1}{B} \sum_{b=1}^B \delta_{bj}$
	- $\pi_b = \sum_{m=1}^{|T_b|} n_m \cdot Q_m(T_b)$ = tree impurity
	- $\pi_{bj}$ = tree impurity after dropping feature
	- $\delta_{bj} = \pi_{bj} - \pi_{b}$ = variable importance (in a single tree)
	- note: features aren't dropped, but permuted: their labels get randomly shuffled so they're useless

# support vector machines

*perceptron*

- decision boundary:
	- $\mathcal L = \{ x: \beta_0 + \beta^\top x = 0 \}$
	- $\beta^\top (x_1 - x_2) = 0$ for all $x_1, x_2 \in \mathcal L$
	- $\beta^\top x_0 = \text{–}\beta_0$ for all $x_0 \in \mathcal L$
	- $\beta$ is orthogonal to hyperplane (set of all points satisfying equation)
- geometric distance to boundary:
	- $\text{dist}(x, \mathcal L) = \frac{1}{\|\beta\|} ( \beta^\top x - \underbracket{\beta^\top x_0}) = \frac{1}{\|\beta\|} ( \beta^\top x + \underbracket{\beta_0}) = \frac{1}{\|\beta\|} ( {\beta_0} + \beta^\top x)$
	- i. pick vector $({x} - {x}_0)$ from point ${x}_0$ (on hyperplane) to $x$
	- ii. project that vector onto the normal vector $\beta$ and get its length (dot product) 
	- iii. normalize result into standard euclidean units
	- a positive distance means the vector is on the side pointed to by $\beta$
- perceptron learning algorithm of rosenblatt:
	- $\text{Loss}(\beta_0, \beta) = \text{–} \sum_{i \in \mathcal M} y_i \cdot (\beta_0 + \beta^\top x)$
		- negate the entire term, because for misclassified points $\mathcal M$, the sign of the of the prediction is always different from the label $y_i$
	- $\begin{pmatrix} \beta \\ \beta_0 \end{pmatrix} \leftarrow \begin{pmatrix} \beta \\ \beta_0 \end{pmatrix} + \rho \begin{pmatrix} y_ix_i \\ y_i \end{pmatrix}$
		- $\frac{\partial \text{Loss}(\beta_0, {\beta})}{\partial {\beta}} = \text{–} \sum_{i\in\mathcal{M}} y_i{x}_i$
		- $\frac{\partial \text{Loss}(\beta_0, {\beta})}{\partial \beta_0} = \text{–} \sum_{i\in\mathcal{M}} y_i$
		- $\rho$ = learning rate
		- iterate through all observations
	- depends on: initial parameter values, processing order of observations, number of iterations
	- in contrast to svm, it converges an arbitrary solution, not a unique and optimal one

*linear svm*

- non seperable case = classes overlap, misclassifications unavoidable
- seperable case = by setting $C \rightarrow \infty$ the penalty for $\xi_i > 0$ becomes infinitely large, forcing all slack-variables to be $\xi_i = 0$ and all support vectors to lie exactly on the boundary

decision rule:

- $G(x) =  \text{sgn}(\frac{1}{\|\beta\|} ( \beta_0 + \beta^\top x))$ = sign of distance to decision boundary
- $g_i \in \{ -1, 1 \}$ = label

margin:

- $M = \min_i~ g_i \cdot \frac{1}{\|\beta\|} \cdot ( \beta_0 + \beta^\top x_i)$
- $g_i \cdot \frac{1}{\|\beta\|} \cdot ( \beta_0 + \beta^\top x_i) \geq M (1 - \xi_i)$
	- margin is the minimum distance of all points to the decision boundary
	- margin width: $2 M = 2 / \| \beta \|$
	- slack variable $\xi_i$ = distance of violation relative to margin, with absolute value of $\xi_i \cdot M$
	- a) $\xi = 0$ (full margin / on or outside the margin)
	- b) $0 < \xi_i \leq 1$ (reduced margin / inside the margin)
	- c) $\xi > 1$ (negative margin / misclassified, on the wrong side of the decision boundary)
- $\min_{\beta,\beta_0} \left(\frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \xi_i\right)$
- $g_i \cdot ( \beta_0 + \beta^\top x_i) \geq 1 - \xi_i$
- $0 \leq \xi_i$
	- simplified, for easier derivation
	- $C$ = cost param, decides margin width (magnitude of beta) vs. training errors

solution to $\max_{\beta_0, \beta} M$:

- $L_p = \frac{1}{2}\|{\beta}\|^2 - C \sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i \cdot \left(g_i \cdot (\beta_0 + {x}_i^\top {\beta}) - ( 1 - \xi_i) \right) - \sum_{i=1}^n \lambda_i \xi_i$
	- lagrange primal function
	- has to be minimzed with respect to $\beta, \xi$
	- $\alpha_i, \lambda_i, \xi_i$ have to be non-negative
- $L_d = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_jg_ig_jx_i^\top x_j$
- $0 \leq \alpha_i \leq C$
	- lagrange dual function is a lower bound and easier to solve (only dependent on $\alpha_i$)
	- i. take derivatives of $L_p$ with respect to $\beta_0, \beta$ and set them to zero
	- ii. substitute back into the original equation
- for optimality $L_d$ also has to satisfy the karush-kuhn-tucker KKT conditions:
	- $\sum_{i=1}^n \alpha_i g_i {x}_i = {\beta}$
	- $\sum_{i=1}^n \alpha_i g_i = 0$
	- $C = \alpha_i + \lambda_i$
	- $\lambda_i \cdot \xi_i = 0$
	- $\alpha_i \cdot \left( g_i({x}_i^\top {\beta} + \beta_0) - (1 - \xi_i ) \right) = 0$
	- $g_i({x}_i^\top {\beta} + \beta_0) - (1 - \xi_i ) \geq 0$
- the last KKT condition means that there are 2 kinds of points:
	- at least either $\alpha_i = 0$ or the term $g_i ({x}_i^\top {\beta} + \beta_0) = (1 - \xi_i)$ must be satisfied
	- a) support vectors: $\alpha_i > 0 \implies g_i ({x}_i^\top {\beta} + \beta_0) = 1 - \xi_i$
		- a1) on one of the boundaries: $\xi_i = 0, \quad 0 < \alpha_i < C$
		- a2) inside the margin: $\xi_i > 0, \quad \lambda_i = 0, \quad \alpha_i = C$
	- b) non support vectors: $g_i ({x}_i^\top {\beta} + \beta_0) > 1 - \xi_i \implies \alpha_i = 0$
- bias term $\beta_0$ computed with the average (for numerical stability) of all support vectors $k$ where $\xi_i = 0$ that satisfy $g_i ({x}_i^\top {\beta} + \beta_0) = 1$
- all points except support vectors can be removed without changing the solution, because $\alpha_i \text{=} 0$ don't influence $\sum_{i=1}^n \alpha_i g_i {x}_i \text{=} {\beta}$. (in linear discriminant analysis, by contrast, all points have influence on the decision rule through the mean vectors and covariance matrices)

*non-linear svm*

- $H(x) = \sum_{m = 1}^M \alpha_m h_m (x)$
- $h_m(x)$
	- basis expansions (usually polynomials, splines)
	- increasing dimensionality from $p$ to $M$, where linear separation is possible
	- $x = (x_1,~\ldots,~ x_p)$ = features
	- $h(x) = (h_1(x),~\ldots,~ h_M(x))^\top$ = transformed features
- $\hat f(x) = \hat \beta_0 + h(x)^\top \hat \beta$
	- hyperplane in $h(x)$
- $\hat G(x) = \text{sgn}(\hat f(x))$
	- not need to normalize by $\|\beta\|$ because we just need the sign, not the geometric distance

solution to $\max_{\beta_0, \beta} M$:

- i) define all steps using the dot product of the transformed features $\langle h({x}_i), h({x}_j)\rangle$
	- $L_d = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j g_i g_j \langle h({x}_i), h({x}_j)\rangle$
	- $f(x) = \sum_{i=1}^n \alpha_i g_i \langle h({x_i}), h({x}_i)\rangle + \beta_0$
	- based on constraint $\sum_{i=1}^n \alpha_i g_i {x}_i = {\beta}$
- ii) kernel trick:
	- $K(u, v) = \langle h(u), h(v)\rangle$
	- $K: \mathbb R^p \times \mathbb R^p \mapsto \mathbb R$
	- use kernel functions instead of basis expansions: compute the relationship between each pair of points (dot product) as if they are in the higher dimension, without actually transforming $h(x)$ the entire feature space
	- kernel types:
		- a) linear kernel: $K(u, v) = u^\top v$
		- b) $d$th-degree polynomial kernel: $K(u, v) = (c_0 + \gamma \langle u, v \rangle)^d$
		- c) radial basis function RBF kernel: $K(u, v) = \text{exp}(- \gamma \cdot \| u-v \|^2)$
		- d) sigmoid kernel: $K(u, v) = \text{tanh}(\gamma \langle u, v \rangle + c_0)$
		- everywhere $\gamma > 0$ and $c_o$ is a hyperparam
	- hyperparam search:
		- large $C$ penalizes missclassifications (more heavily than in linear case), makes rough boundaries, reduces margin width, might overfit
