# least squares regression

*linear regression model*

- $\hat y_i = \hat\beta_0+\sum_{j=1}^px_j \hat\beta_j$
- $\varepsilon_i = y_i- \hat y_i$
- linear combination
- $\mathbb E(y | x_i)$ follows a straight line

*ordinary least squares OLS*

- $\text{RSS}(\hat\beta)=\sum_{i=1}^n(y_i- \hat y_i)^2$ (residual sum of squares)
- sensitive to outliers
- normal equation:
	- same as setting the first derivative to zero
	- ${\hat y} =  X  \cdot {\hat \beta}$
	- ${\hat y} =  X \cdot ({X}^\top{X})^{-1}{X}^\top  \cdot {y}$
	- ${\hat y} = {H} \cdot {y}$ (hat matrix)
- multicollinearity:
	- when features are correlated, $X$ is not invertible → $X$ is not full-rank, $X^\top X$ is singular, no unique solution, unreliable statistical inference (because of its diagonals $d_j$ used in ie. z-test)
	- address with: feature selection, regularization, other models

*OLS inference assumption*

- $\varepsilon_i \sim N_n(0,~\sigma^2)$
- $\varepsilon$ are i.i.d. = independent (from each other), identically (normal) distributed random variables (given the same $X$)
- where $\varepsilon, y$ are random vars, $X$ is a fixed var
- $\hat{ \beta}, \hat \sigma^2$ are statistically independent
	- $\hat{\sigma}^{2} = \frac{\text{RSS}}{n-p-1}$ = unbiased variance estimator → this allows us to do t-tests
	- $\hat{{\beta}}\sim N_{p+1}({\beta},~\sigma^2({X}^\top{X})^{-1})$ follows a multivariate normal distribution
	- $(n\text{–}p\text{–}1)\cdot\hat{\sigma}^2\sim\sigma^2 \cdot \chi_{n-p-1}^2$ follows a scaled chi-squared distribution

*gauss-markov theorem*

- OLS is the best, linear, unbiased estimator (BLUE) under the inference assumption
- "best" means lowest variability
- "unbiased" means $\mathbb E(\hat{{\beta}})={\beta}$

# statistical inference

*t-test*

- = tests if observed mean significantly different from population mean
- assumes $\sigma$ is unknown, $n<30$
-  $H_0: \beta_j = 0$ and $H_1: \beta_j \not = 0$ (population $\mu$ is 0 due to inference assumption)
- ${t_j=\frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{d_j}}}$
- $d_j$ = a diagonal elem of $({X}^\top{X})^{-1}$
- $t_j \sim t_{n-p-1}$ follows t-distribution under null hypothesis
- $F = z_j^2$ when testing a single parameter, the f-statistic and the square of the z- or t-statistic are equal

*z-test*

- = tests if observed mean is significantly different from population mean
- assumes $\sigma$ is known, $n\geq30$
- ${z_j=\frac{\hat{\beta}_j}{{\sigma}\sqrt{d_j}}}$
- $z_j \sim N(0,1)$ follows standard normal distribution

*confidence intervals*

- confidence interval = $\hat{\beta}_j \pm 2 \cdot (\hat{\sigma}\sqrt{d_j})$
- ie. with 95% confidence, the true value of $\beta_j$ lies within 2 standard deviations from the mean 0
- critical values = values from which hypothesis is rejected

*decomposition of variance*

- the spread around the mean can be decomposed into 2 components → this allows us to do f-tests
- $(y_i - \bar y) = (\hat y_i - \bar y) + (y_i - \hat y_i)$
	- $\hat y_i$ = prediction, $y_i$ = actual, $\bar y_i$ = mean of actual
- $\text{TSS = RegSS + RSS}$
	- $\text{TSS} = \sum_{i} (y_i - \bar y)^2$ = total sum of squares, total variation, is $\sigma^2$ when divided by $n \text{-} 1$
	- $\text{RegSS} = \sum_{i} (\hat y_i - \bar y)^2$ = regression sum of squares, explained component, captured by regression
	- $\text{RSS} = \sum_{i} (y_i - \hat y_i)^2$ = residual sum of squares, unexplained component, residuals

*f-test*

- family of statistical tests (ie. comparing coefficient significance, model fits)
- $F_{p_1 - p_0, n - p_1 - 1} \approx \chi^2_{p_1 - p_0}$ approximated with chi-squared for large samples under null-hypothesis

*anova f-test*

- = tests if any of the coefficients are significant
- $H_0: \beta_1 = \ldots = \beta_p = 0$
- $H_1: \exists j: \beta_j \not=0$
- $F = \frac{\text{RegSS}/p}{\text{RSS}/(n-p-1)}$
	- $F \sim F_{p,n-p-1}$ under inference assumption, null hypothesis
	- between group variance = avg explained variance per predictor
	- within group variance = avg unexplained variance per residual degree of freedom
	- the mean sum of squares (meanss) is calculated by dividing it by its corresponding degrees of freedom (df)

# feature selection

to simplify model, drop correlated features, handle $p > n$

*extra-sum-of-squares f-test*

- = tests if more coefficients significantly improve model
- $H_0$: smaller is good enough
- $H_1$: larger model is better $p_0 < p_1 \leq p$
- nested model = smaller model uses a coefficient subset of the larger model
- $F = \frac{(\text{RSS}_0 - \text{RSS}_1)/(p_1 - p_0)}{\text{RSS}_1/(n - p_1 - 1)} \sim F_{p_1 - p_0, n - p_1 - 1}$
- measures delta in residuals per additional parameter, normalized by an estimate of $\sigma ^2$

*r-squared (coefficient of determination)*

- = performance metric, ratio of explained variance
- $R^2 = \frac{\text{RegSS}}{\text{TSS}} = \text{Cor}^2(y, \hat y) \in [0,1]$ (r-squared)
- $\tilde R^2 = 1 - \frac{\text{RSS} / (n-p-1)}{\text{TSS} / (n-1)}$ (adjusted r-squared, also penalizes model size)

*information criteria*

- = performance metric, also penalizing model size
- akaike's information criterion:
	- a) $\sigma^2$ is unknown: $\text{AIC} = n \cdot \log(\text{RSS}/n) + 2p$
	- b) $\sigma^2$ is known: $\text{AIC} = \text{RSS}/\sigma^2 + 2p$
	- $\sigma^2$ = residual variance
	- $\hat{\sigma}^2 = \text{RSS}/n$ (estimate if unknown)
	- best for descriptive models, absolute value can't be interpreted
	- based on kullback-leibler information KL: measuring error in approximation (difference between two distributions)
	- likelihood of correct prediction $L$ (joint probability of observing the data under the assumed model) in original formula replaced with normal distribution because of inference assumption
- bayes information criterion:
	- a) $\sigma^2$ is unknown: $\text{BIC} = \text{-}2 \cdot \max \log L + \log(n) \cdot p$
	- b) $\sigma^2$ is known: $\text{BIC} = \text{RSS}/\sigma^2 + \log(n) \cdot p$
	- favors simpler models than AIC for large $n$, best for predictive models, absolute value can't be interpreted
- mallow's cp:
	- $C_p = \text{RSS}/ \sigma^2 + 2p - n$
	- estimate $\sigma^2$ if unknown
	- mainly used for stepwise regression (iteratively adding/removing features)

*resampling methods*

- estimating error through sampling
- needs metric (ie. mean squared error MSE)
- cross validation:
	- i) split data in $q$ parts (folds)
	- ii) use $\frac{1}{q}$ testing, $\frac{q-1}{q}$ for training
	- iii) repeat $q$ times
	- $\text{Err} = \frac{1}{n} \sum_i \text{Loss}_i$
	- samples occur exactly once at evaluation
	- leave-1-out-cv = $q$ only has 1 element, too compute intensive, high variance
- bootstrap:
	- i) bootstrap samples = create train set by sampling with replacement (usually as large as full dataset $n$)
	- ii) out-of-bag samples = create test set by using leftovers, not used in train set
	- ii) repeat $q$ times
	- $\text{Err} = \frac{1}{q} \sum_i \frac{1}{|\text{OOB}|} \sum_j \text{Loss}_{i,j}$
	- statistically $|\text{OOB}| = (1-\frac{1}{n})^n \approx \frac{1}{3}$
	- samples can be overrepresented at evaluation, too optimistic due to data-leaks
	- leave-1-out-bootstrap = avoids data leaks. for every elem in full dataset $x_i$ only use models that weren't trained on it $D \textbackslash x_i = C^{-i}$. this means $\text{Err}=\frac{1}{n} \sum_{i} \frac{1}{\left|C^{-i}\right|} \sum_{j} \text{Loss}_{i,j}$

*variable selection*

- exhaustive feature search would take too long $2^p$
- backward stepwise selection = full model, recursively drop least useful params, use f-statistic/info criteria
- forward stepwise selection = empty model, recursively add most useful parameters, use f-statistic/info criteria
- best subset regression = use info criteria
- leaps and bounds algorithm = drop search branches (entire parameter subsets), based on info criteria

# dimensionality reduction

aka. feature transformation, using derived inputs as regressors

*digression: decomposition of mean squared error*

- bias-variance tradeoff, reducing variance at the cost of a bit of bias
- $\text{MSE} = \mathbb{E}[(y - \hat y)^2] = \underbracket{\sigma^2} + \underbracket{\text{Var}(\hat y)} + \underbracket{\left (\mathbb{E}[y - \hat y]\right )^2}$
	- $\sigma^2$ = irreducible error, error independent of model (ie. measurement error)
	- $\text{Var}(\hat y)$ = variance of estimates, depends on train set, model instability
	- $\mathbb{E}[y - \hat y]^2$ = squared bias

*digression: spectral decomposition (eigendecomposition)*

- = factorize a square matrix into eigenvectors and eigenvalues
- eigenvector = non-zero vector that has its direction unchanged by the given linear transformation / matrix
- input must be square, symmetric ($S = S^\top$), diagonalizable (as many linearly independent eigenvectors as it has dimensions)
- $S = V A V^{\top}$
	- $S$ = input matrix
	- $V$ = eigenvector columns, normalized, orthogonal ($V^\top = V^{-1}$) matrix
	- $A$ = eigenvalue diagonal elements, which are positive if $S$ is positive definite

*digression: principal component analysis PCA*

- = explaining data in a simpler coordinate system + dimensionality reduction, decorrelation
- preprocess data first: scale (ie. to normal distribution using z-score), center (subtract the mean)
- $Z = XV$
	- intuition: projecting onto directions of largest spread minimizes information loss
	- $Z$ = transformed data, columns are principal components PCs (uncorrelated, have maximum variance) ordered by variance
	- $X$ = input data
	- $V$ = transformation matrix → we want to find this
- $\text{Cov}(X) = V A V^{\top}$
	- orthogonal axes aligned with directions of largest spread, found through eigendecomposition of covariance matrix
	- $\text{Cov}(X)$ = represents the spread, mean variance between vector pair $X_i,~ X_j$ at index $i,j$
	- $V$ = eigenvector columns, normalized, orthogonal ($V^\top = V^{-1}$) matrix
	- $A$ = eigenvalue (= variance) diagonal elements, sorted in descending order
- $\text{Cov}(Z) = A$
	- validating that PCs are uncorrelated: their covariance matrix $\text{Cov}(Z)$ is a diagonal matrix
	- $\text{Cov}(Z) = \text{Cov}(XV) =  V^\top \text{Cov}(X) V = V^\top (V \Lambda V^\top) V = (V^\top V) \Lambda (V^\top V)= A$
	- because for any linear transformation $\text{Cov}(AB) = A \text{Cov}(B) A^\top$

*principal component regression PCR*

- unsupervised method: only considers variance of $X$
- using first $q <p$ principal components for regression
- $y = X \beta + \varepsilon$
- $y = (X V) (V^\top \beta) + \varepsilon$
	- because $VV^\top = I$
- $y = Z \cdot \theta + \varepsilon$
	- defining transformed coefficients $V^\top \beta = \theta$
- $y=Z_{1:q}\theta_{1:q}+Z_{q+1:p}\theta_{q+1:p}+\varepsilon$
	- split coefficients at index $q$
- $y=Z_{1:q}\theta_{1:q}+ \tilde\varepsilon$
	- drop the second term and update error
	- this step addresses multicollinearity
- $\hat \theta = (Z^\top_{1:q} Z_{1:q})^{-1} Z^\top_{1:q} y$
	- fit model, estimate weights $\hat \theta$ with normal equation
- $\tilde{\hat \beta} = V_{1:q} \hat \theta_{1:q}$
	- to interpret results in terms of original variables we can apply $V$ (to undo $V^\top$)
	- $\tilde{\hat \beta} \not = \hat \beta$ due to information loss

*partial least squares PLS regression*

- supervised method: maximizing covariance between $X, y$
- preprocess data first: scale, center
- $T = XW$
	- $T$ = score matrix
	- $X$ = input data
	- $W$ = loading matrix
- ${w}_{k}=\text{argmax}_{{{w}}}\text{Cov}({y},{X}{w})$
	- loading vectors $w_i$ are computed iteratively with 2 constraints
	- $\|{w}_k\|=1$ (normalized)
	- $\text{Cov}(X{w}_k,X{w}_j)=0,\forall j<k$ (subsequent vectors orthogonal to previous  ones)
- $y = T \gamma + \tilde{\varepsilon}$
	- latent variable model (can't be observed directly)
	- $\gamma$ in lower dimension $q < p$
- $y = X \underbracket{W \gamma} + \tilde{\varepsilon}$
	- $W \gamma$ is similar to original $\beta$

*continuum regression*

- $w_k = \text{argmax}_w \{[\text{Cov}(y, Xw)]^2 \cdot [\text{Var}(Xw)]^{\frac{{\delta}}{1-\delta}-1}\}$
- iteratively computes $w_i$ like in PLS but $\delta \in [0;1]$ in ${\frac{{\delta}}{1-\delta}\text{–}1}$ balances covariance (PLS like) and variance (PCR like)
- $\delta = 0$ means OLS, exponent becomes -1
- $\delta = 0.5$ means PLS, exponent becomes 0
- $\delta = 1$ means PCA, exponent becomes $\infty$

# regularization

smoothing out / removing coefficients to reduce variance

*ridge regression*

- OLS with L2-norm penalty, coefficients never reach zero
- $\hat{\beta}_{\text{Ridge}} = {\text{argmin}}_\beta \{\sum_{i} (y_i -\hat y_i)^2 + \underbracket{\lambda\sum_{j=1}^p \beta_j^2} \}$
- $\hat\beta_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$
- alternatively: $\sum \beta_j^2 \leq s$
- $\lambda$ = regularization strength
	- determines degrees of freedom
	- $df$ go from $p$ to 0 as you increase $\lambda$
	- $df = \text{trace}(X(X^TX + \lambda I)^{-1}X^T)$ sum of diagonal elems
- preprocess data first: scale, center, set intercept to $\beta_0 = \bar y$ and drop it from model
- if inputs are orthogonal then $\hat{\beta}_{\text{Ridge}} = \hat \beta / (1 + \lambda)$
- adds $\lambda I$ to $X^\top X$ to make it invertible

*lasso regression*

- OLS with L1-norm penalty, coefficients reach zero
- $\hat{\beta}_{\text{Lasso}} = {\text{argmin}}_\beta \{\sum_{i} (y_i -\hat y_i)^2 + \underbracket{\lambda\sum_{j=1}^p |\beta_j|}\}$
- alternatively: $\sum | \beta_j | \leq s$
- $\lambda$ = regularization strength
- preprocess data first: scale, center, set intercept to $\beta_0 = \bar y$ and drop it from model
- no closed-form solution like ridge

*digression: single value decomposition SVD*

- = factorize any matrix into eigenvectors and singular values
- $X = U D V^\top$
- $U, V$ = eigenvectors / left/right-singular vectors, normalized, orthogonal, in descending order based on $\sigma$
- $D$ = singular values $\sigma$ / eigenvalue roots $\sqrt d_i$ of $U, V$ as diagonal elements
	- singular values are also the number of non-zero values is the rank of the matrix

application: PCA

- $Z = X \cdot V = UDV^\top \cdot V = UD$
	- $V$ are eigenvectors of $X^\top X$ (can be also found with eigendecomposition)
- $\text{Cov}(X) = \frac{1}{n-1} X^\top X = \frac{1}{n-1} VD^2 V^\top$
- $\text{Var}(Z) = \text{Var}(UD) = \frac{1}{n-1} UD^\top UD = D^2$

application: OLS

- ${\hat y} = X \cdot ({X}^\top{X})^{-1}{X}^\top  \cdot {y}$
- ${\hat y} = (UDV^\top) \cdot ({(UDV^\top)}^\top{(UDV^\top)})^{-1}{(UDV^\top)}^\top  \cdot {y}$
- ${\hat y} = UU^\top {y}$
- where $U^\top y$ are coordinates of $y$ with respect to the orthonormal basis $U$
- then $\hat y$ is reconstructed by multiplying with $U$

application: ridge regression

- $\hat\beta_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$
- $\hat y = X \cdot \hat \beta_{\text{Ridge}} = (UDV^\top) \cdot ((UDV^\top)^\top (UDV^\top) + \lambda I)^{-1} (UDV^\top)^\top y$ 
- $\hat y = X \cdot \hat \beta_{\text{Ridge}} = UD (D^2 + \lambda I)^{-1} DU^\top y$ 
- $\hat y = X \cdot \hat \beta_{\text{Ridge}} = \sum_{j=1}^p u_j {\frac{d_j^2}{d_j^2 + \lambda}} u_j^\top y$
- $\frac{d_j^2}{d_j^2+\lambda}$ = shrinkage factor
    - shrinks with larger $\lambda$
    - shrinks directions with smaller $d_j^2$ (proportional to variance of the $j$-th principal component $z_j$ of $X$, left/right-singular vectors eigenvalue)
- shrinkage depends solely on input variance $d_j^2$, not on relationship to $y$, which means:
	- high-importance, low-variance features may get over-shrunken
	- low-importance, high-variance features may get under-shrunken
	- because it implicitly assumes principal components with higher variance $d_j^2$ are always more relevant for predicting $y$
- similarity to PCA:
	- both use principal components
	- PCR discards low-variance PCs (hard threshold) but ridge smoothly downweights them
	- neither consider correlation with $y$ (unsupervised)

# linear classification

*linear regression of indicator matrices*

- = linear boundaries just like LDA, without assuming gaussian classes with shared covariance
- $k \in \mathcal G = \{1, 2, \ldots, K\}$
	- integers from $1$ to $K$ as classes, one-hot-encoded
- $\hat y_k \in \mathbb R =\hat f_k(x) ~~= \hat \beta_{k0} + \sum_{j=1}^p \hat \beta_{kj} x_j ~~= \hat \beta_{k0} + \beta_{kj}^\top x$
	- continuous scores for each class, not probabilities (unlike logistic regression)
	- hard to interpret, because they are based on the encoding
- $\hat Y = X \hat B$
	- solves $K$ independent regressions in each column
	- $Y = [y_1, \ldots, y_K ]$ = indicator matrix (one-hot-encoded)
	- $\hat Y = [\hat y_1, \ldots, \hat y_K ]$ = fitted scores for each class
	- $\hat B = (X^\top X)^{-1}X^\top Y = [\hat \beta_1, \ldots, \hat \beta_K]$ = coefficients for each class
- $\{ x: (\hat{\beta}_{k0} - \hat{\beta}_{l0}) + (\hat{\beta}_k - \hat{\beta}_l)^\top x = 0 \}$
	- hyperplane in $\mathbb R^p$
	- the decision boundary between two classes $k,l$ is the set of all data points where $\hat f_k(x) = \hat f_l(x)$
- $\hat G(x) = \text{argmax}_{k\in \mathcal G} \hat f_k (x)$
	- classification is based on the largest score (due to the binary encoding)
	- prone to overlapping decision boundaries (masking) in multiclass problems

*linear discriminant analysis LDA*

- conditional probability:
	- probability of random variable $G$ being in class $k$
	- $\begin{gather}P(G{=}k \mid {x})=\frac{h_k({x}) \cdot \pi_k}{\sum_{l=1}^K h_l({x}) \cdot \pi_l}\end{gather}$
	- $\pi_k$ = how common class $k$ is overall, where $\Sigma_{k=1}^K \pi_k = 1$ (prior, discrete function)
	- $h_k(x)$ = how common $x$ is in $k$ (posterior, density function)
	- $\sum_{l=1}^K h_l(\text{x}) \cdot \pi_l$ = how common $x$ is overall
- assumptions:
	- i) $\Sigma_k = \Sigma$ (shared covariance matrix)
		- so decision boundaries are linear
	- ii) $h_k(x) \sim \varphi_k(x)$ follows density of a multivariate normal distribution
		- gaussian mixture model = weighted $\pi_k$ sum of gaussian distributions $\varphi(x)$
		- $\varphi_k(x) = \frac{1}{\sqrt{(2\pi)^p \cdot |\Sigma_k|}} \exp\{-\frac{1}{2}(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\}$
		- $\mu_k$ = class mean
- decision rule:
	- intuition: any value on the decision boundary should have equal probability of being in either group
	- compare logarithms of odds (logits) for numerical stability
	- $\log \left (\frac{P(G=k|x)}{P(G=l|x)} \right) = \log \left(\frac{\varphi_k(x)\pi_k}{\varphi_l(x)\pi_l} \right)  = \log(1) = 0$
	- $\log \left (\frac{P(G=k|x)}{P(G=l|x)} \right) = \ldots = \delta_k(x) - \delta_l(x)$
	-  $\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k +  \log\pi_k$ = linear discriminant function
		- linear in $x$ because it follows the form of a linear function $ax + b$
	-  $G(x) = \text{argmax}_k \delta_k(x)$
- estimations from training data:
	- in practice the distributions are unknown, so we estimate $\hat G(x)$
	- $\hat \pi_k = \frac{n_k}{n}$ (class proportion)
	- $\hat \mu_k = \sum_{g_i = k} \frac{x_i}{n_k}$ (class mean)
	- $\hat{{\Sigma}}=\frac{1}{n-K} \sum_{k=1}^K \sum_{g_i=k}({x}_i-\hat{{\mu}}_k)({x}_i-\hat{{\mu}}_k)^{\top}$ (pooled covariance)
	- $n_k$ = number of data points per class
	- $g_i$ = true group number of data point

*quadratic discriminant analysis QDA*

- $\delta_k({x})=-\frac{1}{2} \log |{\Sigma}_k|-\frac{1}{2}({x}-{\mu}_k)^{\top} {\Sigma}_k^{-1}({x}-{\mu}_k)+\log \pi_k$
- different discriminant function
- doesn't assume shared cov matrix (but multivariate normal distribution) → quadratic decision boundaries
- needs more parameters ($\Sigma_k$ for each class) than LDA
- both LDA and QDA work well, even if classes aren't not normally distributed or covariance matrices aren't equal

*regularized discriminant analysis RDA*

- $\hat{{\Sigma}}_k(\alpha)=\alpha \hat{{\Sigma}}_k+(1-\alpha) \hat{{\Sigma}}$
	- $\hat{{\Sigma}}=\frac{1}{\sum_{k=1}^K n_k}(\sum_{k=1}^K n_k \hat{{\Sigma}}_{{k}})$
	- $\alpha \in [0;1]$ - ranging from LDA to QDA
- regularized pooled covariance matrix
- allows to shrink the separate covariances of QDA towards a common covariance as in LDA

*logistic regression*

- used for infrence, can determine feature importance, no assumptions about distributions like LDA
- multiclass classification:
	- log-odds of $x$ being in any class $k$ vs. some arbitrary reference class $K$ (doesn't matter which one) is modeled with a linear combination
	- here we chose $K$ as the reference class
	- $\log \left(\frac{P(G = k|x)}{P(G = K|x)}\right) = \beta_{k,0} + \beta_{k,1} \cdot x$ where $k \not = K$ 
	- $\sum_k P(G = k \mid x) = 1$ (odds should sum up to 1)
	- … convert raw log-odds into probability distribution, satisfy constraint with a softmax/logistic function
	- $P(G \text{=} K\mid x) = \frac{1}{1 + \sum_{l=1}^{K-1} \exp\{\beta_{l,0} + \beta_{l,1} \cdot x\}}$ (reference class)
	- $P(G \text{=} k \mid x) = \frac{\exp\{\beta_{k,0} + \beta_k \cdot x\}}{1 + \sum_{l=1}^{K-1} \exp\{\beta_{l,0} + \beta_{l,1} \cdot x\}}$ (other classes)
- binary classification:
	- here we chose $2$ as the reference class
	- $P(G \text{=} 2 \mid x) = \frac{1}{1 + \exp\{\beta_0 + \beta_1 \cdot x\}} = p_2(x)$ (reference class)
	- $P(G \text{=} 1 \mid x) = \frac{\exp\{\beta_0 + \beta_1 \cdot  x\}}{1 + \exp\{\beta_0 + \beta_1 \cdot x\}} = p_1(x)$ (other classes)
	- $p_1(x) + p_2(x) = 1$
	- common cutoff to classify is 0.5
- next we have to estimate the coefficients $\beta$

log-likelihood function (for binary classification):

- = metric to quantify how well $\beta$ explains the observed data
- joint likelihood = probability of observing all data points
- logarithm converts the product into a sum, simplifies differentiation
- $L(\beta) = \prod_{i=1}^n p_{g_i} (x_i; \beta)$
- $l(\beta) = \log(L(\beta))$
- $l({\beta})=\sum_{i=1}^{n} \log \left( {p_{g_{i}}} ({x}_{i} ; {\beta})\right)$
	- $p_{g_{i}}$ = probability that observation $x_i$​ belongs to its actual class $g_i \in \{1, 2\}$
	- $\beta$ includes intercept
	- we use $\beta$ in the notation to show that it still needs to be estimated
- $l({\beta})=\sum_{i=1}^{n}\{ \underbracket{ {y_i} \cdot \log [p({x}_{i} ; {\beta})]} + \underbracket{{(1-y_i)} \cdot \log [1-p({x}_{i} ; {\beta})]} \}$
	- we can get rid of $g_i$ by just using a boolean flag ${y_i} \in \{0,1\}$ to enable/disable terms
	- $y_i \text{=} 0$ or $y_1 \text{=} 1$ means we want to know the probability of $g_i \text{=} 2$ or $g_i \text{=} 1$
	- we set $p(x_i; \beta) = p_1(x)$ and derive $p_2$ through $p_2(x) = 1 - p_1(x)$
- … after some algebra
- $l({\beta})=\sum_{i=1}^{n}\{ y_i \cdot {\beta}^{\top} {x}_{i}-\log [1+\exp ({\beta}^{\top} {x}_{i})]\}$

maximum likelihood estimation MLE (for binary classification):

- = maximize the log-likelihood $l(\beta)$ to estimate the coefficients $\beta$
- $\beta \text{=} 0$ is a good starting value
- convergence not guaranteed
- first derivative:
	- we take the first derivative (with respect to $\beta$) and set it to zero
	- the resulting "score equations" are nonlinear and cannot be solved directly, so we need an algorithm
	- $\frac{\partial l({\beta})}{\partial {\beta}} = \sum_{i=1}^{n}[y_{i}-p({x}_{i} ; {\beta})] \cdot {x}_{i}=0$
	- $\frac{\partial l({\beta})}{\partial {\beta}} = X^\top (y - p)$
	- $p$ = vector of $p(x_i; \beta_{old})$
- second derivative:
	- we take the second derivative (hessian matrix)
	- $\frac{\partial^{2} l({\beta})}{\partial {\beta} \partial {\beta}^{\top}}=-\sum_{i=1}^{n} p({x}_{i} ; {\beta}) \cdot [1-p({x}_{i} ; {\beta})] \cdot {x}_{i} {x}_{i}^{\top}$
	- $\frac{\partial^2 l({\beta})}{\partial {\beta} \partial {\beta}^{\top}} = -X^\top W X$
	- $W$ = diagonal matrix with weights $p(x_i; \beta_{old}) \cdot (1 - p(x_i; \beta_{old}))$
	- $W$ isn't diagonal from $K \geq 3$ 
- newton-raphson / iteratively reweighted least squares IRLS algorithm:
	- ${\beta}_{\text {new }} \leftarrow {\beta}_{\text {old }}-\left(\frac{\partial^{2} l({\beta})}{\partial {\beta} \partial {\beta}^{\top}}\right)^{-1} \frac{\partial l({\beta})}{\partial {\beta}}$
	- ${\beta}_{\text {new }} \leftarrow \beta_{\text{old}} + (X^\top W X)^{-1} X^\top (y-p)$
	- for $K \geq 3$ we can express the "newton-raphson algorithm" as IRLS, where each iteration solves a weighted OLS problem
	- ${\beta}_{\text {new }} \leftarrow \underbracket{\left({X}^{\top} {W} {X}\right)^{-1} {X}^{\top} {W}} \cdot \underbracket{z}$
		- the first term is a weighted OLS problem 
		- ${z} = {X}{\beta}_{\text{old}} + {W}^{-1}({y}-{p})$ = adjusted response
	- same as: ${\beta}_{\text {new }} \leftarrow {\text{argmin}}_{{\beta}}({z}-{X} {\beta})^{\top} \cdot {W} \cdot ({z}-{X} {\beta})$

# splines

non linear method

*basis expansions*

- = linear combination of basis functions
- $h_m(x) : \mathbb R^p \mapsto \mathbb R$
- $f(x) = \sum_{m = 1}^M \beta_m h_m(x)$
- ie. quadratic, nonlinear, indicator function $I$ (constant based on a boolean condition), …
- $M$ can be larger than $p$ (ie. polynomial terms, interactions $x_j \cdot x_k$, …)

*piecewise polynomials*

- = specifying a function through independent intervals (piecewise functions)
- idea: instead of a single high-degree polynomial, fit multiple low-degree polynomials to intervals of the feature
- assume $x$ has a single feature (univariate)
- use interval boundaries $\xi_i$ (knots) to split feature value range into $k\text{+} 1$ disjoint intervals: $(- \infty, \xi_1),~ [\xi_1, \xi_2),~ \ldots [\xi_{k}, \infty)$
- order = num coefficients = polynom degree + 1
- estimating coefficients:
	- in each interval, fit polynomials of order $M$
	- using $M (k \text{+} 1)$ coefficients (without continuity constraints)
	- example: for a piecewise constant, take average of interval: $\hat \beta_m = \bar y_m$
- continuity constraint:
	- = continuity of function values and derivatives, up to order $M\text{-}2$
	- reduces coefficients by 1 per derivative level per interior knot
	- function continuity: means function values must match between intervals, removes abrupt jumps between intervals $f(\xi^-_i) = f(\xi^+_i)$ (superscript means left/right hand limit)

*splines*

- = piecewise polynomials with continuous derivatives up to order $M\text{-}2$
- ie. cubic splines have continuous first and second derivatives (but not necessarily a third one)
- $f(x) = \sum_{j=1}^M \beta_j x^{j-1} + \sum_{l=1}^k \beta_{M+l} (x - \xi_l)_+^{M-1}$
	- $j$ = iterates over the polynomial degrees, $l$ = iterates over the knots
	- $M\text{+}k$ coefficients
- $h_j(x) = x^{j-1}$ where $j \in [1;M]$
	- polynomial basis
- $h_{M+l}(x) = (x - \xi_l)_+^{M-1}$ where $l \in [1;k]$
	- truncated power basis (necessary for continuity)
	- unstable, in practice B-spline are used instead
	- $(x - \xi_l)_+ = \max(0,~ x - \xi_l)$
- hyper parameters:
	- order $M$ (usually 4)
	- number of knots
	- placement of knots (usually equidistant)
- natural boundary conditions:
	- = continuity constraint + linearity beyond outermost knots
	- require first and second derivative to be zero at both boundaries
	- reduces coefficients by 4 (2 constraints per boundary · 2 boundaries)

*cubic smoothing splines*

- = regularized splines
- no manual placement of knots
- $RSS(f, \lambda) = \underbracket{\sum_{i=1}^n \{y_i - f(x_i)\}^2} + \underbracket{\lambda \int f''(t)^2 dt}$
	- penalizes roughness: interval of $f''$, weighted by $λ$
	- $\lambda = 0$ - no smoothing, interpolates all points
	- $\lambda = \infty$ - maximum smoothing, forces $f$ to become a straight line
- $f(x) = \sum^n_{j=1} N_j(x) \cdot \theta_j$
	- = basis expansion with natural cubic splines
	- it can be proven that natural cubic splines, with knots placed at all data points $x_i$ can achieve minimal penalty (proof omitted)
	- the large number of knots isn't a problem, because we shrink them away
	- $\theta_j$ = coefficients (of natural cubic splines)
	- $N_j$ = basis functions (of natural cubic splines)
- $RSS(\theta, \lambda) = \underbracket{(y - N \theta)^\top (y - N \theta)} + \underbracket{\lambda \theta^\top \Omega_N \theta}$
	- simplifed loss function, now knowing that solution must be a natural cubic spline
	- $\Omega_N = \int N_j^{''}(t) \cdot  N_k^{''}(t) ~ dt$
- $\hat \theta = (N^\top N + \lambda \Omega_N)^{-1} \cdot N^\top y$
	- estimated coefficients
	- generalized form of ridge regression
- $\hat f_\lambda(x) = \sum^n_{j=1} N_j(x) \cdot  \underbracket{\hat \theta_j}$
- $\hat f_\lambda(x) = N \underbracket{(N^\top N + \lambda \Omega_N)^{-1} \cdot N^\top y}$

selecting hyperparameter $\lambda$ for estimated $\hat f_\lambda$:

- $S_\lambda = N (N^\top N + \lambda \Omega_N)^{-1} \cdot N^\top$
	- = smoother matrix
	- too expensive to compute
	- has full rank $n$
- $H_\xi = B_\xi (B_\xi^\top B_\xi)^{-1} B_\xi^\top$
	- = hat matrix
	- approximates $S_\lambda$ using a subset of its basis functions $M \ll n$
	- instead of regularization, it reduces dimensionality
	- needs manual selection of knots and basis subset
	- has reduced rank / trace / projection space $M$
- both are symmetric, positive semidefinite, linear combinations of $y_i$ (linear smoother)
- $df_\lambda = \text{trace}(S_\lambda) = \sum_i S_{ii}(\lambda) = M$
	- sum of diagonal elements are the "effective degrees of freedom" (number of coefficients after regularization)
- $\text{MSE}_{\text{LOO}}(\lambda) = \ldots = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat f_\lambda(x_i)}{1 - S_{ii}(\lambda)} \right)^2$
	- leave-1-out cross validation to find the optimal $\lambda$
	- computing diagonal values $S_{ii}(\lambda)$ individually can get expensive
- $\text{MSE}_{\text{GCV}}(\lambda) = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat f_\lambda(x_i)}{1 - \text{trace}(S_{\lambda})/n} \right)^2$
	- generalized cross-validation
	- approximates $S_{ii}(\lambda)$ with mean of the sum of diagonals (average trace)

# generalized additive models

non linear method

*digression: multiple linear regression models*

- $\hat Y_i = \hat\beta_0+\sum_{j=1}^p X_j \hat\beta_j$
- multivariable, variables are vectors

*digression: generalized linear models GLM*

- = linear regression + link function to handle non-normal data
- $\mathbb E (y \mid X) = \mu = g^{-1} (X \beta)$
- $g(\mu) = X \beta$
- linear regression assumes $\mathbb E (y \mid X)$ follows a straight line (linear input/output relationship)
- when that isn't the case, we can find a link function $g$ to make it linear again
- assumes $y$ follows a distribution of the exponential family
- there is always a well-defined "canonical" link function

*generalized additive models GAM*

- = GLM + linear combination of functions
- $g(\mu(x)) = \alpha + \sum_{j = 1}^p f_j(x_i)$
	- $g$ = link function
	- $\mu(x)$ = conditional expected value of $y$ given $X$
	- $f_j$ = unspecified nonlinear (but smooth) functions
- uses link function $g$, but it makes input/output relationship non-linear – then approximates it with a non-linear model
- assumes $y$ follows a distribution of the exponential family

parameter estimation:

- $RSS(\alpha, f_1, \ldots, f_p) = \underbracket{\sum_{i=1}^n \{y_i - (\alpha + \sum_{j = 1}^p f_j(x_i))\}^2} + \underbracket{ \sum_{j=1}^p \lambda_j \int f_j''(t_j)^2 dt_j}$
- seperate $\lambda_j$ defined for each $f_j$
- cubic splines, with knots placed at all data points $x_i$ can achieve minimal penalty (proof omitted)
- for a unique solution we assume:
	- $X$ is non-singular
	- if $\forall j: \sum_{i=0}^n f_j(x_{ij}) = 0$ then set the average as the intercept $\hat \alpha= \bar y$
- backfitting algorithm:
	- initialize $\alpha \leftarrow  \bar y$ and $\hat f_j \leftarrow  0$
	- iterate $j \in [1;p]$:
		- $\forall i: r_{ij} \leftarrow y_i - \hat{\alpha} - \sum_{k \neq j} \hat{f}_k(x_{ik})$
			- calculate all $i \in [1; n]$ partial residuals without using the $j$th model $\hat f_j$
			- isolate the contribution of the $j$th model by removing the effects of all other models and the intercept (this is for the loss function, so we subtract and don't add)
		- $\hat{f}_j \leftarrow S_j\left( r_{1j}, r_{2j}, \ldots, r_{nj}\right)$
			- fit a cubic smoothing spline $S_j$ to the partial residuals
		- $\hat{f}_j \leftarrow \hat{f}_j - \frac{1}{n}\sum_{i=1}^n \hat{f}_j(x_{ij})$
			- center the $j$th model $\hat f_j$
			- so it has a mean at zero and doesn't overlap with $\hat \alpha$
		- repeat until $\hat f_j$ convergence

# trees

non linear method

*trees*

- partition feature space in subsets that should be as homogenous as possible, then fit a simple model in each (ie. assigning a constant)
- algorithm must decide:
	- tree structure (we only consider binary trees)
	- partition features / variable
	- partition thresholds / point

*regression trees*

- $\hat f(x) = \sum_{m = 1}^M \hat c_m \cdot I(x \in R_m)$
	- $I(x \in R_m)$ = boolean flag, whether $x$ is in that region
	- return the sum of all constants $c_m$ assigned to regions $R_m$ that $x$ belongs to
- $\hat c_m = \frac{1}{n_m} \sum_{x_i \in R_m} y_i$
	- the optimal constant to assign to a region is its average
- partitioning:
	- $\min_{j,s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \right]$
	- the inner optimization problems are easy, you just take the average, the outer one is difficult
	- $j$ = partition feature index ($x_j$)
	- $s$ = partition threshold value
	- $R_1(j,s) = \{ x \mid x_j \leq s \} \quad R_2(j,s) = \{ x \mid x_j > s \}$
- tree pruning:
	- stop splitting if complexity value is too high
	- $c_\alpha(T) = \sum_{m = 1}^{|T|} RSS + \alpha \cdot |T|$ = cost complexity criterion
	- $\alpha$ = tuning parameter, 0 results in full tree, has unique solution
	- $|T|$ = num of leaf nodes

*classification trees*

- majority consensus of labels for each region:
	- $k(m) = \text{argmax}_k \hat p_{mk}$
	- $\hat p_{mk} = \frac{1}{n_m} \sum_{x_i \in R_m} I (y_i = k)$
	- proportion of label $k \in [1;K]$ in node
- node impurity metrics $Q_m(T)$:
	- misclassification error: $\frac{1}{n_m} \sum_{x_i \in R_m} I(y_i \neq k(m)) = 1 - \hat{p}_{mk(m)}$
	- gini index: $\sum_{k \neq k'} \hat{p}_{mk}\hat{p}_{mk'} = \sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk})$
	- entropy: $-\sum_{k=1}^K \hat{p}_{mk} \log \hat{p}_{mk}$
	- cross-entropy and gini index are differentiable, more sensitive to changes, therefore preferred

*random forests*

- few sampels can change entire tree structure, especially further up in the tree
- tree generation algorithm:
	- i) create bootstrap sample
	- ii) generate tree, using only a subset of all features ($\sqrt p$ for classification, $\frac{p}{3}$ for regression)
	- iii) evaluate tree:
		- $\pi_b = \sum_{m=1}^{|T_b|} n_m \cdot Q_m(T_b)$ = tree impurity
		- $\pi_{bj}$ = tree impurity after permutation of feature $x_j$ → permutation means randomly shuffling the labels $y_j$ between features so they loose their predictive power
		- $\delta_{bj} = \pi_{bj} - \pi_{b}$ =  variable importance → if permuting feature $x_j$ increased error, it means $x_j$ was originally useful
- random forest algorithm:
	- generate $b$ trees, read majority consensus/average
	- variable importance = $\hat \theta = \frac{1}{B} \sum_{b=1}^B \delta_{bj}$
	- performance = $\text{MSE}_{\text{OOB}} = \frac{1}{n} \sum_{i=1}^n (y_i - \bar y_i^{\text{OOB}} )$  where the second term is the consensus of all trees
	- explained variance = $1 - \text{MSE}_{\text{OOB}} / \hat \sigma^2_y$

# support vector machines

non linear method

add more dimensions to input space to make linear decision boundaries possible

*perceptron*

- input: linear combination of features, output: class
- decision boundary:
	- $f(x) = \beta_0 + \beta^\top x = 0$
		- all points that satisfy the equation are in the hyperplane $\mathcal L$
		- the hyperplane is defined by its orthogonal vector $\beta$ and it's shift from the origin $\beta_0$
		- which is also its gradient $f'({x}) = {\beta}$
	- $\beta^\top (x_1 - x_2) = 0$ for all $x_1, x_2 \in \mathcal L$
		- difference of any two points on the hyperplane is orthogonal to $\beta$
		- but a point being orthogonal to $\beta$ doesn't mean it's on the hyperplane (because the hyperplane equation includes an offset term $\beta_0$)
	- $\beta^\top x_0 = \text{–}\beta_0$ for all $x_0 \in \mathcal L$
		- follows from definition of $f(x)$
		- any point on the hyperplane must equal $\text{–}\beta_0$ (negative intercept) to maintain the constant distance from the origin that defines the hyperplane
- geometric distance to boundary:
	- $\text{dist}(x, \mathcal L) = \frac{1}{\|\beta\|} f(x)$
	- $\text{dist}(x, \mathcal L) = \frac{1}{\|\beta\|} ( \beta^\top x - \underbracket{\beta^\top x_0}) = \frac{1}{\|\beta\|} ( \beta^\top x + \underbracket{\beta_0}) = \frac{1}{\|\beta\|} f(x)$
	- i. define the vector from point ${x}_0$ (on the hyperplane) to $x$ as $({x} - {x}_0)$
	- ii. project (get length of projection through dot product) that vector onto the normal vector $\beta$
	- iii. scale the result into standard euclidean units
	- a positive distance means the vector is on the side pointed to by $\beta$
- stochastic gradient descent:
	- perceptron learning algorithm of rosenblatt
	- $\text{Loss}(\beta_0, \beta) = - \sum_{i \in \mathcal M} y_i \cdot f(x_i)$
		- $\mathcal M$ = set of misclassified points, where the sign of $f(x)$ is different than label $y_i$
		- the multiplication of the two terms (in case of a misclassification) is negative (increases the loss) and proportional to the distance of the misclassified points to the decision boundary
	- $\begin{pmatrix} \beta \\ \beta_0 \end{pmatrix} \leftarrow \begin{pmatrix} \beta \\ \beta_0 \end{pmatrix} + \rho \begin{pmatrix} y_ix_i \\ y_i \end{pmatrix}$
		- processes one observation at a time
		- $\rho$ = learning rate
		- $\frac{\partial D(\beta_0, {\beta})}{\partial {\beta}} = - \sum_{i\in\mathcal{M}} y_i{x}_i$ = derivative with respect to $\beta$
		- $\frac{\partial D(\beta_0, {\beta})}{\partial \beta_0} = - \sum_{i\in\mathcal{M}} y_i$ = derivative with respect to $\beta_0$
- problem: if data is linearly seperable, sgd will converge, but there are infinitely many possible solutions. but svm finds a unique and optimal solution
- the final solution depends on: the initial parameter values, the order of processing the observations, the number of iterations

*linearly seperable case*

- = there is an optimal decision boundary (simplified version of the non-linearly seperable case)
- $(x_i, g_i)$ = observations
- $g_i \in \{ -1, 1 \}$ = label
- $G(x) =  \text{sgn}(\frac{1}{\|\beta\|} ( \beta_0 + \beta^\top x))$
	- classifier, returns the sign of the distance to the hyperplane
	- remember $\text{dist}(x, \mathcal L) = \frac{1}{\|\beta\|} ( \beta^\top x + {\beta_0})$ 
	- $f(x) = \beta_0 + \beta^\top x = 0$

margin:

- $M = \min_i~ g_i \cdot \frac{1}{\|\beta\|} \cdot ( \beta_0 + \beta^\top x_i)$
- $g_i \cdot \frac{1}{\|\beta\|} ( \beta_0 + \beta^\top x_i) \geq M$
	- remember $\text{dist}(x, \mathcal L) = \frac{1}{\|\beta\|} ( \beta^\top x + {\beta_0})$
	- the margin is the minimum distance of all points to the decision boundary
	- the total margin width separating classes is $2 M = 2 / \| \beta \|$
- $\min_{\beta_0, \beta} \| \beta \|$
- $g_i \cdot ( \beta_0 + \beta^\top x_i) \geq 1$
	- define $M := \frac{1} {\|\beta \|}$ and multiply the inequality by $\| \beta \|$ to get rid of the fraction, so it can be solved
	- the margin size and the magnitude of beta are inversely proportional

solution:

- $\max_{\beta_0, \beta} M$ where $g_i \cdot ( \beta_0 + \beta^\top x_i) \geq 1$
	- convex optimisation problem: quadratic criterion, linear inequality constraints
	- use the lagrange method to solve it
- $L_p = \frac{1}{2}\|{\beta}\|^2 - \sum_{i=1}^n \alpha_i \cdot \left(g_i \cdot (\beta_0 + {x}_i^\top {\beta}) - 1 \right)$
	- lagrange primal function $L_p$ uses the objective function (what we want to minimize) with constraints multiplied by new variables $\alpha_i$ called lagrange multipliers
	- the magnitude (euclidian norm) of the vector $\beta$ is squared and scaled by 1/2 so the derivation is simpler and the multipliers are non-negative $a_i \geq 0$
	- minimizing the magnitude of beta corresponds to maximizing the margin
	- solve with respect to $\beta_0, \beta$
- $L_d = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_jg_ig_jx_i^\top x_j$
- $\alpha_i \geq 0$
	- lagrange dual function $L_d$ is a lower bound and much simpler (only dependent on $\alpha_i$) to solve
	- i. take derivatives of $L_p$ with respect to $\beta_0, \beta$ and set them to zero
	- ii. substitute back into the original equation
- for optimality $L_d$ also has to satisfy the karush-kuhn-tucker KKT conditions:
	- $\sum_{i=1}^n \alpha_i g_i {x}_i = {\beta}$
	- $\sum_{i=1}^n \alpha_i g_i = 0$
	- $\alpha_i \cdot \left( g_i ({x}_i^\top {\beta} + \beta_0) - 1 \right) = 0$
- the last KKT condition means that there are 2 kinds of points:
	- at least either $\alpha_i = 0$ or the term $g_i ({x}_i^\top {\beta} + \beta_0) = 1$ must be satisfied
	- a) support vectors (on the decision boundary): $\alpha_i > 0 \implies g_i({x}_i^\top {\beta} + \beta_0) = 1$
	- b) non support vectors: $g_i({x}_i^\top {\beta} + \beta_0) > 1 \implies \alpha_i = 0$
	- points inside the margin aren't possible, because we assumed linear seperability
- the bias term $\beta_0$ computed using the average (for numerical stability) of all support vectors $k$ such that $\beta_0 = g_k - {x}_k^\top {\beta}$
- the solution ${\beta}$ is a linear combination of support vectors, because $\alpha_i = 0$ don't influence $\sum_{i=1}^n \alpha_i g_i {x}_i = {\beta}$.
- this means that all points except support vectors can be removed without changing the solution. (in linear discriminant analysis, by contrast, all points have influence on the decision rule through the mean vectors and covariance matrices)

*non-linearly seperable case*

- = classes overlap, misclassfications are unavoidable

margin:

- $M = \min_i~ g_i \cdot \frac{1}{\|\beta\|} \cdot ( \beta_0 + \beta^\top x_i)$
- $g_i \cdot \frac{1}{\|\beta\|} \cdot ( \beta_0 + \beta^\top x_i) \geq M (1 - \xi_i)$
- $\sum_i \xi_i \leq \text{const}$
- $0 \leq \xi_i$
	- remember $\text{dist}(x, \mathcal L) = \frac{1}{\|\beta\|} ( \beta^\top x + {\beta_0})$
	- if the minimum distance of all points to the decision boundary (left term) is negative, there's a misclassification (and the right term is also negative)
	- slack variable $\xi_i$ = distance of violation relative to margin, with absolute value of $\xi_i \cdot M$
	- a) $\xi = 0$ (full margin - correctly classified, on or outside the margin)
	- b) $0 < \xi_i \leq 1$ (reduced margin - correctly classified, inside the margin)
	- c) $\xi > 1$ (negative margin - misclassified, on the wrong side of the decision boundary)
- $\min_{\beta_0, \beta} \| \beta \|$
- $g_i \cdot ( \beta_0 + \beta^\top x_i) \geq 1 - \xi_i$
- $\sum_i \xi_i \leq \text{const}$
- $0 \leq \xi_i$
	- simplified, so it can be solved
- $\min_{\beta,\beta_0} \left(\frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \xi_i\right)$
- $g_i \cdot ( \beta_0 + \beta^\top x_i) \geq 1 - \xi_i$
- $0 \leq \xi_i$
	- simplified, for easier derivation
	- $C \sum_{i=1}^n \xi_i$ is equivalent to $\sum_i \xi_i \leq \text{const}$
	- $C$ = cost / tuning parameter, decides margin width, set to $\infty$ in the linearly separable case
	- $C$ directly controls trade-off between margin width (magnitude of beta) and training errors

solution:

- $L_p = \frac{1}{2}\|{\beta}\|^2 - C \sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i \cdot \left(g_i \cdot (\beta_0 + {x}_i^\top {\beta}) - ( 1 - \xi_i) \right) - \sum_{i=1}^n \lambda_i \xi_i$
	- has to be minimzed with respect to $\beta, \xi$
	- $\alpha_i, \lambda_i, \xi_i$ have to be non-negative
- $L_d = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_jg_ig_jx_i^\top x_j$
- $0 \leq \alpha_i \leq C$
	- same function as for the linearly seperable case, but with a different condition
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
- the bias term $\beta_0$ computed using the average (for numerical stability) of all support vectors $k$ where $\xi_i = 0$ such that satisfy $g_i ({x}_i^\top {\beta} + \beta_0) = 1$
- the solution ${\beta}$ is a linear combination of support vectors, because $\alpha_i = 0$ don't influence $\sum_{i=1}^n \alpha_i g_i {x}_i = {\beta}$.

*non-linear svm*

- = mapping data into a higher-dimensional space where linear separation becomes possible
- $H(x) = \sum_{m = 1}^M \alpha_m h_m (x)$
	- linear basis expansion
- $h_m(x)$
	- basis expansion, increasing dimensionality from $p$ to $M$
	- typically polynomials and splines
	- $x = (x_1,~\ldots,~ x_p)$ = features
	- $h(x_i) = (h_1(x_i),~\ldots,~ h_M(x_i))^\top$ = transformed features
- $\hat f(x) = \hat \beta_0 + h(x)^\top \hat \beta$
	- estimated decision boundary, linear model in $h(x)$
- $\hat G(x) = \text{sgn}(\hat f(x))$
	- nonlinear classifier in the original feature space, even though it's linear in the transformed space
	- dividing by $\|\beta\|$ is not necessary here because we don't need a geometric distance, just the sign

solution:

- i) define same steps as before, using a dot product of the transformed features $\langle h({x}_i), h({x}_j)\rangle$
	- $L_d = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j g_i g_j \langle h({x}_i), h({x}_j)\rangle$
	- $f(x) = \sum_{i=1}^n \alpha_i g_i \langle h({x_i}), h({x}_i)\rangle + \beta_0$
		- $f(x) = \beta_0 + h(x)^\top \beta = \beta_0 + h(x)^\top \left ( \sum_{i=1}^n \alpha_i g_i h({x})_i \right) = \beta_0 + \sum_{i=1}^n \alpha_i g_i \langle h({x}), h({x}_i)\rangle$
		- using constraint $\sum_{i=1}^n \alpha_i g_i {x}_i = {\beta}$
- ii) apply kernel trick:
	- use kernel functions instead of basis expansions
	- kernel functions calculate the relationships between every pair of points (dot product) as if they are in the higher dimension, they don't actually do the transformation $h(x)$ of the entire feature space.
	- kernel function: $K(u, v) = \langle h(u), h(v)\rangle$
		- $K: \mathbb R^p \times \mathbb R^p \mapsto \mathbb R$
		- symmetric, positive, semi-definite function
	- kernel types:
		- a) linear kernel: $K(u, v) = u^\top v$
		- b) $d$th-degree polynomial kernel: $K(u, v) = (c_0 + \gamma \langle u, v \rangle)^d$
		- c) radial basis function RBF kernel: $K(u, v) = \text{exp}(- \gamma \cdot \| u-v \|^2)$
		- d) sigmoid kernel: $K(u, v) = \text{tanh}(\gamma \langle u, v \rangle + c_0)$
		- everywhere $\gamma > 0$ and $c_o$ is a constant that need to be chosen
- iii) hyperparam search:
	- use cross validation
	- $C$ is even more important than in nonlinear case
	- large $C$ penalizes missclassifications more heavily, makes less smooth decision boundaries, reduces margin width, might overfit
