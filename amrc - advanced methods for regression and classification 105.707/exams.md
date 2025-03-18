# unstructured list

unstructured list of all questions found on vowi, mattermost, discord, whatsapp, telegram.

- unknown:
	- aic, bic, f statistic, r squared, r adjusted
	- resampling methods (bootstrap, cv), error metrics (mse, mae)
	- linear regression, beta hat, problems if X isn't full rank
	- variable selection, stepwise selection, best subset selection
	- pcr (advantages, disadvantages, differences to pls), pls
	- ridge
	- the connection of pcr, ridge, lasso
	- l1 vs l2 criteria and how variable selection is poassible
	- indicatior classification, also multivariable
	- LDA, setting up a classification rule, decision boundary, bayesian criteria
	- QDA, difference to LDA
	- RDA, key ideas
	- logistic regression, main ideas, 2 group case, model and param estimation with beta zero, likelyhood problem, resulting weighted OLS
	- basis expansions, spline interpolation, solving the knot selection problem, criteria of smoothing splines, degrees of freedom
	- glm/gam, link function conditional expectation, criterias for gam, back fitting algorithm
	- trees criteria, prune, optimal tree size
	- svm
- 2019:
	- Trees: Regression trees, classification trees. General idea, criterion to minimized, for both cases. What measures of node impurity are available? How to avoid overfitting (pruning).
	- SVM: criterion for linearly separable, non-linearly separable case. Kernel trick, kernel functions.
	- GAM: for regression. How does the model look like, what functions mimimize criterion.
	- Mulitple regression model: Ordinary LS solution, how to arrive at it, what to do in near singularity of X^T X, R Ridge Regression, Lasso Regression, how does that look like, what is is different to OLS
	- Spline regression: Criterion to minimize (with penalization of curvature), what functions do minimize this (natural cubic splines)
	- Define PCR and weighted least squares. Compare these two methods Define SVM for the linearly seperable and for the inseperable case.
- 2019:
	- degrees of freedom for smoothing splines
	- trees: regression trees, classification trees, general idea, criterion to be minimized. available measures for node impurity, pruning to avoid overfitting
	- svm: criterion for linearly seperable case, non-linearly seperable case, kernel trick, kernel functions
	- gam: for regression, what the model looks like, which functions minimize criterion
	- multiple regression model, ordinary LS solution, what to do near singularity of X^T X
	- ridge regression, lasso regression, OLS
	- spline regression, criterions to minimize (with penalization of curve), natural cubic splines minimize this
- 2020:
	- SVM: linear seperable, non seperable. explain the optimization formula and how you get there for both, $L_p$, $L_d$, with constraints
	- Splines & Smooting Splines: Explain splines in general, i.e. knots, knot selection and df. Explain smoothing splines, optimization formula and general idea
	- Logistic Regression: Explain logistic regression with focus on the two group case. Explain regression trees with focus on the splitting criteria.
	- Explain Lasso and Ridge regression. Formulas, Lambda parameter, why does Lasso set coeficients exactly to 0, show it visually. The formula for beta^hat_ridge.
	- LDA,QDA,GDA: What is the idea of LDA. What is the Bayesian Theorem what are its assumptions and formula (the phi formula). Formula for LDA and GDA. What are the components of LDA and how do you estimate them. Why can u estimate them?
	- Random Forests: explain the Random Forest algorithm, bagging, selection of features, T_b, pi_bj, theta_bj
- 2020:
	- Smoothing Splines: What are splines, what is the criterion to optimize and the solution (formulas). How to determine Lambda/degrees of freedom. What does degrees of freedom mean in this context, why do we care about it?
	- SVM: Write down the optimization criteria with side constraints, how we get the solution (Lagrange primal, dual function) and KKT conditions for the linearly separable and inseparable case. Why do we need the KKT conditions? What is the Kernel trick? ← For this question I started with basis expansions, and that we put the basis functions in a vector h(x) = (h_1(x), ..., h_M(x)) and define our hyperplane with that. Then I explained that h(x) is only involved in the inner product in the solution and that we don't need to define the basis functions if we have a Kernel function that returns a real value. That seemed to be what Prof. Filzmoser wanted to hear.
	- Regression trees: Splitting criteria (formula), how to grow the tree and how long? Quality measure and cost complexity criterion (formula)
	- PCR: Model, how to get principal components (formulas), solution for the estimated coefficients beta_hat (formula). Why do we need PCR, in what use cases might this be helpful (singularity of X^TX)
- 2020:
	- linear Model , how to calc beta's? (hat matrix..), What problems can happen? (near-sigularities → expoding betas) What to do? i.E Shrinkage: → Explain Diff Ridge & Lasso (how do get the betas zero?)
	- SVM: separable case: how to solve that problem? what side contraints need to take care of, what is the kernel trick? (some talking about projections <h(u), h(v) > and its inner dot product.
- 2021:
	- know formulas 1.1, 2.1
	- F-test, comparing RSS for model selection / variable selection
	- r squared, fit measure, for fit of model to training data; adjusted r squared, possibility for model selection
	- Information criteria: derivation not important, but result. AIC vs BIC: different penalty, p vs log(n) p
	- Resampling: CV: 5-, 10-fold. Bootstrapping
	- Least Squares: Linear model, LS estimator, singularity of X^T X
	- Variable selection: AIC, BIC; forward, backward, both; best subset regression
	- PCR principal comp. regression. What and why?
	- Partial Least Squares. Involves also the response variable, unlike PCR!
	- Shrinkage Methods: when is Sum of beta squared large? In case of near singularity of X^T X, some absolute values of beta may be very large, we don't want that
	- formula for ridge
	- Lasso: shrinkage of |β|! No explicit formula for solution. Why do we get zeros for some of the βj (→ this aims at the figure with the norm balls for l1 norm, where LS solution is more likely to hit corners, where some β are zero)
	- Classification: Basic formulation using Bayesian theorem, what is posterior, prior prob., likelihood. Assumptions on distributions?
	- LDA vs QDA: assumptions on the covariance matrices; discriminant functions.
	- RDA: compare to LDA, QDA
	- Logistic Regression: formula: No distribution assumptions, we get a weighted LS problem (Newton-Raphson algorithm)
	- Nonlinear Methods: Splines, what is that? Definition on p.67 (bottom).
	- Problems/parameters: order of splines, number and placement of knots
	- Natural cubic splines, linear extrapolation outside the data
	- Smoothing splines; (7.1) Formula for RSS, penalization of curvature. How to select the degrees of freedom? What is the role of lambda
	- GAM, GLM: what is that? Formula for PRSS(...) = ....
	- Tree based methods: Regression. p 88: objective function
	- How to tune between under-, overfitting.
	- Classification: Gini, deviance measure?
	- Random Forests: what are they doing?
	- Bootstrapped data, out of bag data, variable importance?
	- SVM: objective functions, constraints
	- What is the kernel-trick?
- 2022:
	- Random Forest
	- Classification Tree
	- Smoothing splines, the formula for the PRSS (if lambda goes to 0 or infinity, how does it affect the spline), how f(x) is computed with splines (knots are at every data point), the newly adjusted matrix notation for RSS computation
	- GAM
	- SVM: Optimization problem in SVMs, for the linearly and non-linearly seperable case, under what constraints and how is it computed, what is the kernel trick and why is it useful in this case
- 2023:
	- logistic regression
	- random forest
	- lasso & ridge regression
	- pcr, svm
	- gam
- 2025 jan (exam prep lecture):
	- SVM: what is the optimization problem in separable and non separable case, What’s the outcome of beta? (13.2.1)
	- Regression trees: how to build, what splitting criteria (page 89), how split into two subregions - splitpoints, how to prune tree, sum of residual sum of squares of both regions
	- Smoothing splines: what is it? how can you find one? 7.1 formula for criterion, next formula under 7.1 for natural cubic splines, solution for f, how to avoid overfitting, degrees of freedom for each node (N in this case), trace of head matrix, formula with pruning param (fλ = N(N⊤N+ λΩN)−1N⊤y, everything on page 69/70, unique maximizer, hat Matrix and trace of hat Matrix (Page 70), Lambda tuning Paramete
	- ⁠regularization: difference between ridge and lasso regression, problems, argmin formulas (objective functions and understanding what's behind them), constraints, advantages disadvantages for both, for lasso we don't have a explicit formula von ß, lasso gives beta values that are equal to 0, variable selection, figure 3.1
- 2025 feb (first):
	- Aic, bic and differentiate variable selection
	- PCR and relationship with ridge regression
	- Random forest (only variable importance)
	- Gam (Mathematically explain how those functions are obtained, like backfitting algo)
- 2025 feb (second):
	- PLS
	- LS singularity problem, welches problem gibs beim statistical testing/inference
	- LDA what is the base
	- smooth Splines
	- SVM non seperable case, what has to be minimized

# sorted by topic

merged, deduplicated, and topic-sorted list of exam questions based on the lecture notes structure.

I. Fundamentals

- Model Selection & Evaluation
	- AIC vs BIC: Different penalties (p vs log(n)p)
	- Cross-validation (k-fold, LOOCV) vs Bootstrap methods
	- Mallows' Cp for stepwise regression
	- R² vs Adjusted R² interpretation

- Resampling & Error Metrics
	- MSE vs MAE interpretation
	- Out-of-bag error estimation in bootstrap
	- Training/test set splitting strategies

II. Linear Regression

- Ordinary Least Squares
	- Normal equations derivation (XᵀXβ = Xᵀy)
	- Problems with near-singular XᵀX matrices
	- Gauss-Markov theorem assumptions

- Variable Selection
	- Stepwise methods (forward/backward/both)
	- Best subset selection vs information criteria
	- Leaps and Bounds algorithm implementation

- Dimensionality Reduction
	- PCR vs PLS: Mathematical formulation and differences
	- Continuum regression concept
	- Relationship between PCR and ridge regression

- Shrinkage Methods
	- Ridge regression: β̂_ridge formula, bias-variance tradeoff
	- Lasso: Geometric interpretation of variable selection
	- Elastic Net vs Adaptive Lasso variants

III. Linear Classification

- Discriminant Analysis
	- LDA derivation from Bayesian decision rule
	- QDA vs LDA: Covariance matrix assumptions
	- RDA regularization approach
	
- Logistic Regression
	- Multinomial vs binomial formulation
	- Newton-Raphson algorithm for MLE
	- Weighted least squares connection

IV. Nonlinear Methods

- Basis Expansions
	- Natural cubic splines: Penalized RSS formulation
	- Knot selection strategies (fixed vs adaptive)
	- Degrees of freedom in smoothing splines

- Generalized Additive Models
	- Backfitting algorithm steps
	- Link functions vs identity functions in GAMs
	- PRSS formulation with smoothness penalties

- Tree-Based Methods
	- Regression vs classification tree splitting criteria
	- Cost-complexity pruning (α parameter tuning)
	- Gini index vs entropy impurity measures

- Random Forests
	- Bootstrap aggregation mechanics
	- Variable importance measures
	- Out-of-bag error estimation

- Support Vector Machines
	- Separable vs non-separable case optimization
	- Kernel trick mathematical foundation
	- Lagrange primal/dual formulation with KKT conditions
	- Margin maximization geometric interpretation

V. Special Topics

- Model Comparisons
	- Bias-variance tradeoff in different methods
	- PCR vs PLS vs Ridge use cases
	- Lasso vs Elastic Net for high-dimensional data

- Implementation Aspects
	- Handling categorical variables in different methods
	- Computational complexity comparisons
	- Hyperparameter tuning strategies (λ, α, C)

# most popular topics

sorted by popularity.

recent trend (between 2023 and 2025): increased focus on kernel methods (SVM) and practical implementation details (i.e. degrees of freedom in splines). the most formula-heavy topics are SVM, splines and regularization methods.

_(1) SVM_

- Optimization criteria for separable/non-separable cases, kernel trick, Lagrange multipliers [2019][2020][2022][2023][2025].  
- examples:
	- *"Write the optimization problem for separable and non-separable cases, including constraints and KKT conditions."* [2020][2025].  
	- *"Explain the kernel trick and why it avoids explicit basis expansions."* [2020][2022][2025].

_(2) regularization methods_

- Ridge vs. Lasso regression (differences, formulas, variable selection properties) [2019][2020][2021][2025].  
- Connection between PCR, Ridge, and Lasso [unknown][2021][2025].
- examples:
	- *"Compare Ridge and Lasso regression: objective functions, geometric interpretation (L1/L2 norms), and variable selection."*[2019][2020][2021][2025].  

_(3) splines_

- Smoothing splines (criterion with penalized curvature, natural cubic splines, degrees of freedom) [2019][2020][2021][2022][2025].  
- examples:
	- *"Derive the criterion for smoothing splines with a penalty term. How does λ affect the solution?"*[2019][2020][2021][2025].  

_(4) regression/classification trees_

- Splitting criteria (RSS, Gini, deviance), pruning, overfitting prevention [2019][2020][2021][2025].  
- examples:
	- *"Describe cost-complexity pruning and how to select the optimal tree size."*[2019][2020][2021].  
	- *"Explain bagging, feature selection, and variable importance measures."*[2020][2021][2022].  

_(5) PCR/PLS_

- Principal Component Regression vs. Partial Least Squares (differences, formulas, use cases) [2019][2020][2021][2025].  
- examples:
	- *"Explain how PCR addresses multicollinearity and its relationship to Ridge regression."*[2020][2021][2025].  

_(6) GAM/GL_

- Backfitting algorithm, link functions, penalized regression splines [2019][2020][2021][2025].  
- examples:
	- *"Mathematically explain the backfitting algorithm for GAMs."* [2020][2021][2025].  

_(7) LDA/QDA_

- Bayesian classification rules, covariance assumptions, discriminant functions [2020][2021].  
- examples:
	- *"Explain LDA/QDA using Bayesian classification rules, including prior probabilities, likelihood functions, and posterior probabilities."*

# most popular formulas

requested formulas across exam questions, grouped by topic, sorted by popularity.

_Linear Regression & Regularization_

- Ridge Regression:  
  $$\hat{\beta}^{\text{ridge}} = \arg\min_{\beta} \{ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p x_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p \beta_j^2 \}$$
[2019][2021][2025]
  
- Lasso Regression:  
  $$\hat{\beta}^{\text{lasso}} = \arg\min_{\beta} \{ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p x_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p |\beta_j| \}$$
[2020][2021][2025]

- OLS Solution:  
  $$\hat{\beta} = (X^\top X)^{-1}X^\top y$$
[2019][2021]

_SVM_

- Separable Case (Primal):  
  $$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t. } y_i(x_i^\top \beta + \beta_0) \geq 1$$
[2020][2023][2025]

- Non-Separable Case:  
  $$\min_{\beta,\beta_0,\xi} \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t. } y_i(x_i^\top \beta + \beta_0) \geq 1-\xi_i,\ \xi_i \geq 0$$
[2020][2025]

- Kernel Trick:  
  $$K(x_i,x_j) = \langle h(x_i), h(x_j) \rangle$$
[2020][2022]

_Splines_

- Smoothing Spline Criterion:  
  $$\text{PRSS}(f,\lambda) = \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 dt$$
[2020][2022][2025]

- Degrees of Freedom:  
  $$\text{df}_\lambda = \text{tr}(S_\lambda)$$
 where $$S_\lambda = N(N^\top N + \lambda \Omega_N)^{-1}N^\top$$
[2019][2020][2025]

_Model Selection_

- AIC:  
  $$\text{AIC} = -2\log L + 2p$$
[2021][2025]
  
- BIC:  
  $$\text{BIC} = -2\log L + \log(n) \cdot p$$
[2021][2025]

- Adjusted $R^2$:  
  $$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$$
[2019][2021]

_Classification_

- LDA Discriminant Function:  
  $$\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log\pi_k$$
[2020][2025]
  
- Logistic Regression Likelihood:  
  $$L(\beta) = \prod_{i=1}^n p(x_i)^{y_i}(1-p(x_i))^{1-y_i}$$
 where $$p(x) = \frac{e^{\beta_0 + x^\top\beta}}{1 + e^{\beta_0 + x^\top\beta}}$$
[2020][2021][2023]

_Trees_

- Regression Tree Splitting Criterion:  
  $$\min \left\{ \sum_{i:x_i \in R_1} (y_i - \hat{c}_1)^2 + \sum_{i:x_i \in R_2} (y_i - \hat{c}_2)^2 \right\}$$
[2020][2025]

_PCR/PLS_

- Principal Components:  
  $$Z_m = Xv_m$$
 where $$v_m$$
 is the $$m$$
-th eigenvector of $$X^\top X$$
[2020][2025]

_GAM_

- Backfitting Update:  
  $$f_j^{(k+1)} = S_j\left( y - \beta_0 - \sum_{l \neq j} f_l^{(k)} \right)$$
[2025]
