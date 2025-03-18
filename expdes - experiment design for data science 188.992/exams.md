# 2023-01-26

## (1) question

Name 5 of the ACM Statement on Algorithmic Transparency and Accountability and explain (a) what problem they address and (b) what can be done to solve this problem

**answer:**

- i) awareness
	- problem: algorithms can contain hidden biases in design/implementation/use that harms people
	- solution: stakeholders should recognize potential harmful biases in design
- ii) access and redress
	- problem: people can be adversely affected by algorithmic decisions without recourse
	- solution: there should be questioning and correction mechanisms for those adversely affected by algorithmic decisions
- iii) accountability
	- problem: institutions may avoid responsibility and blame system complexity
	- solution: institutions must be held accountable for their systems, regardless of the explanation capability of models
- iv) explanation
	- problem: decisions can be opaque
	- solution: clear explainations for decision making
- v) data provenance
	- problem: training data collection can introduce errors and biases that discriminate
	- solution: documentation of data collection and manipulation methods
- vi) auditability
	- problem: harmful outcomes of models and data may be difficult to investigate
	- solution: systems should enable inspection
- vii) validation and testing
	- problem: systems may produce discriminatory or harmful outcomes
	- solution: rigorous evaluation of systems

## (2) question

- Most prominent advantages of automated algorithms making decision, and why?
- Most prominent disadvantages of automated algorithms making decision, and why?
- What are challenges in making decisions of algorithms transparent?
- What are disadvantages of making algorithms decisions transparent?

**answer:**

Q1: advantages of automated algorithmic decision-making

- scalability, efficiency, consistency
- less human bias, more standardized

Q2: disadvantages of automated algorithmic decision-making

- algorithm ethics: 6 types of concerns
- epistemic = evidence quality issues
	- i) inconclusive evidence – not reliable
	- ii) inscrutable evidence – not interpretable
	- iii) misguided evidence – data has low quality: garbage in, garbage out
- normative = fairness concerns
	- iv) unfair outcomes – discriminating
	- v) transformative effects – changes our perception and society (ie. profiling)
- traceability = accountability problem
	- vi) traceability – harm can't be traced

Q3: challenges in making algorithmic decisions transparent

- technical / theoretical complexity
- inherent opaqueness of black box deep learning models

Q4: disadvantages in making algorithmic decisions transparent

- security risks = exposing vulns
- gaming the system = exposing unwanted mechanics
- competitive concerns = theft of intellectual property
- reduced performance = simpler models might underfit
- cost = interpretability/transparency needs additional labor and expertise

## (3) question

- Given an ad-blocker what performance metric is most relevant + formula.
- What statistical tests can be used to measure performance against another system (name two) and explain why they can be used.
- What errors can be made in statistical testing and how can they be reduced

**answer:**

Q1: Given an ad-blocker what performance metric is most relevant + formula.

- users are more tolerant of occasionally seeing ads than having broken website features
- therefore precision (not blocking legitimate content) is more important than recall (blocking all ads)
- alternatively we can optimize for both values using a tuned f1 score
- sensitivity (recall) = tp / (tp + fn) → true positive ratio (how many relevant are selected?)
- precision = tp / (tp + fp) → how many selected are relevant?
- f1 score = 2 · (prec · rec) / (prec + rec) → punish low values of either, can be tuned

Q2: What statistical tests can be used to measure performance against another system (name two) and explain why they can be used.

- we want to compare 2 models
- we rank the tests by statistical power
- **A) assuming dependence**
	- i) paired sample t-test
		- = tests if difference in means from related samples is significantly different (ie. same model under different conditions, two models on the same samples, naturally correlated observations)
		- most powerful when normality assumptions are met
		- appropriate since measurements are paired (same test sets)
	- ii) wilcoxon signed-rank test
		- = tests whether the median of the differences differs from zero from related samples (alternative to paired sample t-test)
		- non-parametric alternative to paired t-test, doesn't assume normality
		- considers magnitude of differences, not just direction
		- more robust against outliers
		- useful if precision differences are skewed
- **B) assuming independence**
	- i) independent/two sample t-test
		- = tests if difference in means from independent samples is significantly different (ie. perf between two models, baseline vs. model)
		- assuming normality
		- larger sample sizes needed compared to dependent tests
	- ii) mann-whitney u test
		- = tests the probability that a randomly selected value from one population exceeds a randomly selected value from another population & tests whether the medians of two populations are equal from independent samples (alternative to independent t-tests)
		- better for small sample sizes
		- compares medians rather than means, making it more robust
- **C) assuming overlap of training set**
	- i) mcnemar's test → doesn't work with continuous variable as metric
		- = tests whether there are differences in proportions from related samples with binary classification (alternative to paired sample t-test for nominal data - ie. cross validation)
		- cheap to compute: requires only one fit for each algorithm
		- particularly suitable for binary classification
		- low false positive rate and computationally efficient
	- ii)  (5x2cv) paired t-test → not mentioned in lecture
		- can handle overlapping training sets in cross-validation: uses 5 iterations of 2-fold cross-validation to avoid excessive training set overlap
		- the combined 5x2cv F-test is recommended over the t-test version
	- iii) friedman test
		- = tests whether samples come from the same distribution by comparing ranks from 3 or more related samples (alternative to one-way / repeated-measures anova - ie. repeated measurements on same subjects)
		- suitable for comparing 3+ algorithms across CV folds, accounts for dependencies
		- uses rank orders rather than raw values, making it robust to the dependency issues
		- can be followed by post-hoc tests with appropriate corrections (like Bonferroni) to determine specific differences between models

Q3: What errors can be made in statistical testing and how can they be reduced

- error types are inversely related - reducing one typically increases the other
- $\alpha$ = false positive / wrongly reject $H_0$
	- lower significance level $\alpha$
	- increase sample size:
		- larger samples narrow confidence intervals / reduces std err. but if we already have enough confidence, it just inflates any effect to significance
	- optimize number of expriments:
		- repeated/parallel testing inflates probability of false positives (multiplicity problem)
		- for independent tests, the probability compounds: $\bar{\alpha} = 1 - (1 - \alpha)^{m}$ (= family-wise error rate)
		- for 5 tests the significance level becomes: $\bar \alpha = 1 - (1 - 0.05)^{5} = 0.23$
		- bonferroni correction = divide $\alpha$ by the $m$ number of tests performed ($\alpha/m$) → makes significant results harder to obtain
	- use correct tests in cross validation:
		- in cross-validation there is a training set overlap (ie. in 5-fold-cv around 75% of train sets are shared for pairwise fold comparisons)
		- even when data is normally distributed, the overlapping training sets create dependencies between folds that invalidate the use of standard paired tests
		- use non parametrized or specialized tests instead: McNemar test, 5x2 cross-validation paired t-test
- $\beta$ = false negative / wrongly accept $H_0$
	- increase power of test
	- increase sample size
- $1 \text{–} \beta$ = power of test / sensitivity / true positive rate (the test's ability to detect a difference, when one exists)
	- tests with higher power should be preferred (usually parametric tests are better than non-parametric ones, due to knowing distribution params)
	- for fixed $\alpha$: the smaller the effect (difference between groups $\delta \text{=} |\mu_0 \text{–} \mu_1|$), the harder it becomes to tell the two hypotheses apart, which increases the probability for a false negative → when $\delta$ decreases, $\beta$ increases and $1 \text{–} \beta$ decreases
	- for fixed $\delta$: the power of test increases/decreases with $\alpha$

## (4) question

Given a dataset of social media texts with timestamps, possible evaluation strategies could be time-based split or cross-validation

- What are advantages and disadvantages of the two options
- Which one would you choose and why

**answer:**

time-based split:

- similar to holdout split
- but no shuffling: preserves temporal dependencies, doesn't mix past and future data (avoids data leakage)
- can have multiple split options: forward-chaining-cv, block-chaining, rolling window, expanding window, …
- important for: time series forecasting, behavioral modeling, customer prediction models
- pros:
	- reflects real-world deployment scenarios
	- can use the temporal dimension in dataset effectively
	- better at handling concept drift: user data distribution might drift with time (ie. linguistic patterns, user activity, seasonality/evolving patterns)
- cons:
	- not suitable for small datasets, data might be sparse in certain times
	- sensitive to temporal anomalies

cross-validation:

- i) shuffle
- ii) split data in $k$ parts (folds)
- iii) use $\frac{1}{k}$ testing, $\frac{k-1}{k}$ for training
- iv) evaluate $k$ times, each time using a different fold
- pros:
	- less sensitive to data distribution anomalies, generalizes better if topics are relatively stable
	- lets you use more of your data for both training and eval
- cons:
	- more compute intensive than a single split
	- can't use temporal dimension in dataset effectively

recommendation:

- we don't know anything about the dataset or the classification task
- temporal dimension is likely to be predictive for social media data, as user behavior and language evolve over time

# 2022-01-25

## (1) question

Consider the following Confusion Matrix obtained from comparing the outputs of a machine learning classifier with the ground truth:

|                 | Classified positive | Classified negative |
| --------------- | ------------------- | ------------------- |
| Actual positive | 611                 | 89                  |
| Actual negative | 194                 | 106                 |

Calculate Accuracy, Sensitivity, Specificity, Precision, and Recall. Write the equation for each and fill in the numbers.

Notes:

- Presenting a final, real number is not necessary, a fraction is sufficient.
- you can use LaTeX Math notation in your answers, but it should not even be necessary.

Now consider the following Cost Matrix, showing the costs associated with making certain errors.

| Error Cost      | Classified positive | Classified negative |
| --------------- | ------------------- | ------------------- |
| Actual positive | 0                   | 12                  |
| Actual negative | 2                   | 0                   |

Given these costs, would you optimize the machine learning classifier towards Precision or Recall? Explain your answer!

**answer:**

Q1: Calculate Accuracy, Sensitivity, Specificity, Precision, and Recall. Write the equation for each and fill in the numbers.

- tp = 611
- tn = 106
- fp = 194
- tn = 89
- accuracy = (tp + tn) / (tp + fp + tn + fn) = (611 + 106) / (611 + 106 + 194 + 89) = 717/1000 = 0.717
- specificity = tn / (tn + fp) = 106 / (106 + 194) = 53/150 = 0.3533
- sensitivity/recall = tp / (tp + fn) = 611 / (611 + 89) = 611/700 = 0.8728
- precision = tp / (tp + fp) = 611 / (611 + 194) = 611/805 = 0.7590
- f1 score = 2 · (prec · rec) / (prec + rec) = 2 · (0.7590 · 0.8728) / (0.7590 + 0.8728) = 0.8119

Q2: Given these costs, would you optimize the machine learning classifier towards Precision or Recall? Explain your answer!

- the cost of false negatives (12) is a lot higher than false positives (2)
- we should optimize for recall rather than precision to minimize the total cost of misclassification → recall minimizes false negatives, by measuring the proportion of actual positives correctly identified
- while this might lead to more false positives, their lower cost makes this trade-off acceptable

## (2) question

For comparison of two machine learning based email spam classifiers A and B, a ground truth annotated dataset is randomly split into 90% training data and 10% test data. Training and test data are used in the same manner for both A and B. This procedure is carried out 20 times.

1. Explain the concept of repeated test-train spilts and how it differs from other evaluation setups (briefly compare to at least two other strategies). What are advantages and disadvantages?
2. In order to find out whether one of the classifiers is statistically significantly better than the other with regard to the chosen evaluation measure, a statistical test should be carried out. Which test is applicable here and why? (If multiple tests are applicable, pick the one with highest power.)
3. Which types of errors can you make in statistical hypothesis testing? Give a brief definition of each. How can you minimize the chances of making these errors?

**answer:**

Q1: Explain the concept of repeated test-train spilts and how it differs from other evaluation setups (briefly compare to at least two other strategies). What are advantages and disadvantages?

- (repeated) holdout split:
	- shuffle, split into 2 sets (test, train), repeat
	- pros:
		- better understanding of model stability and performance variance → dependency on a single random split might be unrepresentative
		- allows estimation of confidence intervals for performance metrics
		- works well with small samples, where a single split may not be reliable
		- more flexible than k-fold CV since number of repetitions and split ratios can be freely chosen
	- cons:
		- bias: samples can be under/over-represented in training sets
		- sensitive to random seed selection
		- more compute intensive than a single split
		- may not be necessary for very large datasets where a single split could be sufficient
		- not suitable for time-series data
- bootstrapping:
	- i) shuffle
	- ii) sample $n$ data points with replacement, meaning each observation has an equal probability of being selected each time
	- iii) repeat $k$ times, generating $k$ datasets
	- pros:
		- robust error estimates
		- works well with small datasets
		- allows for estimation of statistical properties
		- good for ensemble methods like bagging
	- cons:
		- bias: sample might be overrepresented in train set → errors due to duplicate samples
		- samples can be overrepresented in training set
		- computationally most intensive
		- may overestimate model performance
		- can be sensitive to outliers
- cross validation:
	- i) shuffle
	- ii) split data in $k$ parts (folds)
	- iii) use $\frac{1}{k}$ testing, $\frac{k-1}{k}$ for training
	- iv) evaluate $k$ times, each time using a different fold
	- leave-1-out-cv = $k$ only has 1 element, accurate but too compute intensive
	- stratified-cv = each fold must have the same class distribution as the original dataset
	- nested-cv = also cv within the train set to find hyperparams
	- forward-chaining-cv = successively adds more historical timeseries data for the same test set
	- pros:
		- bias: samples occur exactly once over all train sets
		- more stable estimates
		- typically less compute intensive (ie. k=10) than repeated holdout splits
	- cons:
		- may not preserve data distribution in each fold
		- requires careful stratification for imbalanced datasets

best practices for the spam classifier comparison:

- multiplicity problem / bonferroni correction:
	- the probability of obtaining at least one false positive increases with each additional test
	- bonferroni corrected significance level: 0.05/20=0.0025
	- makes it harder to detect genuine differences between the classifiers
- average performance metrics across all 20 iterations to get stable estimates
- use the same random splits (and seeds) for both classifiers A and B to ensure fair comparison
- consider stratified sampling to maintain class distribution across splits if spam/non-spam ratio is imbalanced

Q2: In order to find out whether one of the classifiers is statistically significantly better than the other with regard to the chosen evaluation measure, a statistical test should be carried out. Which test is applicable here and why? (If multiple tests are applicable, pick the one with highest power.)

- chosen metric: precision → users are more tolerant of occasionally seeing ads than missing important emails, therefore precision (not blocking legitimate content) is more important than recall (blocking all ads)
- see 2023-01-26 exam, question 3

Q3: Which types of errors can you make in statistical hypothesis testing? Give a brief definition of each. How can you minimize the chances of making these errors?

- see 2023-01-26 exam, question 3

## (3, 6) question

Below are 2 statements from the Modelers' Hippocratic Oath (Derman, 2012) and 4 rules for responsible big data research (Zook et al., 2017). For EACH of the 6 statements/rules, answer the following questions:

- (i) Explain the statement/rule and state what aspect of Data Science it is meant to warn about. 
- (ii) Give one concrete example of a situation in Data Science in which this statement/rule is applicable. 
- (iii) Explain which measures should be taken by Data Scientists to satisfy the statement/rule.  

Make sure that you number your answers clearly as A(i),(ii),(iii), …

- A. Though I will use models boldly to estimate value, I will not be overly impressed by mathematics.
- B. I will remember that I didn't make the world, and it doesn't satisfy my equations.
- C. Acknowledge that data are people and can do harm.
- D. Consider the strengths and limitations of your data; big does not automatically mean better
- E. Recognize that privacy is more than a binary value.
- F. Design your data and systems for auditability.

**answer:**

- A) Though I will use models boldly to estimate value, I will not be overly impressed by mathematics.
- B) I will remember that I didn't make the world, and it doesn't satisfy my equations.
	- explaination: don't place excessive faith in models. essentially, "all models are wrong, but some are useful". no model can fully capture real-world complexity.
	- examples: financial modeling. during the 2008 financial crisis sophisticated risk models failed to predict market collapse. black swan events can happen anytime.
	- measures: document implicit model assumptions and limitations, test models against real-world scenarios, be skeptic of model decisions, update model on concept drift, don't oversimplify reality.
- C) Acknowledge that data are people and can do harm.
	- explaination: careless handling of data can cause tangible harm to people.
	- examples: data leaks can make sensitive information public.
	- measures: treat all data as potentially sensitive until proven otherwise, think of ethics, protect privacy.
- D) Consider the strengths and limitations of your data; big does not automatically mean better
	- explaination: emphasizes the importance of data quality and context, limitations, biases, potential conflicts of interest.
	- examples: medical records might exclude certain demographic groups due to limited healthcare access, leading to biased conclusions.
	- measures: document implicit data limitations, biases, provenance, evolution, context in which data was collected. consider whether the data is appropriate for the intended analysis.
- E) Recognize that privacy is more than a binary value.
	- explaination: privacy exists on a spectrum and depends on context, data type, potential uses rather than being simply "private" or "public".
	- examples: health data might be considered private in most contexts, but sharing it with medical researchers.
	- measures: consider varying levels of data sensitivity, adapt privacy measures to specific use cases, implement contextual privacy protections.
- F) Design your data and systems for auditability.
	- explaination: create transparent/interpretable/tracable systems that can be reviewed and evaluated for accuracy, fairness, ethical compliance and ensure accountability.
	- examples: high-stakes decision-making systems like loan approval algorithms, auditability allows for detecting and correcting biases or errors that might discriminate against certain groups.
	- measures: implement logging systems for model decisions and performance, create clear audit/documentation trails for decisions, design transparent processes for external review.

## (4) question

Data Management Plans

- a) Explain the five common themes / key aspects that need to be addressed in a data management plan and provide examples for each
- b) Describe the motivation and key concepts underlying machine-actionable DMPs

**answer:** 

Q1: Explain the five common themes / key aspects that need to be addressed in a data management plan and provide examples for each

- checklist template, tailored to community, for awareness training
- living document, changes through project
- should follow fair-principles
- i) data set description
	- aka. description of data to be collected & created
	- = type, source, volume, format
- ii) standards, metadata
	- aka. methodologies for data collection & management
	- = methodology, community standards, experiment setup details (who, when, conditions, tools, versions), metadata formats (ie. dublin core, premis)
- iii) ethics and Intellectual property
	- = privacy, sensitive data, rights, permissions, consent
- iv) data sharing
	- aka. plans for data sharing and access
	- = version, access privileges, embargo periods, storage location, mutability, license
	- open data sharing = lets you cite, license, search data (ie.  zenedo, re3data, data repo)
- vi) archiving, preservation
	- aka. strategy for long-term preservation
	- = lifetime, which data to store, duration, storage/backup strategy, cost considerations, repository choice
	- persistent identifiers = user ids (orcid), data object identifier (doi) → physical location can change

Q2: Describe the motivation and key concepts underlying machine-actionable DMPs

- disadvantages of dmps:
	- just for awareness training, compliance
	- tedious to maintain by humans, error prone
	- can't be verified, machine processed, just "promises"
- advantages of madmps: (based on semantic web)
	- uses community standards, vocabularies (ie. dublin core, premis)
	- uses persistent ids like orcid or doi
	- structured, machine-readable/actionable format
	- allows machine validation, verification
	- allows complex querying by stakeholders, can be integrated in other workflows
	- dynamic, can be updated through lifecycle

## (5) question

Reproducibility

- a) List and describe common sources of irreproducibility from a computing perspective
- b) Describe the PRIMAD model of reproducibility types and explain the insights gained by priming (modifying) the respective elements

**answer:**

Q1: List and describe common sources of irreproducibility from a computing perspective

- technical sources
	- software dependencies: undefined dependencies/versions, missing/incomplete/malfunctioning code, undocumented params and configs, breaking apis
	- hardware dependencies: non-portability, unknown arch and os, build issues, filesystem/encoding differences
- data related sources
	- missing/inaccessible data, undefined versions/subset indices/preprocessing steps, difference in quality/format, data drift over time
- documentation sources:
	- missing documentation/metadata/provenance information on system/process/experimental design - especially for manual
- process related sources
	- experimental setup: non-deterministic behavior, unspecified rng seed, timing and performance variation, order-dependent operations with side effects, missing environment variables
	- human factors: complexity, implicit knowledge, subjectivity/lack of standardization

Q2: Describe the PRIMAD model of reproducibility types and explain the insights gained by priming (modifying) the respective elements

- http://drops.dagstuhl.de/opus/volltexte/2016/5817/pdf/dagrep_v006_i001_p108_s16041.pd
- some benefits of priming overlap:
	- any element (except actor, research objective) improves correctness of hypothesis
	- repeating as an effort achieves determinism through exact replication
- primad:
	- p - platform/stack
		- = software env and infrastructure used to conduct experiments
		- effort: porting
		- benefits: portability
	- r - research objective
		- = goals and questions being investigated
		- effort: re-using
		- benefits: re-purpose, re-using code in different cross-disciplinary settings, resource efficiency, correctness of hypothesis
	- i - implementation
		- = software implementation of methodology
		- effort: re-coding
		- benefits: correctness of implementation, portability, efficiency, more outreach
	- m - method
		- = methodology, algorithms
		- effort: validating
		- benefits: correctness of hypothesis, validation with different methodogical approaches
	- a - actors
		- = people involved in conducting the experiments
		- effort: independent verification
		- benefits: sufficiency of information, independent verification, transparency
	- d - raw data
		- = raw data used in the experiment
		- effort: generalizing
		- benefits: re-use for different settings
	- d - parameters
		- = config values/settings that control the experimental process
		- effort: parameter sweep
		- benefits: robustness testing, sensitivity analysis, parameter sweep, determinism

## (7) question

Consider an experimental setup in which you want to compare two machine learning algorithms A (SVM) and B (CNN). In total, the data set consists of 15.000 instances, and

you are using cross validation with k=50 folds and F-measure (with macro averaging) as performance criteria.

1. Explain the concept of cross validation and how it differs from other evaluation setups (briefly compare to at least two other strategies). What are advantages and disadvantages?
2. Outline at least two different strategies to perform a significance test on the generated measurements (F-measure values; or other depending on the strategy). For each strategy, which significance test is applicable and how many sampled values are underlying your test, i.e. which value does N have?
3. Which types of errors can you make in statistical hypothesis testing? Give a brief definition of each. How can you minimize the chances of making these errors?

**answer:**

Q1: Explain the concept of cross validation and how it differs from other evaluation setups (briefly compare to at least two other strategies). What are advantages and disadvantages?

- see this exam, question 2

Q2: Outline at least two different strategies to perform a significance test on the generated measurements (F-measure values; or other depending on the strategy). For each strategy, which significance test is applicable and how many sampled values are underlying your test, i.e. which value does N have?

- see 2023-01-26 exam, question 3

Q3: Which types of errors can you make in statistical hypothesis testing? Give a brief definition of each. How can you minimize the chances of making these errors?

- see 2023-01-26 exam, question 3

# 2019-01-23

##  (1) question

Reproducibility

- What is PRIMAD, name the components and explain what "priming" each of them achieves.
- Which issues are there in reproducibility (from a programming perspective)?

**answer:** see see 2022-01-25 exam, question 5

##  (2, 4) question

Trust in AI/ML

- What are the benefits of the automation of decision making?
- What are the issues of the automation of decision making?
- What are the benefits of "black-box" algorithms? / What are disadvantages of making algorithms decisions transparent?
- What are the drawbacks of "black-box" algorithms? / What are challenges in making decisions of algorithms transparent?

**answer:** see 2023-01-26 exam, question 2

##  (3) question

WEKA / CV, statistical significance

The data is split into train and test exactly the same way for two algorithms (classification of patents) and repeated 20 times. There was a simple WEKA workflow of loading the data, standardizing, CV, KNN, averaging results, outputting results.

- What is CV? Explain it with a figure for k=4. 
- What is leave-one-out CV? Explain the benefits/drawbacks
- What is wrong with the WEKA workflow?
- Which performance measures can be used to measure the performance of the algorithms? Which one makes most sense. Explain the measure (formula)
- Which statistical test can be used to compare the algorithms / test the significance? Explain why it can be used here.
- Which types of errors are there? What are they? How can they be reduced/prevented?

**answer:**

Q1: What is CV? Explain it with a figure for k=4. 

- i) shuffle
- ii) split data in $k$ parts (folds)
- iii) use $\frac{1}{k}$ testing, $\frac{k-1}{k}$ for training
- iv) evaluate $k$ times, each time using a different fold

Q2: What is leave-one-out CV? Explain the benefits/drawbacks

- loocv / leave-1-out-cv = cross validation where $k$ only has 1 element
- pros:
	- minimal bias, uses nearly all data points for training (n-1 observations), exhaustive validation method
	- valuable for small datasets (where data efficiency is crucial)
	- deterministic results with no randomness in the splitting process, yields the same performance estimate every time it's run on a given dataset
- cons:
	- too compute intensive, requires training n separate models, where n is the number of observations
	- can produce high variance in performance metrics, test sets of single observations may lead to unstable performance metrics

Q3: What is wrong with the WEKA workflow?

- simply repeating CV 20 times with the same split may not give reliable performance estimates
- no parameter optimization for the algorithms (the k parameter for KNN)
	- the algorithm suffers from the "curse of dimensionality" with high-dimensional patent data
	- the choice of k value significantly impacts results and should be optimized
- k-fold cross-validation isn't stratified to ensure balanced class distributions
- data leakage: the standardization step should be performed within each fold, not on the entire dataset upfront

Q4: Which performance measures can be used to measure the performance of the algorithms? Which one makes most sense. Explain the measure (formula)

- the cost of reviewing redundant patents (low precision) is far less than the cost of missing a relevant patent (low recall) due to the risk of expensive lawsuits, invalid patent grants
- the (tuned) f1 score can also make sense to punish low values of either, because there is a recall-precision-tradeoff
- sensitivity/recall = tp / (tp + fn) → true positive ratio - how many relevant are selected?
- precision = tp / (tp + fp) → how many selected are relevant?
- f1 score = 2 · (prec · rec) / (prec + rec) → punish low values of either, can be tuned

 Q5: Which statistical test can be used to compare the algorithms / test the significance? Explain why it can be used here.
 
-  see 2023-01-26 exam, question 3

Q6: Which types of errors are there? What are they? How can they be reduced/prevented?

-  see 2023-01-26 exam, question 3

##  (5) question

Data Citation

- What are the two challenges with data citation and list the (unsuccessful) approaches to overcome them.
- Describe the approach of the RDA Data Citation WG to resolve this issue.

**answer:**

Q1: What are the two challenges with data citation and list the (unsuccessful) approaches to overcome them.

- i) dynamic nature of data
	- = data gets corrected, extended over time
	- unsuccessful solutions:
		- using "accessed at" (no change history or version control)
		- aggregating changes into larger releases, delaying releaaes
	- proposed solution: versioning with timestamps
- ii) granularity issues
	- unsuccessful solutions:
		- storing subset (doesn't scale)
		- citing textually (imprecise)
		- listing indices (imprecise)
	- proposed solution: query-based citations instead of storing static subsets
- iii) technology dependencies
	- unsuccessful solutions:
		- basic manual documentation of data versions (error prone, not reproducible)
		- simple doi, static snapshots of databases
	- proposed solution: technology-independent architecture

Q2: Describe the approach of the RDA Data Citation WG to resolve this issue.

- recommendations by research data alliance working group on dynamic data citation (rda wg)
- 4 recommendations to address 3 challenges
- c1) dynamic nature of data (additions/corrections over time)
	- i) preparing data & query store = data versioning, timestamping, query store → version, timestamp, checksum for each subset
- c2) granularity issues (citing precise subsets)
	- ii) data persistence = query uniqueness, stable sorting, result set verification, query timestamping, query pid, store query, citation text
- c3) technology dependencies (stability across system changes)
	- iii) resolving a pid = landing page (for humans and machines), machine actionability
	- iv) data infra modifications = technology migration, migration verification → persistent identifier (pid) stays the same even after migration

## (6) question

Statistical testing

- Given an experimental setup with 10.000 instances, 20-fold cross validation (k-fold with k=20) and accuracy as performance measurement.
	- List two approaches for statistical significance testing
	- What's the sample size, e.g. what's $N$ in this case.
- What errors can be made in statistical hypothesis testing? Explain them briefly. And how to reduce the possibility to make them?

**answer:**

Q1: List two approaches for statistical significance testing

- see 2023-01-26 exam, question 3

Q2: What's the sample size, e.g. what's $N$ in this case?

- there are 20 folds, each fold consisting of 10.000 / 20 = 500 samples
- the test set is always 1 fold, therefore $N$ = 500

Q3: What errors can be made in statistical hypothesis testing? Explain them briefly. And how to reduce the possibility to make them?

- see 2023-01-26 exam, question 3

##  (7) question

Experimental Setup

- Given a social media classification system, 1 mio posts which include an unique id, some text and a timestamp. You have two experimental setups, once with time-based split and one with cross-validation split.
- Describe both, e.g. with a sketch.
- Which approach would you suggest? What are the advantages and disadvantages of both?

**answer:** see 2023-01-26 exam, question 4
