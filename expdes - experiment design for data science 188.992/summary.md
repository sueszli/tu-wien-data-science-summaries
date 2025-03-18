# introduction

*data science*

- gain insights from data
- through computation, statistics, visualization
- empirical science

*data science process*
 
- i) ask question
	- research question
- ii) get data
	- privacy
- iii) explore data
	- preprocess, clean, transform
	- missing data, outliers, anomalies, patterns
	- descriptive statistics, plots
- iv) model data
	- fit, validate
	- high bias = underfitting, inaccurate
	- high variance = overfitting, not generalizing
- v) communicate results
	- explain
	- correlation vs. causation

*crisp-dm process*

- = cross industry standardprocess for data dining
- i) business understanding
- ii) data understanding
- iii) data preparation
- iv) modeling
- v) evaluation
- vi) deployment

*legal compliance*

- general data protection regulation (gdpr):
	- personal data = identity, certificates, documents, records, health, …
	- citizens control how personal data is processed
	- no discriminatory algorithms
- eu data governance act:
	- public sector data to be reused for research (ie. in health) but not openly available
	- common european data spaces - sharing data across borders, sectors
	- voluntary data contribution by businesses
- eu data act:
	- access to smart device data
	- fairer contracts for small businesses
	- data use in emergencies by public authorities
	- easy cloud service switching by customers
- eu ai act (risk based):
	- minimal risk = no regulation → for common stuff (recommender systems, spam filters, …)
	- transparency risk = transparency requirements → for risk of impersonation, deception (chatbots, generative models, …)
	- high risk = compliance assessment in entire life cycle, citizens can file complaints → for a risk to health, safety (vehicles, medical devices, critical infra, general purpose ai…)
	- unacceptable risk = prohibition → for violation of human rights (social scoring by government, …)

*algorithm ethics: 6 types of concerns*

- epistemic = evidence quality issues
	- i) inconclusive evidence – not reliable
	- ii) inscrutable evidence – not interpretable
	- iii) misguided evidence – data has low quality: garbage in, garbage out
- normative = fairness concerns
	- iv) unfair outcomes – discriminating
	- v) transformative effects – changes our perception and society (ie. profiling)
- traceability – accountability problem
	- vi) traceability – harm can't be traced

# experiments

*experiments*

- valid = accuracy of instruments
- reliable = consistent under repetition, similar conditions
- hypothesis = expected effect of input on output
- types:
	- pilot experiment = checking instruments
	- natural experiment = uncontrolled conditions, just observations (ie. economics, meteorology)
	- field experiment = partially controlled conditions, some stimuli provided (ie. social sciences)
	- controlled experiment = fully controlled conditions

*variables*

- input = independent var
	- values are called 'control'
	- 'factorial' experiments search input space exhaustively
- output = dependent var, returns performance metric
- control = kept constant
- interfering = extraneous, nuisance, influencing some vars → placebo group tries to study them
- confounding = influences both input and output (ie. gender influences selected treatment, chance of recovery)
- latent = not directly measurable (ie. user trust)

*scales*

- nominal = enums (ie. gender)
- ordinal = have order (ie. ratings)
- interval = have numeric scale, no zero point / reference point (ie. time)
- ratio = have zero point (ie. allow mult, division)

*model validation*

- holdout split:
	- test set, train set, val set (subset of train set or a seperate set)
	- if you have over 1mio samples, use 98/2 split, otherwise 70/30
	- bias: sample might be underrepresented in train set
- bootstrapping:
	- i) shuffle
	- ii) sample $n$ data points with replacement, meaning each observation has an equal probability of being selected each time
	- iii) repeat $k$ times, generating $k$ datasets
	- datasets can be used for hyperparam testing
	- metrics can be aggregated
	- bias: sample might be overrepresented in train set
- cross validation:
	- i) shuffle
	- ii) split data in $k$ parts (folds)
	- iii) use $\frac{1}{k}$ testing, $\frac{k-1}{k}$ for training
	- iv) evaluate $k$ times, each time using a different fold
	- leave-1-out-cv = $k$ only has 1 element, accurate but too compute intensive
	- stratified-cv = each fold must have the same class distribution as the original dataset
	- nested-cv = also cv within the train set to find hyperparams
	- forward-chaining-cv = successively adds more historical timeseries data for the same test set
	- avoid data leak: hyperparam search within the same fold
	- bias: samples occur exactly once over all train sets

*metrics*

|                     | predicted positive PP       | predicted negative PN        |
| ------------------- | --------------------------- | ---------------------------- |
| actually positive P | true positive TP            | false negative FN (error II) |
| actually negative N | false positive FP (error I) | true negative TN             |

- numerical:
	- mae - mean absolute error = $\frac{1}{n} \sum{|\hat y_i - y_i|}$
	- rmse - root mean squared error = $\sqrt{\frac{1}{n} \sum{\hat y_i - y_i}}$ → punish larger values
- discrete:
	- should be computed for each class, as binary: class vs all others
	- accuracy = (tp + tn) / (tp + fp + tn + fn)
	- specificity = tn / (tn + fp) → true negative ratio
	- sensitivity/recall = tp / (tp + fn) → true positive ratio - how many relevant are selected?
	- precision = tp / (tp + fp) → how many selected are relevant?
	- f1 score = 2 · (prec · rec) / (prec + rec) → punish low values of either, can be tuned

# hypothesis testing

*hypothesis testing*

- i) find data distribution:
	- empirical sampling = use sample ratios
	- parametric statistics = assumes normal distribution (ignored in practice), then changes params
	- nonparametric statistics = no assumptions on distribution or params, rank-based methods
- ii) decide whether to reject $H_0$, bounded by some likelihood of being wrong
	- works similar to "proof by contradiction"
	- critical values = values in original range, not z-distribution
	- p-values = smaller values (likelihood of more extreme results than $\bar x$) are stronger evidence against $H_0$
- example:
	- $H_0$: accuracy has not improved vs. $H_1$: accuracy has improved
	- one-tailed test = reject $H_0$ at upper / lower 5% for alpha 5% → means 5% observations are at least as extreme as the last observed accuracy
	- two-tailed test = reject $H_0$ at upper and lower 2.5% for alpha 5%

*reducing errors errors*

- error types are inversely related - reducing one typically increases the other
- $\alpha$ = **false positive** / wrongly reject $H_0$
	- lower significance level $\alpha$
	- increase sample size:
		- larger samples narrow confidence intervals / reduces std err. but if we already have enough confidence, it just inflates any effect to significance
	- optimize number of experiments:
		- repeated/parallel testing inflates probability of false positives (multiplicity problem)
		- for independent tests, the probability compounds: $\bar{\alpha} = 1 - (1 - \alpha)^{m}$ (= family-wise error rate)
		- for 5 tests the significance level becomes: $\bar \alpha = 1 - (1 - 0.05)^{5} = 0.23$
		- bonferroni correction = divide $\alpha$ by the $m$ number of tests performed ($\alpha/m$) → makes significant results harder to obtain
	- use correct tests in cross validation:
		- in cross-validation there is a training set overlap (ie. in 5-fold-cv around 75% of train sets are shared for pairwise fold comparisons)
		- even when data is normally distributed, the overlapping training sets create dependencies between folds that invalidate the use of standard paired tests
		- use non parametrized or specialized tests instead: McNemar test, 5x2 cross-validation paired t-test
- $\beta$ = **false negative** / wrongly accept $H_0$
	- increasing power of tests = decreasing beta
- $1 \text{–} \beta$ = **power of test** / sensitivity (the test's ability to detect a difference, when one exists)
	- increase sample size
	- prefer tests with higher power (parametric tests > non-parametric tests, due to knowing distribution params)
	- for fixed $\alpha$: decrease $\delta \text{=} |\mu_0 \text{–} \mu_1|$
		- the smaller the effect/difference between group, the harder it becomes to tell the two hypotheses apart, which increases the probability for a false negative
	- for fixed $\delta$: increase $\alpha$

*central limit theorem - clt*

- = if you repeatedly take samples from any population and average them, those averages will be normally distributed, regardless of the original population's distribution
- for $N\geq30$ samples
- sample:
	- $\sigma_{\bar{x}} = \sigma / \sqrt N$ = standard error (std dev of the mean of the means of samples)
	- $\bar{x} \approx \mu$ as $n \rightarrow \infty$
- population:
	- if $\sigma$ is unknown: make an estimate $\hat \sigma := \sigma_{\bar{x}}$ (or simply $s$)
		- this is commonly the case
		- then $\sigma_{\bar{x}} \text{=} \sigma / \sqrt N$ becomes $\hat \sigma_{\bar{x}} \text{=} s / \sqrt N$
	- if $\mu, \sigma$ is unknown:
		- make up an arbitrary mean for the population as the threshold/target value (not recommended)

*overview*

- parameterized tests:
	- *z-test* = tests if **new observed mean** $\bar x$ is significantly different from population mean (if $N\geq30$, $\sigma$ known)
	- *single sample t-test* = tests if **new observed mean** $\bar x$ is significantly different from population mean (if $N<30$, $\sigma$ unknown)
	- *independent/two sample t-test* = tests if difference in means from **independent samples** is significantly different (ie. perf between two models, baseline vs. model)
	- *paired sample t-test* = tests if difference in means from **related samples** is significantly different (ie. same model under different conditions, two models on the same samples, naturally correlated observations)
	- *anova* = tests if difference in means from **3 or more independent samples** is significantly different - you can then identify that one using post-hoc tests (ie. instead of multiple t-tests)
- non-parameterized tests:
	- when normality can't be assumed
	- *sign test* = tests whether the median of the differences differs from zero from **related samples** (alternative to paired sample t-test, less power but simpler)
	- *mann-whitney u test* = tests the probability that a randomly selected value from one population exceeds a randomly selected value from another population & tests whether the medians of two populations are equal from **independent samples** (alternative to independent t-tests)
	- *wilcoxon signed-rank test* = tests whether the median of the differences differs from zero from **related samples** (alternative to paired sample t-test)
	- *kruskal-walllis test* = tests whether samples come from the same distribution by comparing mean ranks from **2 or more independent samples** (alternative to one-way anova, only makes sense from ≥3 groups)
	- *mcnemar test* = tests whether there are differences in proportions from **related samples** with binary classification (alternative to paired sample t-test for nominal data - ie. cross validation)
	- *friedman test* = tests whether samples come from the same distribution by comparing ranks from **3 or more related samples** (alternative to one-way / repeated-measures anova - ie. repeated measurements on same subjects)

*z-test*

- = tests if new observed mean $\bar x$ is significantly different from population mean (if $N\geq30$, $\sigma$ known)
- assumes: $N\geq30$ samples, $\sigma$ known, normality, independence, interval/ratio scale level, no outliers
- $Z = \frac{\bar{x} - \mu}{{\sigma} / \sqrt{n}}$ → converts to z-distribution / standard distribution where $\mu \text{=} 0, s \text{=} 1$

*t-test*

- assumes: $N<30$ samples, $\sigma$ unknown, (normality, independence, interval/ratio scale level, no outliers)
- uses std err to approximate sigma, so $\sigma_{\bar{x}} \text{=} \sigma / \sqrt N$ becomes $\hat \sigma_{\bar{x}} \text{=} s / \sqrt N$
- t-distribution has heavier tails, because we have less confidence and more extreme values
- **single sample t-test**:
	- = tests if new observed mean $\bar x$ is significantly different from population mean (if $N<30$, $\sigma$ unknown)
	- use t-distribution with $N \text{–} 1$ degrees of freedom
	- $t = \frac{\bar{x} - \mu}{{s} / \sqrt{n}}$
- **two sample t-test**:
	- = tests if difference in means from independent samples is significantly different (ie. perf between two models, baseline vs. model)
	- use t-distribution with $n_1 \text{+} n_2 \text{–} 2$ degrees of freedom (df) for critical values
	- can be one-tailed or two-tailed test
	- $t_{\bar{x}_{1}-\bar{x}_{2}} = \frac{\bar{x}_{1} - \bar{x}_{2}}{\hat{\sigma}_{\bar{x}_{1}-\bar{x}_{2}}}$
	- standard error = $\hat{\sigma}_{\bar{x}_{1}-\bar{x}_{2}} = \sqrt{\hat{\sigma}^{2}_{pooled}(\frac{1}{{N}_{1}} + \frac{1}{{N}_{2}})}$
	- pooled variance = $\hat{\sigma}^2_{pooled} = \frac{(N_1 - 1)s^2_1 + (N_2 - 1)s^2_2}{N_1 + N_2 - 2}$
- **paired sample t-test:**
	- = tests if difference in means from related samples is significantly different (ie. same model under different conditions, two models on the same samples, cross validation, naturally correlated observations)
	- requires fewer subjects since each person serves as their own control
	- minimizes variance (by having half as many test problems), increases confidence
	- $t_\delta = \frac{\bar{x}_{\delta} - {\mu}_{\delta}}{s_{\delta} / \sqrt{N_{\sigma}}}$
	- i) compute differences $\delta = |\mu_1 - \mu_2|$
	- ii) mean $\mu_{\delta}$ and std dev $\hat \sigma_{\delta}$ of differences
	- iii) initially assume that there is no significant difference $H_0 \text{: } \mu_{\delta} = 0$
	- iv) compute test statistics

*anova*

- = tests if difference in means from 3 or more independent samples is significantly different - you can then identify that one using post-hoc tests (ie. instead of multiple t-tests)
- assumes: normality, independence, homogenity of variance
- $H_0 \text{: } \mu_1 = \mu_2 = \dots = \mu_k$

*non-parametric tests*

- doesn't assume normal distribution in data → but in practice the assumptions for parametric tests are just ignored and stuff still works well
- for non-interval/ratio data, ordinal scales, data with outliers
- **sign test**:
	- = tests whether the median of the differences differs from zero from related samples (alternative to paired sample t-test)
	- assumes: ordinal scales, paired samples
	- less powerful than other tests, but simpler to use
	- null hypothesis assumes 50/50 chance of one value being larger than the other
	- $H_0 \text{: } \text{Pr}(X > Y) = 0.5$
- **wilcoxon signed-rank test**:
	- = tests whether the median of the differences differs from zero from related samples (alternative to paired sample t-test)
	- tests whether the distribution of differences is symmetrical around zero
	- assumes: interval/ordinal scales, paired samples (before/after measurements)
	- better than sign test because it considers the magnitude of differences
- **mann-whitney u test**:
	- = tests the probability that a randomly selected value from one population exceeds a randomly selected value from another population & tests whether the medians of two populations are equal from independent samples (alternative to independent t-tests)
- **kruskal-walllis test**:
	- = tests whether samples come from the same distribution by comparing mean ranks from 2 or more independent samples (alternative to one-way anova, only makes sense from ≥3 groups)
	- works with unequal sample sizes
- **friedman test**:
	- = tests whether samples come from the same distribution by comparing ranks from 3 or more related samples (alternative to one-way / repeated-measures anova - ie. repeated measurements on same subjects)
	- good for repeated measurements on same subjects

# data management

*data citation principles*

- citing data is hard
- 8 data citation principles:
	- importance = data is just as important as the publication
	- credit and attribution = to authors
	- evidence = for claims
	- unique id = for persistence, even when underlying layer changes
	- access = for human/machine accessibility
	- persistence = uses unique id
	- specificity and verifiability = provenance, for access time, version, portion/subset of data
	- interoperability and flexibility = among different communities
- other benefits:
	- giving/receiving credit, re-using existing knowledge, preventing scientific misconduct
	- identification, documentation, context, impact, transparency, reproducibility

*data management plan (dmp)*

- checklist template, tailored to community, for awareness training
- living document, changes through project
- should follow fair-principles
- i) data set description
	- aka. description of data to be collected & created
	- = type, source, volume, format
- ii) standards, metadata
	- aka. methodologies for data collection & management
	- = methodology, community standards, experiment setup details (who, when, conditions, tools, versions), metadata formats (ie. dublin core, premis)
- iii) ethics and intellectual property
	- = privacy, sensitive data, rights, permissions, consent
- iv) data sharing
	- aka. plans for data sharing and access
	- = version, access privileges, embargo periods, storage location, mutability, license
	- open data sharing = lets you cite, license, search data (ie.  zenedo, re3data, data repo)
- vi) archiving, preservation
	- aka. strategy for long-term preservation
	- = lifetime, which data to store, duration, storage/backup strategy, cost considerations, repository choice
	- persistent identifiers = user ids (orcid), data object identifier (doi) → physical location can change

*fair data principles*

- f - findable = metadata for search
- a - accessible = for machines, clear access privileges
- i - interoperable = across different communities and domains
- r - reusable = clear license, documentation (sum of the 3 other rules)

*machine actionable dmp (madmp)*

- not just for awareness training, compliance  
- not maintained by humans, less error prone, dynamic  
- uses community standards, vocabularies (ie. dublin core, premis)  
- uses persistent ids like orcid or doi  
- based on semantic web: structured, machine-readable/actionable/verifiable format, not just "promises", complex queries by stakeholders

*dynamic data citation recommendations*

- recommendations by research data alliance working group on dynamic data citation (rda wg)
- 4 recommendations to address 3 challenges
- c1) dynamic nature of data (additions/corrections over time)
	- i) preparing data & query store = data versioning, timestamping, query store → version, timestamp, checksum for each subset
- c2) granularity issues (citing precise subsets)
	- ii) data persistence = query uniqueness, stable sorting, result set verification, query timestamping, query pid, store query, citation text
- c3) technology dependencies (stability across system changes)
	- iii) resolving a pid = landing page (for humans and machines), machine actionability
	- iv) data infra modifications = technology migration, migration verification → persistent identifier (pid) stays the same even after migration

*digital preservation*

- physical preservation (bit level) = redundancy, fault tolerance, recovery, storage migration, access privileges
- logical preservation = environment emulation for old fileformats, verifiability, reproducibility
- semantic preservation = change in context (ie. changing laws, social norms, meaning of concepts)

# reproducibility

*ethics and privacy (by ACM, 2017)*

- https://www.acm.org/binaries/content/assets/public-policy/2017_usacm_statement_algorithms.pdf
- i) awareness
	- problem - harm and bias in design/implementation/use to individuals/society
	- solution - stakeholders should be aware
- ii) access and redress
	- problem - those harmed don't get help
	- solution - regulators should enforce mechanisms for questioning and redress of grievances
- iii) accountability
	- problem - institutions deflect responsibility
	- solution - institutions should be held accountable, regardless of how interpretable their models are
- iv) explanation
	- problem - no interpretability
	- solution - systems should have interpretability of processes and decisions (especially for public policy)
- v) data provenance
	- problem - mined data might have errors or harmful biases
	- solution - data collection, processing and possible biases should be documented and open (or just accessible to auditors, in case of privacy concerns)
- vi) auditability
	- problem - harmful cases not logged
	- solution - models, algorithms, data, decisions should be logged to audit cases where harm is suspected
- vii) validation and testing
	- problem - no rigorous validation
	- solution - models should be regularly validated, results should be documented and open

*primad - reproducibility levels*

- http://drops.dagstuhl.de/opus/volltexte/2016/5817/pdf/dagrep_v006_i001_p108_s16041.pdf
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

*reproducing papers*

- peer-reviewers not incentivised to reproduce
- few conferences on reproduction results
- reproducibility papers must be interesting → challenge claims, address gap with new method

*documentation frameworks*

- prov-o:
	- w3c recommended ontology for provenance information (change of objects)
	- models entities, activities, agents
	- can be integrated with other standards like: foaf, dublin core, premis
	- versioning, changelog, dependencies, …
- huggingface model cards:
	- readme.md and yaml files for documentation
	- partially autogenerated
	- not following fair principles
- fair4ml:
	- ontology for machine learning model metadata
	- integrates rdf, schema.org, codemeta
- codemeta:
	- for software
	- handles citations, versions, dependencies, descriptions
	- standardized metadata templates
	- cross-platform compatibility with github and package managers
- croissant specification:
	- extension to schema.org for ml dataset description
	- attempt to standardize documentation
	- allowing datasets to be loaded without reformatting

*automated documentation*

- context model:
	- combines static and dynamic analysis
	- i) static analysis:
		- defines steps, platforms, services, calls, dependencies, licenses
		- uses archimate implemented in OWL
	- ii) dynamic analysis:
		- process migration framework (pmf)
		- records system calls and resource access
		- analyzes file formats using pronom and premis
- vframework:
	- record and replay + redeployment + verification
	- processes expressed as "research objects"
	- can trace causes of differing behaviors in reproduced experiments
