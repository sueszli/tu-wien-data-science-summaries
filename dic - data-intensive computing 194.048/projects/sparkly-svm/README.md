# Compare Spark text classification experiments


In Data-intensive Computing (194.048, summer 2024), the team classified review categories. It used Spark to build TF-IDF features, selected features by chi-square, and evaluated an SVM.

[Configuration-level results](results.csv) retain the recorded feature limit, regularization, iteration cap, original run parameter (p), and training and validation F1. The report lists a test F1 of 0.606259 for the configuration chosen on validation. Training F1 is not a substitute for held-out performance.

Run the result reader with Python 3. It ranks the archived runs by validation F1 and does not train a model.

    python3 summarize.py

No review rows, course solutions, or cluster configuration are included.
