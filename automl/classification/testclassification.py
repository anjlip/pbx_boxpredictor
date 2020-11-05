import sklearn.datasets
import sklearn.metrics

from tpot import TPOTClassifier

import multiprocessing

# other imports, custom code, load data, define model...

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=1)

pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))

