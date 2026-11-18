"""Candidate classifiers compared in the experiments.

Every entry returns an sklearn-compatible estimator with predict_proba. The
scaler sits inside each pipeline, so scaling statistics are always fitted on
the training fold only.
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import SEED


class EpochMLP(BaseEstimator, ClassifierMixin):
    """MLP trained epoch by epoch with partial_fit so the training and
    validation log loss of every epoch can be recorded."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, epochs=25, batch_size=1024, seed=SEED):
        self.hidden = hidden
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xv = self.scaler_.transform(X_val) if X_val is not None else None
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                                  learning_rate_init=self.lr, batch_size=self.batch_size,
                                  random_state=self.seed)
        rng = np.random.RandomState(self.seed)
        self.classes_ = np.array([0, 1])
        self.history_ = {"epoch": [], "train_loss": [], "val_loss": []}
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xs))
            self.mlp_.partial_fit(Xs[perm], y[perm], classes=self.classes_)
            self.history_["epoch"].append(ep + 1)
            sub = perm[: min(len(perm), 200_000)]
            self.history_["train_loss"].append(log_loss(y[sub], self.mlp_.predict_proba(Xs[sub])[:, 1], labels=[0, 1]))
            if Xv is not None:
                self.history_["val_loss"].append(log_loss(y_val, self.mlp_.predict_proba(Xv)[:, 1], labels=[0, 1]))
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def lgbm(**kw):
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, n_jobs=2, verbose=-1)
    base.update(kw)
    return LGBMClassifier(**base)


# name -> (family, short config string, factory, max training rows or None)
MODELS = {
    "LogisticRegression": ("Linear / probabilistic", "L2, C=1.0, lbfgs, scaled",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)), None),
    "GaussianNB": ("Linear / probabilistic", "var_smoothing=1e-9",
                   lambda: GaussianNB(), None),
    "KNN": ("Instance based", "k=31, distance weights, scaled, 150k-row subsample",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=2)),
            150_000),
    "DecisionTree": ("Tree", "max_depth=12, min_samples_leaf=50",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED), None),
    "RandomForest": ("Ensemble (bagging)", "100 trees, max_depth=18, leaf=20, max_samples=0.25",
                     lambda: RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=20,
                                                    max_samples=0.25, n_jobs=2, random_state=SEED), None),
    "ExtraTrees": ("Ensemble (randomised trees)", "150 trees, max_depth=20, leaf=20",
                   lambda: ExtraTreesClassifier(n_estimators=150, max_depth=20, min_samples_leaf=20,
                                                max_samples=0.25, bootstrap=True, n_jobs=2, random_state=SEED), None),
    "LightGBM": ("Boosting", "400 trees, lr=0.05, 63 leaves",
                 lambda: lgbm(), None),
    "XGBoost": ("Boosting", "400 trees, lr=0.05, depth=7, hist",
                lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                                      colsample_bytree=0.8, tree_method="hist", n_jobs=2,
                                      random_state=SEED, eval_metric="logloss"), None),
    "MLP": ("Neural network", "64-32 ReLU, Adam lr=1e-3, alpha=1e-4, 25 epochs",
            lambda: EpochMLP(), None),
}
