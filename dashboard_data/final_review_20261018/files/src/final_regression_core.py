"""Train-only classical regression and purged nested temporal partitions.

All estimators consume a fixed causal schema: current volatility (1), the
original 28-day daily/minute features (175), and original HAR features (44).
Targets passed to fit are unchanged absolute seven-horizon daily volatilities.
Every scaler, sample-weight normalization and parameter fit uses supplied rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

FAMILIES = ('kNN', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'XGBoost', 'SVR')
METHODS = ('grid', 'random', 'bayesian', 'genetic')
HYBRID = 'HAR-XG'
HORIZON = 7
SCHEMA_WIDTH = 220


def validate_matrix(X):
    """Return finite model inputs, preserving the documented 220-column schema.

    Parameters
    ----------
    X : array-like of shape (n_samples, 220)
        Completed origin-day causal inputs. Column zero is positive volatility.

    Returns
    -------
    ndarray
        Float matrix without imputation or fitted transformation.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != SCHEMA_WIDTH or not np.isfinite(X).all() or (X[:, 0] <= 0).any():
        raise ValueError('Expected finite 220-column inputs with positive current volatility.')
    return X


def lag_features(X, input_window):
    """Select daily lag columns and all seven known-decay inputs causally."""
    if input_window not in (7, 14, 21, 28):
        raise ValueError('Input window must be 7,14,21 or28 days.')
    return np.column_stack([X[:, 1:1+6*input_window], X[:, 169:176]])


def sample_weights(base, power):
    """Normalize training-only weights; power2 aligns relative and level SSE."""
    scaled = (base / np.max(base)) ** float(power)
    return scaled / np.mean(scaled)


class FinalRegressor(RegressorMixin, BaseEstimator):
    """Causal relative-target model with inverse transform to original units.

    Parameters
    ----------
    family : str
        One of FAMILIES or 'HAR-XG'. The hybrid fits two independent components.
    params : dict or None
        Hyperparameters selected solely within a purged training support.
    seed : int
        Reproducible estimator seed, separate from the search sampler seed.
    threads : int
        Numerical workers per fit. Outer task parallelism normally uses one.
    """
    def __init__(self, family='HAR-XG', params=None, seed=42, threads=1):
        self.family = family
        self.params = params
        self.seed = seed
        self.threads = threads

    def _xgb(self, p):
        return XGBRegressor(n_estimators=int(p.get('n_estimators', 64)),
            max_depth=int(p.get('max_depth', 2)), learning_rate=float(p.get('learning_rate', .05)),
            reg_lambda=float(p.get('reg_lambda', 10.)), min_child_weight=5,
            subsample=.8, colsample_bytree=.8, base_score=0.,
            objective='reg:squarederror', tree_method='hist', max_bin=64,
            multi_strategy='one_output_per_tree', n_jobs=self.threads, random_state=self.seed)

    def fit(self, X, y):
        """Fit on supplied training rows only; y must contain all seven labels."""
        X = validate_matrix(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X), 7) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Expected seven finite nonnegative target columns.')
        p = dict(self.params or {})
        self.n_features_in_ = SCHEMA_WIDTH
        base = X[:, 0]
        z = y / base[:, None] - 1.
        self.input_window_ = int(p.get('input_window', 14))
        self.weight_power_ = float(p.get('weight_power', 2.))
        weights = sample_weights(base, self.weight_power_)
        A = lag_features(X, self.input_window_)
        if self.family == 'HAR-XG':
            self.ridge_weight_ = float(p.get('ridge_weight', .5))
            self.har_mask_ = np.arange(44) != 2  # exact span1 downside redundancy
            H = X[:, 176:][:, self.har_mask_]
            self.har_model_ = make_pipeline(StandardScaler(), Ridge(
                alpha=float(p.get('har_alpha', 100.)), solver='cholesky'))
            self.har_model_.fit(H, z, ridge__sample_weight=weights)
            self.base_model_ = self._xgb(p)
            self.base_model_.fit(A, z, sample_weight=weights)
        elif self.family == 'kNN':
            self.model_ = make_pipeline(StandardScaler(), KNeighborsRegressor(
                n_neighbors=int(p.get('n_neighbors', 20)), weights=p.get('weights', 'distance'),
                p=2, algorithm='brute', n_jobs=self.threads))
            self.model_.fit(A, z)  # kNN has no training sample_weight API
        elif self.family == 'Ridge':
            self.model_ = make_pipeline(StandardScaler(), Ridge(
                alpha=float(p.get('alpha', 10.)), solver='cholesky'))
            self.model_.fit(A, z, ridge__sample_weight=weights)
        elif self.family == 'Lasso':
            self.model_ = make_pipeline(StandardScaler(), Lasso(alpha=float(p.get('alpha', .001)),
                max_iter=20000, tol=.001, selection='cyclic', random_state=self.seed))
            self.model_.fit(A, z, lasso__sample_weight=weights)
        elif self.family == 'DecisionTree':
            self.model_ = DecisionTreeRegressor(max_depth=p.get('max_depth', 4),
                min_samples_leaf=int(p.get('min_samples_leaf', 10)), random_state=self.seed)
            self.model_.fit(A, z, sample_weight=weights)
        elif self.family == 'RandomForest':
            self.model_ = RandomForestRegressor(n_estimators=int(p.get('n_estimators', 64)),
                max_depth=p.get('max_depth', 8), min_samples_leaf=int(p.get('min_samples_leaf', 5)),
                max_features=.7, random_state=self.seed, n_jobs=self.threads)
            self.model_.fit(A, z, sample_weight=weights)
        elif self.family == 'XGBoost':
            self.model_ = self._xgb(p)
            self.model_.fit(A, z, sample_weight=weights)
        elif self.family == 'SVR':
            self.model_ = make_pipeline(StandardScaler(), MultiOutputRegressor(LinearSVR(
                C=float(p.get('C', .01)), epsilon=float(p.get('epsilon', .01)),
                loss='squared_epsilon_insensitive', dual=False, max_iter=20000,
                tol=.0001, random_state=self.seed), n_jobs=1))
            self.model_.fit(A, z, multioutputregressor__sample_weight=weights)
        else:
            raise ValueError(f'Unknown model family: {self.family}')
        return self

    def predict(self, X):
        """Return seven forecasts in original percentage-point volatility units."""
        X = validate_matrix(X)
        A = lag_features(X, self.input_window_)
        if self.family == 'HAR-XG':
            hp = np.maximum(X[:, [0]] * (1. + self.har_model_.predict(X[:, 176:][:, self.har_mask_])), 0.)
            xp = np.maximum(X[:, [0]] * (1. + self.base_model_.predict(A)), 0.)
            return self.ridge_weight_ * hp + (1. - self.ridge_weight_) * xp
        return np.maximum(X[:, [0]] * (1. + self.model_.predict(A)), 0.)


def search_space(family):
    """Return predeclared finite domains; order is the deterministic Grid order."""
    common = {'input_window': [14, 7, 28]}
    if family == 'kNN':
        return dict(common, n_neighbors=[20, 5, 60], weights=['distance', 'uniform'])
    if family in ('Ridge', 'Lasso'):
        levels = [10., .01, 1000.] if family == 'Ridge' else [.001, .0001, .01]
        return dict(common, alpha=levels, weight_power=[2, 0])
    if family == 'DecisionTree':
        return dict(common, max_depth=[4, 2, 8], min_samples_leaf=[10, 3, 20])
    if family == 'RandomForest':
        return dict(common, max_depth=[8, 4, None], min_samples_leaf=[5, 15], n_estimators=[24, 48])
    if family in ('XGBoost', 'HAR-XG'):
        space = dict(common, max_depth=[2, 1, 3], learning_rate=[.05, .1],
                     n_estimators=[64, 128], weight_power=[2, 0])
        if family == 'HAR-XG':
            space.update(har_alpha=[100., 10., 1000.], ridge_weight=[.5, .25, .75, 1., 0.])
        return space
    if family == 'SVR':
        return dict(common, C=[.01, .001, .1], epsilon=[.01, .1])
    raise ValueError('Unknown family.')


def purged_inner_folds(origins, n_splits=2, min_train=70, horizon=7):
    """Partition supplied daily-origin POSITIONS into expanding purged folds.

    The complete seven-target training vector ends strictly before the first
    validation origin. Gaps remain calendar-position gaps; no row compression.
    Returns lists of original panel positions, not indices into a filtered copy.
    """
    origins = np.asarray(origins)
    if origins.ndim != 1 or origins.dtype.kind not in 'iu' or len(origins) < min_train + n_splits*14:
        raise ValueError('Insufficient increasing chronological origins for nested folds.')
    if (np.diff(origins) <= 0).any():
        raise ValueError('Origins must be unique and increasing.')
    candidates = np.flatnonzero(np.searchsorted(origins, origins - horizon, side='left') >= min_train)
    if not len(candidates):
        raise ValueError('Insufficient purged training history.')
    first = max(int(candidates[0]), len(origins)//2)
    blocks = np.array_split(origins[first:], n_splits)
    folds = []
    for validation in blocks:
        train = origins[origins + horizon < validation[0]]
        if len(train) < min_train or len(validation) < 7 or train[-1] + horizon >= validation[0]:
            raise ValueError('Invalid temporal training/validation separation.')
        folds.append((train.copy(), validation.copy()))
    return folds


def model_label(family, method):
    """Stable label shared by logs, frozen models, forecasts and inferential family."""
    return f'{family} [{method}]'
