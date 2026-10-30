"""Deterministic session features and a train-fitted scikit-learn pipeline.

Use make_preprocessing_pipeline() inside each cross-validation fold. The fitted
artifact supplied in this project is only for the fixed 80/20 split.
"""
import inspect
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

COUNT_COLUMNS = ['Administrative', 'Informational', 'ProductRelated']
DURATION_COLUMNS = [c + '_Duration' for c in COUNT_COLUMNS]
NOMINAL_COLUMNS = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']
BASE_LOG_COLUMNS = COUNT_COLUMNS + DURATION_COLUMNS
ENGINEERED_LOG_COLUMNS = ['TotalPages', 'TotalDuration', 'AvgTimePerPage',
                          'AvgProductTime', 'AvgAdminTime', 'AvgInfoTime']
BOUNDED_COLUMNS = ['BounceRates', 'ExitRates', 'SpecialDay',
                   'ProductPageShare', 'ProductTimeShare']
BINARY_COLUMNS = ['Weekend', 'NoTrackedPages', 'ZeroDurationSession',
                  'HasInformational', 'HasAdministrative']
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def safe_ratio(numerator, denominator):
    """Return zero for a zero denominator; missing inputs stay missing."""
    n = np.asarray(numerator, dtype=float)
    d = np.asarray(denominator, dtype=float)
    result = np.zeros_like(n, dtype=float)
    np.divide(n, d, out=result, where=d != 0)
    result[np.isnan(n) | np.isnan(d)] = np.nan
    return result


class SessionFeatureEngineer(TransformerMixin, BaseEstimator):
    """Row-wise features only: never read Revenue or learn population statistics.

    include_pagevalues selects one of two matched comparison variants. Its
    prediction-time availability is unverified, so the no-argument default is
    the conservative without-PageValues variant. EDA alone does not prove leakage.
    """
    def __init__(self, include_pagevalues=False):
        self.include_pagevalues = include_pagevalues

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Input must be a pandas DataFrame with original column names.')
        if 'Revenue' in X or 'source_row_id' in X:
            raise ValueError('Remove target and row identifiers before preprocessing.')
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if 'Revenue' in X or 'source_row_id' in X:
            raise ValueError('Remove target and row identifiers before preprocessing.')
        z = X.copy() if self.include_pagevalues else X.drop(columns=['PageValues'], errors='ignore').copy()
        # min_count keeps a missing component missing instead of silently summing it as 0.
        z['TotalPages'] = z[COUNT_COLUMNS].sum(axis=1, min_count=3)
        z['TotalDuration'] = z[DURATION_COLUMNS].sum(axis=1, min_count=3)
        z['AvgTimePerPage'] = safe_ratio(z['TotalDuration'], z['TotalPages'])
        z['AvgProductTime'] = safe_ratio(z['ProductRelated_Duration'], z['ProductRelated'])
        z['AvgAdminTime'] = safe_ratio(z['Administrative_Duration'], z['Administrative'])
        z['AvgInfoTime'] = safe_ratio(z['Informational_Duration'], z['Informational'])
        z['ProductPageShare'] = safe_ratio(z['ProductRelated'], z['TotalPages'])
        z['ProductTimeShare'] = safe_ratio(z['ProductRelated_Duration'], z['TotalDuration'])
        z['NoTrackedPages'] = (z['TotalPages'] == 0).astype(int)
        z['ZeroDurationSession'] = ((z['TotalPages'] > 0) & (z['TotalDuration'] == 0)).astype(int)
        z['HasInformational'] = (z['Informational'] > 0).astype(int)
        z['HasAdministrative'] = (z['Administrative'] > 0).astype(int)
        # Numeric category codes are labels, not measured quantities.
        for col in NOMINAL_COLUMNS:
            z[col] = z[col].astype('string').astype(object).where(z[col].notna(), np.nan)
        z['Weekend'] = z['Weekend'].astype(float)
        return z

    def get_feature_names_out(self, input_features=None):
        original = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        derived = ENGINEERED_LOG_COLUMNS + ['ProductPageShare', 'ProductTimeShare',
                                           'NoTrackedPages', 'ZeroDurationSession',
                                           'HasInformational', 'HasAdministrative']
        return np.asarray([c for c in original if self.include_pagevalues or c != 'PageValues'] + derived, dtype=object)



def make_onehot_encoder():
    """Use the dense-output keyword supported by the installed sklearn version."""
    options = {'handle_unknown': 'ignore'}
    if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
        options['sparse_output'] = False
    else:
        options['sparse'] = False
    return OneHotEncoder(**options)


def make_preprocessing_pipeline(include_pagevalues=False):
    """Return a NEW unfitted pipeline; fit it using training rows only.

    log1p is used for nonnegative, right-skewed counts, durations and averages.
    Bounded rates/shares are scaled without logging. Flags stay 0/1.
    PageValues is log-transformed and scaled only when include_pagevalues=True.
    Use both explicit True/False settings on identical data and CV folds for the
    comparison. No clipping, target encoding, selection or oversampling is applied.
    """
    log_numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ])
    bounded_numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    nominal = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_onehot_encoder()),
    ])
    columns = ColumnTransformer([
        ('log', log_numeric, BASE_LOG_COLUMNS + ENGINEERED_LOG_COLUMNS + (['PageValues'] if include_pagevalues else [])),
        ('bounded', bounded_numeric, BOUNDED_COLUMNS),
        ('binary', SimpleImputer(strategy='most_frequent'), BINARY_COLUMNS),
        ('category', nominal, NOMINAL_COLUMNS),
    ], remainder='drop', verbose_feature_names_out=True)
    return Pipeline([('features', SessionFeatureEngineer(include_pagevalues=include_pagevalues)), ('columns', columns)])
