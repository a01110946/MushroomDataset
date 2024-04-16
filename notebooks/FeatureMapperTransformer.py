from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from .utils import get_feature_names_out_with_mapping

class FeatureMapperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_transformer, categorical_transformer, numeric_features, categorical_features):
        self.numeric_transformer = numeric_transformer
        self.categorical_transformer = categorical_transformer
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.feature_mapping = None

    def fit(self, X, y=None):
        ct = ColumnTransformer([
            ('numeric', self.numeric_transformer, self.numeric_features),
            ('categorical', self.categorical_transformer, self.categorical_features)
        ])
        ct.fit(X)  # Fit the ColumnTransformer
        self.feature_mapping = get_feature_names_out_with_mapping(ct)  # Generate mapping
        return self

    def transform(self, X, y=None):
        ct = ColumnTransformer([
            ('numeric', self.numeric_transformer, self.numeric_features),
            ('categorical', self.categorical_transformer, self.categorical_features)
        ])
        return ct.transform(X)