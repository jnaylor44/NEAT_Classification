import pandas as pd
import numpy as np
import copy
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        return X
    
class DataPreprocessor:
    def __init__(self, df):
        self.label_encoders = {}
        self.scalers = {}
        self.numerical_columns = []
        self.categorical_columns = []
        self.outlier_remover = OutlierRemover()
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                self.numerical_columns.append(column)
            else:
                self.categorical_columns.append(column)
        
        numerical_data = df[self.numerical_columns].copy()
        categorical_data = df[self.categorical_columns].copy()
        
        if self.numerical_columns:
            self.numerical_imputer.fit(numerical_data)
            numerical_imputed = self.numerical_imputer.transform(numerical_data)
            self.outlier_remover.fit(pd.DataFrame(numerical_imputed, columns=self.numerical_columns))
            numerical_imputed_df = self.outlier_remover.transform(numerical_imputed)
            numerical_imputed_df = pd.DataFrame(numerical_imputed, 
                                               columns=self.numerical_columns, 
                                               index=df.index)

        if self.categorical_columns:
            self.categorical_imputer.fit(categorical_data)
            categorical_imputed = self.categorical_imputer.transform(categorical_data)
            categorical_imputed_df = pd.DataFrame(categorical_imputed, 
                                                 columns=self.categorical_columns, 
                                                 index=df.index)

        
        for column in self.numerical_columns:
            self.scalers[column] = StandardScaler()
            self.scalers[column].fit(numerical_imputed_df[[column]])

        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.label_encoders[column].fit(categorical_imputed_df[column].astype(str))

    def transform_row(self, row):
        transformed = []
        row_df = pd.DataFrame([row])
        numerical_data = row_df[self.numerical_columns]
        categorical_data = row_df[self.categorical_columns]
        
        if self.numerical_columns:
            numerical_imputed = self.numerical_imputer.transform(numerical_data)
            numerical_imputed_df = pd.DataFrame(numerical_imputed, 
                                               columns=self.numerical_columns)
        
        if self.categorical_columns:
            categorical_imputed = self.categorical_imputer.transform(categorical_data)
            categorical_imputed_df = pd.DataFrame(categorical_imputed, 
                                                 columns=self.categorical_columns)
        
        for column in row.index:
            if column in self.categorical_columns:
                imputed_value = categorical_imputed_df[column].iloc[0]
                value = self.label_encoders[column].transform([str(imputed_value)])[0]
                max_val = len(self.label_encoders[column].classes_) - 1
                transformed.append(value / max_val if max_val > 0 else 0)
            elif column in self.numerical_columns:
                imputed_value = numerical_imputed_df[column].iloc[0]
                value_df = pd.DataFrame({column: [float(imputed_value)]})
                scaled_value = self.scalers[column].transform(value_df)[0][0]
                transformed.append(scaled_value)
        
        return transformed