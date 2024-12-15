import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.features = None
        self.targets = None
    
    def create_features(self):
        """Create advanced features from financial data"""
        # Calculate year-over-year growth rates
        self.data['Revenue_Growth'] = self.data['Revenue (millions)'].pct_change() * 100
        self.data['Net_Income_Growth'] = self.data['Net Income (millions)'].pct_change() * 100
        self.data['EBITDA_Growth'] = self.data['EBITDA (millions)'].pct_change() * 100
        
        # Profitability ratios
        self.data['Return_on_Assets'] = self.data['Net Income (millions)'] / self.data['Total Assets (millions)'] * 100
        self.data['Profit_Margin'] = self.data['Net Income (millions)'] / self.data['Revenue (millions)'] * 100
        
        # Debt metrics
        self.data['Debt_to_Assets'] = self.data['Long Term Debt (millions)'] / self.data['Total Assets (millions)'] * 100
        
        return self.data
    
    def prepare_ml_dataset(self, target_column='Net Income (millions)', test_size=0.2):
        """Prepare dataset for machine learning"""
        # Drop non-numeric and unnecessary columns
        features = self.data.drop(columns=[target_column, 'year'])
        
        # Prepare target variable
        target = self.data[target_column]
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, features.columns
