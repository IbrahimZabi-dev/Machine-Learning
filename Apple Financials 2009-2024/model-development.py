import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialPredictor:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
    
    def train_models(self):
        """Train multiple models"""
        # Linear Regression
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr
        self.predictions['Linear Regression'] = lr.predict(self.X_test)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self.predictions['Random Forest'] = rf.predict(self.X_test)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        self.predictions['Gradient Boosting'] = gb.predict(self.X_test)
    
    def evaluate_models(self):
        """Evaluate model performance"""
        results = {}
        for name, pred in self.predictions.items():
            mse = mean_squared_error(self.y_test, pred)
            r2 = r2_score(self.y_test, pred)
            results[name] = {'MSE': mse, 'R2': r2}
        
        return results
    
    def feature_importance(self, model_name='Random Forest'):
        """Get feature importance"""
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def plot_predictions(self, feature_names):
        """Plot prediction results and feature importance"""
        plt.figure(figsize=(15, 10))
        
        # Actual vs Predicted Plot
        plt.subplot(2, 1, 1)
        plt.scatter(self.y_test, self.predictions['Random Forest'])
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2)
        plt.title('Actual vs Predicted Net Income')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # Feature Importance Plot
        plt.subplot(2, 1, 2)
        importances = self.feature_importance()
        indices = np.argsort(importances)[::-1]
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        plt.savefig('feature_importance_plot.png')
        plt.close()
