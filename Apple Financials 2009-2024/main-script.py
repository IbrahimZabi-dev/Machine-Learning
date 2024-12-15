import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import FinancialPredictor

def main():
    # Load data
    loader = DataLoader('data/apple_2009-2024.csv')
    data = loader.load_data()
    data = loader.preprocess_data()
    
    # Feature Engineering
    engineer = FeatureEngineer(data)
    data = engineer.create_features()
    X_train, X_test, y_train, y_test, feature_names = engineer.prepare_ml_dataset()
    
    # Model Development
    predictor = FinancialPredictor(X_train, X_test, y_train, y_test)
    predictor.train_models()
    
    # Evaluate Models
    results = predictor.evaluate_models()
    print("Model Performance Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Mean Squared Error: {metrics['MSE']}")
        print(f"  R2 Score: {metrics['R2']}")
    
    # Plot Results
    predictor.plot_predictions(feature_names)

if __name__ == "__main__":
    main()
