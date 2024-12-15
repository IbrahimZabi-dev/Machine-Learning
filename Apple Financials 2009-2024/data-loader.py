import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        """Load CSV data"""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def clean_currency(self, column):
        """Clean currency columns by removing $ and , """
        return self.data[column].str.replace('$', '').str.replace(',', '').astype(float)
    
    def preprocess_data(self):
        """Preprocess the financial data"""
        # Convert currency columns
        currency_columns = [
            'EBITDA (millions)', 'Revenue (millions)', 
            'Gross Profit (millions)', 'Op Income (millions)', 
            'Net Income (millions)'
        ]
        
        for col in currency_columns:
            self.data[col] = self.clean_currency(col)
        
        return self.data
