import pandas as pd
from sklearn.model_selection import train_test_split
from ml_models import train_and_save_models, load_models_and_predict
from generate_buy_signals import generate_buy_signals
from generate_sell_signals import generate_sell_signals
from backtest import backtest
from optimize import optimize
from get_strategies import get_strategies
from set_params import set_params

# Paths to the datasets
train_paths = {
    '5m': '/data/aapl_5m_train.csv',
    '1m': '/data/aapl_1m_train.csv',
    '1h': '/data/aapl_1h_train.csv',
    '1d': '/data/aapl_1d_train.csv'
}

test_paths = {
    '5m': '/data/aapl_5m_test.csv',
    '1m': '/data/aapl_1m_test.csv',
    '1h': '/data/aapl_1h_test.csv',
    '1d': '/data/aapl_1d_test.csv'
}

# Function to load and preprocess data (including adding technical indicators)
def load_and_preprocess(path):
    data = pd.read_csv(path)
    # Add preprocessing steps here (e.g., dropping unnecessary columns, adding technical indicators)
    return data

datasets = {interval: load_and_preprocess(path) for interval, path in train_paths.items()}

# Example for a single dataset
interval = '5m'  # Example: Use 5-minute interval data
data = datasets[interval]

# Define features and target variable
features_columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Add your technical indicators to this list
target_column = 'Buy'  # Assume this column is already created during preprocessing

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data[features_columns], data[target_column], test_size=0.2, random_state=42)

# Specify where to save trained models
model_paths = {'svc': '/path/to/svc_model.joblib', 'xgb': '/path/to/xgb_model.joblib', 'lr': '/path/to/lr_model.joblib'}

# Train and save models
train_and_save_models(data, features_columns, target_column, model_paths)

# Load test dataset for the chosen interval
test_data = pd.read_csv(test_paths[interval])

# Generate buy signals
buy_signals = generate_buy_signals(test_data, features_columns, model_paths)

# Generate sell signals
sell_signals = generate_sell_signals(test_data, features_columns, model_paths)

# Perform backtesting
initial_cash = 10000  # Starting cash
results = backtest(test_data, buy_signals, sell_signals, initial_cash=initial_cash)

# Print results
print(f"Final Portfolio Value: {results['final_portfolio_value']}")
print(f"Total Return: {results['total_return']}")
