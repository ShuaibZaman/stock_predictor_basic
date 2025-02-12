from data_collection import fetch_stock_data
from data_processing import preprocess_data
from train_model import build_model, train_model
from test import make_predictions
from evaluate import evaluate_model

# Parameters
TICKER = 'AAPL'
START_DATE = '2010-01-01'
END_DATE = '2025-01-01'
SEQUENCE_LENGTH = 60

# Step 1: Fetch data
data = fetch_stock_data(TICKER, START_DATE, END_DATE)

# Step 2: Preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data(data, sequence_length=SEQUENCE_LENGTH)

# Step 3: Build and train model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)
model = train_model(model, X_train, y_train)

# Step 4: Test model
predictions = make_predictions(model, X_test, scaler)

# Step 5: Evaluate performance
rmse = evaluate_model(y_test, predictions)
print(f"Model RMSE: {rmse}")

# Step 6: Save model
model.save('../models/stock_predictor.h5')
print("Model saved")
