import numpy as np

def make_predictions(model, X_test, scaler):
    """
    Use the trained model to make predictions.
    :param model: Trained LSTM model.
    :param X_test: Test features.
    :param scaler: Scaler object for inverse transformation.
    :return: Predicted values.
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions
