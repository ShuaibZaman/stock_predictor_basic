from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(y_test, predictions):
    """
    Evaluate the model's performance using RMSE.
    :param y_test: True values.
    :param predictions: Predicted values.
    :return: RMSE score.
    """
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse
