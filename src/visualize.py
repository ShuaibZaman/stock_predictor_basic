import matplotlib.pyplot as plt
import numpy as np

from src.main import predictions, y_test, scaler


def visualize_predictions(predictions, y_test, scaler):
    """
    Visualizes the model's predictions against the actual stock prices.

    :param predictions: Array of predicted stock prices (inverse transformed).
    :param y_test: Test labels (still in normalized form).
    :param scaler: The scaler used for normalization (to inverse transform y_test).
    """
    # Inverse transform y_test to get the original price scale.
    y_test_inversed = scaler.inverse_transform(np.reshape(y_test, (-1, 1)))

    # Create a range for the x-axis (number of test samples)
    x_range = np.arange(len(y_test_inversed))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x_range, y_test_inversed, label='Actual Stock Price', color='blue')
    plt.plot(x_range, predictions, label='Predicted Stock Price', color='red')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction vs Actual')
    plt.legend()
    plt.show()


visualize_predictions(predictions, y_test, scaler)
