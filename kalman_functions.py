# kalman_functions.py

import numpy as np

def kalman_init(x0, P0):
    """
    Initialize the Kalman filter with an initial state estimate (x0) and covariance estimate (P0).

    Args:
    - x0 (float): Initial state estimate.
    - P0 (float): Initial covariance estimate.

    Returns:
    - x (float): Initial state estimate.
    - P (float): Initial covariance estimate.
    """
    return x0, P0

def kalman_predict(x, P, Q):
    """
    Predict the next state given the current state and covariance, along with the process noise covariance.

    Args:
    - x (float): Current state estimate.
    - P (float): Current covariance estimate.
    - Q (float): Process noise covariance.

    Returns:
    - x_pred (float): Predicted state estimate.
    - P_pred (float): Predicted covariance estimate.
    """
    x_pred = x
    P_pred = P + Q
    return x_pred, P_pred

def kalman_update(x_pred, P_pred, measurement, R):
    """
    Update the state estimate given the predicted state and covariance, along with the measurement and measurement noise covariance.

    Args:
    - x_pred (float): Predicted state estimate.
    - P_pred (float): Predicted covariance estimate.
    - measurement (float): Measurement value.
    - R (float): Measurement noise covariance.

    Returns:
    - x_updated (float): Updated state estimate.
    - P_updated (float): Updated covariance estimate.
    """
    K = P_pred / (P_pred + R)
    x_updated = x_pred + K * (measurement - x_pred)
    P_updated = (1 - K) * P_pred
    return x_updated, P_updated
