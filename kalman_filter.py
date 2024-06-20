import numpy as np
from kalman_functions import kalman_init, kalman_predict, kalman_update
import time

def kalman_filter(voice_data, Q, R, timeout=60):
    """
    Apply Kalman filter to voice data.

    Args:
        voice_data (ndarray): Input voice data.
        Q (float): Process noise covariance.
        R (float): Measurement noise covariance.
        timeout (float): Timeout limit for filtering process in seconds. Default is 60 seconds.

    Returns:
        ndarray: Filtered voice data.
    """
    filtered_voice = []

    # Initialize Kalman filter
    x, P = kalman_init(voice_data[0], 1)

    # Set start time
    start_time = time.time()

    # Kalman filtering loop
    for measurement in voice_data:
        # Check for timeout
        if time.time() - start_time > timeout:
            raise TimeoutError("Kalman filter process timed out")

        # Prediction
        x_pred, P_pred = kalman_predict(x, P, Q)

        # Update
        try:
            x, P = kalman_update(x_pred, P_pred, measurement, R)
            filtered_voice.append(x)
        except Exception as e:
            print(f"Error during Kalman filter update: {e}")
            break  # Break the loop if an error occurs

    return np.array(filtered_voice)
