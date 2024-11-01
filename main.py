import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Data Stream Simulation Function
def generate_data_stream(num_points=1):
    """Simulates a data stream with regular patterns, seasonality, and random noise."""
    time = np.arange(num_points)
    signal = np.sin(0.1 * time) + 0.5 * np.random.normal(size=num_points)
    return signal

# Adaptive Anomaly Detection Function with Moving Average
def adaptive_anomaly_detection(data, window_size=20, threshold=3):
    anomalies = []
    if len(data) >= window_size:
        window_data = list(data)[-window_size:]
        mean = np.mean(window_data)
        std_dev = np.std(window_data)
        
        if abs((data[-1] - mean) / std_dev) > threshold:
            anomalies.append((len(data) - 1, data[-1]))
    return anomalies

# Real-Time Data Stream Monitoring with Visualization
def real_time_monitor(window_size=20):
    stream_data = deque(maxlen=window_size)
    anomalies = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Data Stream')
    anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')

    try:
        for _ in range(100):
            new_data = generate_data_stream(1)[0]
            stream_data.append(new_data)

            new_anomalies = adaptive_anomaly_detection(stream_data, window_size=window_size)
            anomalies.extend(new_anomalies)

            ax.clear()
            ax.plot(range(len(stream_data)), stream_data, label='Data Stream')
            
            anomaly_indices = [idx for idx, _ in anomalies]
            anomaly_values = [stream_data[idx - len(stream_data)] for idx in anomaly_indices]
            ax.plot(anomaly_indices, anomaly_values, 'ro', label='Anomalies')

            ax.legend()
            plt.pause(0.1)

    except Exception as e:
        print("Error in real_time_monitor:", e)
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    real_time_monitor()