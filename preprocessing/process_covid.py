"""
Preprocess the covid data to get the cases and time.
"""

# Importing necessary libraries
import jax
import jax.numpy as jnp

def data_smoother(interval, window_size=7):
    """
    Smooth the data using a convolutional window.
    Default window size is 7 (weekly average)
    Args:
        interval: jax array
        window_size: int
    Returns:
        smoothed_data: jax array
    """
    window = jnp.ones(window_size) / window_size
    return jnp.convolve(interval, window, mode='same')

def process_covid_data(data, start_day, end_day):
    """
    Process the covid data to get the cases and time.
    Args:
        data: numpy array
        start_day: int
        end_day: int
    Returns:
        time: jax array
        cases: jax array
        smooth_cases: jax array
    """
    # The first column is cumulative cases, the second is daily cases
    data = jnp.array(data)
    days = jnp.arange(0, data.shape[0])

    time = days[start_day:end_day] - days[start_day]
    cases = data[start_day:end_day, 1]
    smooth_cases = data_smoother(cases)

    return time, cases, smooth_cases