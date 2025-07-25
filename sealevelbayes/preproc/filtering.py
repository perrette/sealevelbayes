from scipy.signal import butter, sosfiltfilt, filtfilt, savgol_filter

def filter_timeseries(values, frequency=30, order=3, lowpass=True, sos=True):
    """Apply Butterworth filter

    Parameters
    ----------
    values: numpy array
        time series to filter (yearly data points)
    frequency: float
        cutoff frequency in years (default=30)
    order: int
        order of the filter (default=3)
    lowpass: bool
        if True, apply a low-pass filter, otherwise apply a high-pass filter
    
    Source
    ------
    Suggested by co-pilot, but references here:
    https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    """
    cutoff_frequency = 1/frequency / 0.5  # Normalized cutoff frequency (1/30 years, normalized by Nyquist frequency)
    if sos:
        sos_f = butter(N=order, Wn=cutoff_frequency, btype='low' if lowpass else 'high', output="sos")  # Design a 3rd order low-pass Butterworth filter
        filtered = sosfiltfilt(sos_f, values)  # Apply the filter
    else:
        b, a = butter(N=order, Wn=cutoff_frequency, btype='low' if lowpass else 'high')  # Design a 3rd order low-pass Butterworth filter
        filtered = filtfilt(b, a, values)  # Apply the filter
    return filtered
