import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from datetime import datetime, timedelta
from glob import glob

# Function to load data from a single Callisto FITS file
def load_fits_data(file):
    with fits.open(file) as hdu:
        data = hdu[0].data.astype(np.float32)
        hdr = hdu[0].header
        t_start = hdr['CRVAL1']
        dt = hdr['CDELT1']
        nt = hdr['NAXIS1']
        time_sec = t_start + dt * np.arange(nt)
        date_str = hdr['DATE-OBS']
        base_time = datetime.strptime(date_str, "%Y/%m/%d")
        time_axis = [base_time + timedelta(seconds=float(s)) for s in time_sec]
        freqs = hdu[1].data['Frequency'][0]
    return time_axis, freqs, data

# Function to combine multiple FITS files and prepare data for plotting
def process_multiple_fits(file_list, frequencies):
    data_list, time_list = [], []
    freqs = None

    # Load data from each file
    for file in file_list:
        t, f, d = load_fits_data(file)
        time_list.append(t)
        data_list.append(d)
        if freqs is None:
            freqs = f  # Assume all files have the same frequency channels
        else:
            assert np.allclose(freqs, f), f"Frequency mismatch in {file}"

    # Flip frequency axis and data (skip first 6 channels)
    freq_flipped = np.flip(freqs)[6:]
    data_flipped = [np.flip(d, axis=0)[6:, :] for d in data_list]

    # Combine time and data arrays
    time_combined = np.concatenate(time_list)
    data_combined = np.hstack(data_flipped)
    

    # Subtract median to remove background
    #median_spectrum = np.nanmedian(data_combined, axis=1, keepdims=True)
    #data_subtracted = data_combined - median_spectrum
    data_subtracted = data_combined
    data_subtracted =  10**(data_subtracted/10)
    median_spectrum = np.nanmin(data_subtracted, axis=1, keepdims=True)
    data_subtracted = data_subtracted/median_spectrum

    
    # Filter time range (14:15:00 to 15:00:00 UT)
    start_time = datetime(2024, 5, 29, 14, 20, 0)
    end_time = datetime(2024, 5, 29, 15, 0, 0)
    time_num = mdates.date2num(time_combined)
    mask = (time_num >= mdates.date2num(start_time)) & (time_num <= mdates.date2num(end_time))
    time_combined = time_combined[mask]
    data_subtracted = data_subtracted[:, mask]

    return time_combined, freq_flipped, data_subtracted

# Function to compute running mean (moving average)
def running_mean(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to plot smoothed time profiles with running mean
def plot_time_profile(time_combined, freq_combined, data_subtracted, frequencies, window_size=5):
    plt.figure(figsize=(12, 4), dpi=100)
    
    for freq in frequencies:
        # Find the closest frequency in freq_combined
        idx = np.argmin(np.abs(freq_combined - freq))
        closest_freq = freq_combined[idx]
        print(f"Plotting smoothed time profile for {closest_freq:.2f} MHz (closest to {freq:.2f} MHz)")
        
        # Smooth by averaging over 5 channels (idx-2 to idx+2)
        start_index = max(0, idx - 2)
        end_index = min(data_subtracted.shape[0], idx + 3)  # +3 to include idx+2
        smoothed_time_series = np.nanmean(data_subtracted[start_index:end_index, :], axis=0)
        
        # Apply running mean to the smoothed time series
        running_mean_series = running_mean(smoothed_time_series, window_size)
        # Adjust time array to match the length of running mean output
        trim = (window_size - 1) // 2
        time_trimmed = time_combined[trim:-trim] if trim > 0 else time_combined
        running_mean_series1 =  10**(running_mean_series/10)
        # Plot the running mean smoothed time series
        plt.plot(time_trimmed, running_mean_series/np.max(running_mean_series), label=f"{closest_freq:.2f} MHz ")

    # Format the plot
    plt.xlabel('Time [UT, 2024-05-29]')
    plt.ylabel('Normalized Intensity')
    plt.title('Time Profile (Callisto Data - Alexandria)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    #plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.savefig('callisto_time_profile.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage for the specific files
file_list = [
    "UDAIPUR_20240529/EGYPT-Alexandria_20240529_141500_02.fit.gz",
    "UDAIPUR_20240529/EGYPT-Alexandria_20240529_143000_02.fit.gz",
    "UDAIPUR_20240529/EGYPT-Alexandria_20240529_144501_02.fit.gz",
    "UDAIPUR_20240529/EGYPT-Alexandria_20240529_150000_02.fit.gz"
]
frequencies = [110, 135]
time_combined, freq_combined, data_subtracted = process_multiple_fits(file_list, frequencies)
plot_time_profile(time_combined, freq_combined, data_subtracted, frequencies, window_size=5)