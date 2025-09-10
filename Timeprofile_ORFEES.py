import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from datetime import datetime, timedelta

# Load the ORFEES FITS file
file = 'int_orf20240529_140000_0.1.fts'
hdul = fits.open(file)

# Extract observation start time from header
date_obs = hdul[0].header['DATE-OBS']  # e.g., '2024-05-29'
time_obs = hdul[0].header['TIME-OBS']  # e.g., '13:59:59:980'

# Parse TIME-OBS with milliseconds
time_parts = time_obs.split(':')
if len(time_parts) == 4:  # Format is HH:MM:SS:MMM (milliseconds)
    corrected_time = f"{time_parts[0]}:{time_parts[1]}:{time_parts[2]}.{time_parts[3]}"
    start_time = datetime.strptime(f"{date_obs} {corrected_time}", "%Y-%m-%d %H:%M:%S.%f")
else:
    start_time = datetime.strptime(f"{date_obs} {time_obs}", "%Y-%m-%d %H:%M:%S")
print(f"Start time (TIME-OBS): {start_time}")

# Extract frequency data for Band 1
freq_data = hdul[1].data
freq_b1 = freq_data['FREQ_B1'][0]  # Extract the frequency array
print(f"Band 1 frequencies shape: {freq_b1.shape}")
print(f"Band 1 frequencies: {freq_b1[:5]} to {freq_b1[-5:]} (total {len(freq_b1)})")

# Extract time and intensity data for Band 1
time_b1 = hdul[2].data['TIME_B1']  # Raw time values (assumed milliseconds)
print(f"Raw TIME_B1 sample (ms): {time_b1[:5]}, ..., {time_b1[-5:]}")
print(f"Raw TIME_B1 range (ms): {time_b1.min()} to {time_b1.max()}")

# Convert milliseconds to seconds and adjust to midnight UTC
time_sec = time_b1 / 1000.0  # Convert milliseconds to seconds
print(f"Converted TIME_B1 sample (s): {time_sec[:5]}, ..., {time_sec[-5:]}")
print(f"Converted TIME_B1 range (s): {time_sec.min()} to {time_sec.max()}")

# Convert time in seconds to UTC datetime list from midnight
base_time = datetime(2024, 5, 29, 0, 0, 0)  # Midnight UTC
utc_times = [base_time + timedelta(seconds=float(t)) for t in time_sec]
print(f"UTC time range: {min(utc_times)} to {max(utc_times)}")

# Extract and transpose Stokes I data for Band 1
stokesi_b1 = hdul[2].data['STOKESI_B1'].T  # Transpose to (freq, time) shape
print(f"STOKESI_B1 shape: {stokesi_b1.shape}")

# Manually define desired frequencies (in MHz)
desired_frequencies = [173.2, 228.0,270.6]  # Example: user-specified frequencies
freq_indices = [np.argmin(np.abs(freq_b1 - f)) for f in desired_frequencies]
selected_frequencies = [freq_b1[idx] for idx in freq_indices]
print(f"Desired frequencies: {desired_frequencies}")
print(f"Closest available frequencies: {[f'{f:.2f}' for f in selected_frequencies]}")

# Subtract median to remove background
median_spectrum = np.nanmedian(stokesi_b1, axis=1, keepdims=True)
data_subtracted = stokesi_b1 - median_spectrum

# Define time filter range
start_filter = datetime(2024, 5, 29, 14, 20, 0)  # 14:15:00 UT
end_filter = datetime(2024, 5, 29, 15, 20, 0)      # 15:00:00 UT

# Apply time filter
time_num = mdates.date2num(utc_times)
mask = (time_num >= mdates.date2num(start_filter)) & (time_num <= mdates.date2num(end_filter))
filtered_times = np.array(utc_times)[mask]
data_filtered = data_subtracted[:, mask]  # Filter along time axis

print(f"Filtered time range: {min(filtered_times) if len(filtered_times) > 0 else 'None'} to {max(filtered_times) if len(filtered_times) > 0 else 'None'}")
print(f"Number of selected time samples: {np.sum(mask)}")

# Function to compute running mean (moving average)
def running_mean(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plot smoothed time profiles with running mean
if len(filtered_times) > 0:
    plt.figure(figsize=(12, 4),dpi=100)
    window_size = 5
    for idx, freq in zip(freq_indices, selected_frequencies):
        print(f"Plotting smoothed time profile for {freq:.2f} MHz")
        # Smooth by averaging over 5 channels (idx-2 to idx+2)
        start_index = max(0, idx - 2)
        end_index = min(data_filtered.shape[0], idx + 3)  # +3 to include idx+2
        smoothed_time_series = np.nanmean(data_filtered[start_index:end_index, :], axis=0)
        
        # Apply running mean to the smoothed time series
        running_mean_series = running_mean(smoothed_time_series, window_size)
        # Adjust time array to match the length of running mean output
        trim = (window_size - 1) // 2
        time_trimmed = filtered_times[trim:-trim] if trim > 0 else filtered_times
        
        # Plot the running mean smoothed time series
        plt.plot(time_trimmed, running_mean_series/np.max(running_mean_series), label=f"{freq:.2f} MHz")

    # Format the plot
    plt.xlabel('Time [UT,2024-05-29]')
    plt.ylabel('Normalized Intensity ')
    plt.title('ORFEES Time Profile')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    #plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.savefig('orfees_time_profile.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data available in the specified time range (14:15:00–15:00:00 UT).")

# Close the FITS file
hdul.close()