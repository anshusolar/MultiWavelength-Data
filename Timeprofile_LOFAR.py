import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.time import Time, TimeDelta
import os

def load_lofar_data(fits_file, f_min, f_max, t_start=None, t_end=None):
    """Load and process LOFAR Stokes I data with time and frequency zooming."""
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            data = hdul[0].data.astype(np.float32)
            if len(data.shape) != 2:
                raise ValueError(f"Expected 2D data array, got shape {data.shape}")
            # Extract time information from header
            date_obs = header.get('DATE-OBS', '2024/05/29')
            time_obs = header.get('TIME-OBS', '00:00:00.000000')
            file_t_start = Time(f"{date_obs.replace('/', '-')[:10]}T{time_obs[:8]}", format='isot')
            time_step = header.get('CDELT1', 0.25)  # Seconds
            n_times = data.shape[0]
            times_sec = np.arange(n_times) * time_step
            times = file_t_start + TimeDelta(times_sec, format='sec')
            file_t_end = file_t_start + TimeDelta((n_times - 1) * time_step, format='sec')
            # Extract frequencies
            table = hdul[1].data
            freqs = table[0][0]  # Frequencies in MHz
            # Process data for spectrum (normalized, log-transformed)
            data = data.T  # Transpose to (freqs, times)
            data_clipped = np.clip(data, 1e-3, None)
            log_data = np.log10(data_clipped)
            median_spectrum = np.nanmedian(log_data, axis=1, keepdims=True)
            data_processed = log_data - median_spectrum
            # Apply time and frequency masks
            freq_mask = (freqs >= f_min) & (freqs <= f_max)
            freqs_zoom = freqs[freq_mask]
            data_spectrum = data_processed[freq_mask, :]  # For spectrum
            data_raw = data[freq_mask, :]  # Raw data for time profile
            if t_start is not None and t_end is not None:
                t_start = Time(t_start, format='isot')
                t_end = Time(t_end, format='isot')
                time_mask = (times >= t_start) & (times <= t_end)
                times_zoom = times[time_mask]
                data_spectrum = data_spectrum[:, time_mask]
                data_raw = data_raw[:, time_mask]
            else:
                times_zoom = times
                t_start, t_end = file_t_start, file_t_end
            print(f"File: {fits_file}")
            print(f"Time range: {t_start} to {t_end}, {len(times_zoom)} times")
            print(f"Freq range: {f_min} to {f_max}, {len(freqs_zoom)} freqs")
            print(f"Spectrum data shape: {data_spectrum.shape}")
            return times_zoom, freqs_zoom, data_spectrum, data_raw, (t_start, t_end)
    except Exception as e:
        print(f"Error processing LOFAR FITS file {fits_file}: {e}")
        return None, None, None, None, None

def combine_lofar_files(lofar_files, f_min, f_max, zoom_time_range=None):
    """Combine data from multiple LOFAR files with optional time zooming."""
    all_times, all_spectrum_data, all_raw_data, all_time_ranges = [], [], [], []
    freqs = None
    for file in lofar_files:
        t_start = zoom_time_range[0] if zoom_time_range else None
        t_end = zoom_time_range[1] if zoom_time_range else None
        times, freqs_file, data_spectrum, data_raw, time_range = load_lofar_data(file, f_min, f_max, t_start, t_end)
        if times is not None and len(times) > 0:
            all_times.append(times)
            all_spectrum_data.append(data_spectrum)
            all_raw_data.append(data_raw)
            all_time_ranges.append(time_range)
            if freqs is None:
                freqs = freqs_file
            elif not np.allclose(freqs, freqs_file, rtol=1e-3):
                print(f"Warning: Frequency mismatch in {file}")
    if not all_times:
        raise ValueError("No valid LOFAR data loaded from any file.")
    # Combine times and data
    combined_times = np.concatenate([t.datetime for t in all_times])
    combined_times = Time(combined_times, format='datetime')
    combined_spectrum_data = np.concatenate(all_spectrum_data, axis=1)
    combined_raw_data = np.concatenate(all_raw_data, axis=1)
    # Sort by time to ensure chronological order
    sort_idx = np.argsort(combined_times)
    combined_times = combined_times[sort_idx]
    combined_spectrum_data = combined_spectrum_data[:, sort_idx]
    combined_raw_data = combined_raw_data[:, sort_idx]
    # Combine time ranges
    t_start = min(tr[0] for tr in all_time_ranges)
    t_end = max(tr[1] for tr in all_time_ranges)
    if zoom_time_range:
        t_start = Time(zoom_time_range[0], format='isot')
        t_end = Time(zoom_time_range[1], format='isot')
        time_mask = (combined_times >= t_start) & (combined_times <= t_end)
        combined_times = combined_times[time_mask]
        combined_spectrum_data = combined_spectrum_data[:, time_mask]
        combined_raw_data = combined_raw_data[:, time_mask]
    print(f"Combined time range: {t_start} to {t_end}")
    print(f"Combined spectrum data shape: {combined_spectrum_data.shape}")
    print(f"Combined raw data shape: {combined_raw_data.shape}")
    return combined_times, freqs, combined_spectrum_data, combined_raw_data, (t_start, t_end)

def running_mean(data, window_size):
    """Compute running mean (moving average) for smoothing."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_zoomed_lofar_time_profile(times, freqs, data_raw, time_range, freq_centers=[30, 50, 70], y_range=None):
    """Plot zoomed LOFAR time profile for multiple narrow frequency ranges (±1 MHz) with normalized running mean."""
    if times is None or freqs is None or data_raw is None or len(times) == 0 or len(freqs) == 0:
        raise ValueError("Invalid or empty data: cannot plot time profile.")
    
    fig = plt.figure(figsize=(12, 4), dpi=100)
    ax = fig.add_subplot(111)
    time_edges = mdates.date2num(times.datetime)
    if len(time_edges) > 1:
        time_edges = np.concatenate([time_edges, [time_edges[-1] + (time_edges[-1] - time_edges[-2])]])
    else:
        time_edges = np.concatenate([time_edges, [time_edges[0] + 0.25 / 86400]])
    
    # Plot time profile for each frequency center using raw data with running mean and normalization
    colors = ['black', 'blue', 'red', 'green', 'purple']  # Colors for multiple lines
    window_size = 5  # Window size for running mean
    for i, freq_center in enumerate(freq_centers):
        # Narrow frequency range: ±1 MHz around freq_center
        freq_range = (freq_center - 1, freq_center + 1)
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in range {freq_range[0]}–{freq_range[1]} MHz")
            time_profile = np.zeros(len(times))
        else:
            time_profile = np.nanmean(data_raw[freq_mask, :], axis=0)
            # Apply running mean to the time profile
            time_profile = running_mean(time_profile, window_size)
            # Normalize by maximum value
            time_profile = time_profile / np.max(time_profile)
            # Adjust time array to match running mean output length
            trim = (window_size - 1) // 2
            time_edges_trimmed = time_edges[:-1][trim:-trim] if trim > 0 else time_edges[:-1]
        ax.plot(time_edges_trimmed, time_profile, color=colors[i % len(colors)], 
                label=f'{freq_center:.0f} MHz')  # Use center frequency in legend
    
    ax.set_xlabel(f'Time (UT, {time_range[0].strftime("%Y-%m-%d")})')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('LOFAR Time Profile ')
    ax.legend()
    #ax.grid(True, linestyle='--', alpha=0.7)
    # Automatic x-axis ticks in HH:MM, rotated 90 degrees
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlim(mdates.date2num(time_range[0].datetime), mdates.date2num(time_range[1].datetime))
    if y_range:
        ax.set_ylim(y_range[0], y_range[1])
    plt.tight_layout()
    plt.savefig('lofar_time_profile.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

# Main Execution
lofar_dir = 'D:/USO/TIME_PROFILE/lofar_fits_files'  # Specify your directory if needed
lofar_files = [
    'LOFAR_20240529_140500_LBA_OUTER_S0.fits',
    'LOFAR_20240529_142000_LBA_OUTER_S0.fits',
    'LOFAR_20240529_143500_LBA_OUTER_S0.fits',
    'LOFAR_20240529_145000_LBA_OUTER_S0.fits',
    'LOFAR_20240529_150500_LBA_OUTER_S0.fits'
]
lofar_files = [os.path.join(lofar_dir, f) for f in lofar_files]
f_min, f_max = 25, 85  # LOFAR frequency range
freq_centers = [30, 50, 70]  # Central frequencies for time profile (averages ±1 MHz)
zoom_time_range = ('2024-05-29T14:20:00', '2024-05-29T15:00:00')  # Zoomed time range
y_range_profile = None  # Zoomed y-axis range for time profile (None for auto-scaling)

# Verify files exist
existing_files = [f for f in lofar_files if os.path.exists(f)]
if not existing_files:
    raise ValueError(f"No LOFAR FITS files found: {lofar_files}")

# Combine and process LOFAR data with zoomed time range
times, freqs, _, raw_data, time_range = combine_lofar_files(existing_files, f_min, f_max, zoom_time_range)

# Plot zoomed time profile
plot_zoomed_lofar_time_profile(times, freqs, raw_data, time_range, freq_centers, y_range_profile)