#!/usr/bin/env python
# coding: utf-8

import matplotlib
try:
    # Try to set interactive widget backend
    matplotlib.use('module://ipympl.backend_nbagg')
except ImportError:
    print("Warning: ipympl not installed. Falling back to TkAgg backend for interactivity.")
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        print("Warning: TkAgg not available. Falling back to non-interactive inline backend.")
        matplotlib.use('inline')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import datetime
import os
import csv
import sys

# Check if the backend supports interactivity
is_interactive = matplotlib.get_backend().lower() in ['module://ipympl.backend_nbagg', 'tkagg', 'qt5agg', 'qtagg']
if not is_interactive:
    print("Warning: Current Matplotlib backend does not support interactivity. "
          "Mouse movement and click events will not work. "
          "Consider installing ipympl (`pip install ipympl`) and running in Jupyter with `%matplotlib widget`.")

# Set Matplotlib font
matplotlib.rc('font', family='Times New Roman', size=12)

# Load the FITS file for 150.9 MHz
file_150 = 'nrh2_1509_h80_20240529_141500c01_b.fts'
try:
    hdu = fits.open(file_150)
except FileNotFoundError:
    print(f"Error: FITS file '{file_150}' not found. Please check the file path.")
    sys.exit(1)

# Extract data cube
data_nrh_150 = hdu[1].data

# Define image extent in arcseconds
extent = (
    -hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960,  # xmin
    +hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960,  # xmax
    -hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960,  # ymin
    +hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960   # ymax
)

# Extract start time and frequency from file name
file_name = os.path.basename(file_150)
date_time_str = file_name.split('_')[3]  # Extract '20240529'
time_str = file_name.split('_')[4][:6]   # Extract '141500'
frequency_str = file_name.split('_')[1]   # Extract '1509'
frequency = f"{float(frequency_str) / 10} MHz"  # Convert to '150.9 MHz'
start_time = datetime.datetime.strptime(
    f"{date_time_str} {time_str}",
    '%Y%m%d %H%M%S'
)

# Set time step to 250 milliseconds (0.25 seconds)
time_step = 0.25  # seconds
no_images = len(data_nrh_150)

# Initialize output files (create headers if files don't exist)
def initialize_output_files():
    """Initialize CSV and text output files with headers"""
    csv_filename = 'clicked_points.csv'
    txt_filename = 'click_output.txt'
    
    # Initialize CSV file with header if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Session_ID', 'Image_Time', 'Frequency', 'CRPIX1', 'CRPIX2', 'R_SUN', 
                            'X_PIXEL', 'Y_PIXEL', 'HEIGHT', 'PIXEL_VALUE', 'X_ARCSEC', 'Y_ARCSEC'])
    
    # Initialize text file with session header if it doesn't exist
    if not os.path.exists(txt_filename):
        with open(txt_filename, 'w') as f:
            f.write(f"NRH Data Analysis Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    return csv_filename, txt_filename

# Initialize output files
csv_filename, txt_filename = initialize_output_files()

# Generate unique session ID for this run
session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# User-specified times (multiple times)
user_times = [
    '2024-05-29 14:26:04.000',
    '2024-05-29 14:26:51.000',
    '2024-05-29 14:27:04.000',
    '2024-05-29 14:27:40.000',
    '2024-05-29 14:27:53.000',
    '2024-05-29 14:28:03.000',
    '2024-05-29 14:28:12.000',
    '2024-05-29 14:28:16.000',
    '2024-05-29 14:28:35.000',
    '2024-05-29 14:28:48.000',
    '2024-05-29 14:29:17.000',
    '2024-05-29 14:29:30.000',
    '2024-05-29 14:30:42.000',
    '2024-05-29 14:31:01.000',
    '2024-05-29 14:32:13.000',
    '2024-05-29 14:34:22.000',
    '2024-05-29 14:34:35.000',
    '2024-05-29 14:36:26.000',
    '2024-05-29 14:36:45.000',
]

print(f"Session ID: {session_id}")
print(f"Processing {len(user_times)} time points...")
print(f"Output files: {csv_filename}, {txt_filename}")

# Function to handle mouse movement
def create_motion_handler(ax, data, extent, img_time, frequency, fig, title, title_text, session_id, time_idx, total_times, crpix1, crpix2, r_sun):
    def on_motion(event):
        if not is_interactive:
            return
        if event.inaxes != ax:  # Check if cursor is inside the plot
            title.set_text(title_text)
            fig.canvas.draw_idle()
            return
        # Get cursor position in arcseconds
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        # Convert arcseconds to pixel indices
        x_pixel = int((x - extent[0]) / (extent[1] - extent[0]) * (data.shape[1] - 1))
        y_pixel = int((y - extent[2]) / (extent[3] - extent[2]) * (data.shape[0] - 1))
        # Check if pixel indices are within bounds
        if 0 <= x_pixel < data.shape[1] and 0 <= y_pixel < data.shape[0]:
            pixel_value = data[y_pixel, x_pixel]
            # Calculate height in solar radii
            height = np.sqrt((x_pixel - crpix1)**2 + (y_pixel - crpix2)**2) / r_sun
            # Update title with pixel indices, pixel value, and height
            title.set_text(
                f"Session: {session_id} | Time {time_idx + 1}/{total_times}\n"
                f"Image Time: {img_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"Frequency: {frequency}\n"
                f"Pixel: ({x_pixel}, {y_pixel}), Value: {pixel_value:.2e}\n"
                f"Height: {height:.6f} solar radii"
            )
            fig.canvas.draw_idle()
        else:
            # Reset title if cursor is out of bounds
            title.set_text(title_text)
            fig.canvas.draw_idle()
    return on_motion

# Function to handle mouse clicks
def create_click_handler(ax, data, extent, img_time, frequency, crpix1, crpix2, r_sun, fig, title, title_text, session_id):
    def on_click(event):
        if not is_interactive:
            return
        if event.inaxes != ax:  # Check if click is inside the plot
            print("Click outside plot area")
            return
        
        # Get click position in arcseconds
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            print("Invalid click coordinates")
            return
        
        # Convert arcseconds to pixel indices
        x_pixel = int((x - extent[0]) / (extent[1] - extent[0]) * (data.shape[1] - 1))
        y_pixel = int((y - extent[2]) / (extent[3] - extent[2]) * (data.shape[0] - 1))
        
        print(f"Click at (x={x:.2f}, y={y:.2f}) arcsec -> Pixel ({x_pixel}, {y_pixel})")
        
        # Check if pixel indices are within bounds
        if 0 <= x_pixel < data.shape[1] and 0 <= y_pixel < data.shape[0]:
            # Calculate height
            height = np.sqrt((x_pixel - crpix1)**2 + (y_pixel - crpix2)**2) / r_sun
            pixel_value = data[y_pixel, x_pixel]
            
            # Format image time
            img_time_formatted = img_time.strftime('%H:%M:%S.%f')[:-3]
            
            output_text = f"""
{'='*50}
CLICKED POINT INFORMATION:
{'='*50}
Session ID: {session_id}
Image Time: {img_time_formatted}
Frequency: {frequency}
CRPIX1: {crpix1}
CRPIX2: {crpix2}
R_SUN: {r_sun}
X_PIXEL: {x_pixel}
Y_PIXEL: {y_pixel}
HEIGHT: {height:.6f} solar radii
PIXEL_VALUE: {pixel_value:.2e}
COORDINATES: ({x:.2f}, {y:.2f}) arcsec
{'='*50}
"""
            
            # Display output
            print(output_text)
            
            # Force flush stdout
            sys.stdout.flush()
            
            # Save to text file (append mode)
            try:
                with open(txt_filename, 'a') as f:
                    f.write(output_text + '\n')
                
                # Save to CSV file (append mode)
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        session_id,
                        img_time_formatted,
                        frequency,
                        crpix1, crpix2, r_sun,
                        x_pixel, y_pixel, f"{height:.6f}",
                        f"{pixel_value:.2e}", f"{x:.2f}", f"{y:.2f}"
                    ])
                
                print(f"✓ Data saved to files: {csv_filename}, {txt_filename}")
                
                # Count total clicks in current session
                try:
                    with open(csv_filename, 'r') as f:
                        session_clicks = sum(1 for line in f if session_id in line)
                    print(f"✓ Total clicks in current session: {session_clicks}")
                except:
                    print("✓ Data saved successfully")
                
            except Exception as e:
                print(f"Error saving data: {e}")
                import traceback
                traceback.print_exc()
            
            # Update title with detailed info
            detailed_title = (
                f"Session: {session_id}\n"
                f"Image Time: {img_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"Frequency: {frequency}\n"
                f"CRPIX1:{crpix1} CRPIX2:{crpix2} R_SUN:{r_sun:.2f}\n"
                f"X_PIXEL:{x_pixel} Y_PIXEL:{y_pixel} HEIGHT:{height:.6f}"
            )
            title.set_text(detailed_title)
            
            # Mark the clicked point
            ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2, label='Clicked Point' if not any('Clicked Point' in str(line.get_label()) for line in ax.lines) else '')
            
            # Update legend if needed
            if not ax.legend_ or len(ax.lines) == 1:
                ax.legend()
            
            fig.canvas.draw_idle()
        else:
            print(f"Click out of bounds: Pixel ({x_pixel}, {y_pixel}) not in data shape {data.shape}")
    
    return on_click

# Process each user-specified time
for time_idx, user_time_str in enumerate(user_times):
    try:
        user_time = datetime.datetime.strptime(user_time_str, '%Y-%m-%d %H:%M:%S.%f')
        
        # Calculate the image index
        time_diff = (user_time - start_time).total_seconds()
        img_no = int(time_diff / time_step)

        # Validate the image index
        if img_no < 0 or img_no >= no_images:
            print(f"Error: The specified time {user_time_str} is out of range for the data cube.")
            continue

        print(f"\nProcessing time {time_idx + 1}/{len(user_times)}: {user_time_str}")

        # Create the plot
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot2grid((1, 1), (0, 0))
        data = data_nrh_150[img_no][2]
        
        # Check if data is valid
        if data.size == 0:
            print("Error: Data array is empty")
            continue
        
        # Compute maximum intensity and its location
        max_NRH = np.max(data)
        y_pixel, x_pixel = np.unravel_index(np.argmax(data), data.shape)
        # Calculate height in solar radii
        crpix1 = hdu[1].header['CRPIX1']
        crpix2 = hdu[1].header['CRPIX2']
        r_sun = hdu[1].header['SOLAR_R']
        height = np.sqrt((x_pixel - crpix1)**2 + (y_pixel - crpix2)**2) / r_sun
        # Convert to arcseconds
        x_arcsec = extent[0] + (extent[1] - extent[0]) * x_pixel / (data.shape[1] - 1)
        y_arcsec = extent[2] + (extent[3] - extent[2]) * y_pixel / (data.shape[0] - 1)
        print(f"Maximum intensity (max_NRH): {max_NRH:.2e}")
        print(f"Pixel location of maximum intensity: (x={x_pixel}, y={y_pixel})")
        print(f"Height of maximum intensity: {height:.6f} solar radii")
        print(f"Arcsecond coordinates of maximum intensity: (x={x_arcsec:.2f}, y={y_arcsec:.2f}) arcsec")
        
        # Plot contours
        level = [0.5 * max_NRH, 0.95 * max_NRH]
        contour = plt.contour(data, levels=level, colors='darkviolet', extent=extent)
        
        # Add a circle representing one solar diameter (radius = 960 arcseconds)
        kk = 960  # Solar radius in arcseconds (diameter = 1920 arcseconds)
        circle = plt.Circle((0, 0), kk, color='red', fill=False, linewidth=2)
        ax.add_artist(circle)

        # Mark the maximum intensity point
        ax.plot(x_arcsec, y_arcsec, 'g*', markersize=15, label='Max Intensity')
        
        img_time = start_time + datetime.timedelta(seconds=img_no * time_step)
        img_time_str = img_time.strftime('%Y%m%d_%H%M%S_%f')[:19]
        title_text = (
            f"Session: {session_id} | Time {time_idx + 1}/{len(user_times)}\n"
            f"Image Time: {img_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
            f"Frequency: {frequency}"
        )
        title = plt.title(title_text)
        plt.xlabel('X (arcseconds)')
        plt.ylabel('Y (arcseconds)')
        ax.set_aspect('equal')
        ax.legend()

        # Connect event handlers only if interactive
        if is_interactive:
            # Connect the motion event handler
            motion_handler = create_motion_handler(ax, data, extent, img_time, frequency, fig, title, title_text, session_id, time_idx, len(user_times), crpix1, crpix2, r_sun)
            fig.canvas.mpl_connect('motion_notify_event', motion_handler)

            # Connect the click event handler
            click_handler = create_click_handler(ax, data, extent, img_time, frequency, crpix1, crpix2, r_sun, fig, title, title_text, session_id)
            fig.canvas.mpl_connect('button_press_event', click_handler)
        else:
            print("Interactive features disabled due to non-interactive backend.")

        # Print initial header information
        print("="*50)
        print("FITS HEADER INFORMATION:")
        print("="*50)
        print(f"CRPIX1: {crpix1}")
        print(f"CRPIX2: {crpix2}")
        print(f"SOLAR_R: {r_sun}")
        print(f"Data shape: {data.shape}")
        print(f"Image index: {img_no}")
        print(f"Unit of data: {hdu[1].header.get('BUNIT', 'Unknown')}")
        print("="*50)
        if is_interactive:
            print("Click on the plot to get point information!")
        else:
            print("Plot is non-interactive; click information not available.")
        print("="*50)

        # Save the plot
        plt.savefig(f'nrh_{frequency_str}_{img_time_str}_contour_{session_id}.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Error processing time {user_time_str}: {e}")
        continue

print(f"\nSession {session_id} complete!")
print(f"All click data saved to:")
print(f"  - CSV: {csv_filename}")
print(f"  - Text: {txt_filename}")

# Close the FITS file
hdu.close()