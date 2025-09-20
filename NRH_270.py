#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import datetime
import os

# Set Matplotlib font
matplotlib.rc('font', family='Times New Roman', size=12)

# Load the FITS file for 270.6 MHz
file_2706 = 'nrh2_2706_h80_20240529_141500c01_b.fts'
hdu = fits.open(file_2706)

# Extract data cube
data_nrh_2706 = hdu[1].data

# Define image extent in arcseconds
extent = (
    -hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960,  # xmin
    +hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960,  # xmax
    -hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960,  # ymin
    +hdu[1].header['CRPIX1'] / hdu[1].header['SOLAR_R'] * 960   # ymax
)

# Extract start time and frequency from file name
file_name = os.path.basename(file_2706)  # Get the file name without path
date_time_str = file_name.split('_')[3]  # Extract '20240529'
time_str = file_name.split('_')[4][:6]   # Extract '141500' from '141500c01'
frequency_str = file_name.split('_')[1]   # Extract '2706' from 'nrh2_2706_h80'
frequency = f"{float(frequency_str) / 10} MHz"  # Convert '2706' to '270.6 MHz'
start_time = datetime.datetime.strptime(
    f"{date_time_str} {time_str}",
    '%Y%m%d %H%M%S'
)

time_step = 1.0  # seconds (changed to 1 second)
image_interval = 4  # Since 1 second = 4 images (0.25 seconds each)
no_images = len(data_nrh_2706)

# Define resume time (2024-05-29 14:22:40.000)
resume_time = datetime.datetime.strptime('20240529 142240.000', '%Y%m%d %H%M%S.%f')

# Define end time (example: 2024-05-29 14:30:00.000)
end_time = datetime.datetime.strptime('20240529 143700.000', '%Y%m%d %H%M%S.%f')

# Calculate the starting and ending image indices
time_diff_start = (resume_time - start_time).total_seconds()
start_img_no = int(time_diff_start / 0.25)  # Image index based on 0.25s intervals

time_diff_end = (end_time - start_time).total_seconds()
end_img_no = min(int(time_diff_end / 0.25), no_images - 1)  # Ensure it doesn't exceed no_images

# Ensure start_img_no and end_img_no are within valid range
if start_img_no < 0 or start_img_no >= no_images:
    print(f"Error: Resume time {resume_time} is out of range for the data cube.")
    hdu.close()
    exit()

if end_img_no < start_img_no:
    print(f"Error: End time {end_time} is before resume time {resume_time}.")
    hdu.close()
    exit()

# Ensure output directory exists
output_dir = r'D:\USO\Nadiya\nrh2_png\nrh2_2706'
os.makedirs(output_dir, exist_ok=True)

# Loop through images from start_img_no to end_img_no, stepping by 4 (1 second)
for img_no in range(start_img_no, end_img_no + 1, image_interval):
    plt.figure(figsize=(6, 6))  # Create a new figure for each image
    ax = plt.subplot2grid((1, 1), (0, 0))
    data = data_nrh_2706[img_no][2]
    max_NRH = np.max(data)
    level = [0.5 * max_NRH, 0.99 * max_NRH]
    plt.contour(data, levels=level, colors='yellow', extent=extent)  # Use yellow color, default line width
    kk = 960
    circle = plt.Circle((0, 0), kk, color='red', fill=False)
    ax.add_artist(circle)
    img_time = start_time + datetime.timedelta(seconds=(img_no * 0.25))  # Time based on image index
    img_time_str = img_time.strftime('%Y%m%d_%H%M%S_%f')[:19]  # Format to include milliseconds
    plt.title(f"Image Time: {img_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\nFrequency: {frequency}")
    plt.xlabel('X (arcseconds)')
    plt.ylabel('Y (arcseconds)')
    ax.set_aspect('equal')  # Set square axis (equal aspect ratio)
    # Save each plot to the specified directory with a unique filename
    output_path = os.path.join(output_dir, f'nrh_{frequency_str}_{img_time_str}_contour.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    #print(f"Saved image: {output_path}")

# Close the FITS file
hdu.close()