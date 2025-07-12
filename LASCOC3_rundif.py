import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import sunpy.map
import glob
import astropy.units as u
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from IPython.display import Video, display
from PIL import Image

def createCircularMask(h, w, center=None, radius=None):
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

# Output directory for images
output_dir = os.path.normpath('C:/Users/knadi/Desktop/lasco_c3_rundiff_frames')
os.makedirs(output_dir, exist_ok=True)

# Clear previous images in output directory to avoid size mismatches
for f in glob.glob(os.path.join(output_dir, 'lasco_c3_rundiff_*.png')):
    try:
        os.remove(f)
        print(f"Removed old file: {f}")
    except Exception as e:
        print(f"Error removing {f}: {e}")

# Load LASCO C3 data
lasco_c3 = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Soho_data/Lasco_C3/*.fits')
lasco_c3.sort()
print(f"Number of LASCO C3 files: {len(lasco_c3)}")

# Process consecutive pairs for running difference
n_pairs = len(lasco_c3) - 1
print(f"Number of pairs to process: {n_pairs}")

# Fixed figure size and DPI for consistent image output
fig_width, fig_height = 8, 8
dpi = 300
target_size = (int(fig_width * dpi), int(fig_height * dpi))  # e.g., (2400, 2400)

for i in range(n_pairs):
    try:
        # Load consecutive LASCO C3 images
        map1 = sunpy.map.Map(lasco_c3[i])
        map2 = sunpy.map.Map(lasco_c3[i + 1])
        print(f"Processing LASCO C3 pair at {map1.date} and {map2.date}")

        # Check exposure time
        if map1.exposure_time.value == 0 or map2.exposure_time.value == 0:
            print(f"Warning: Zero exposure time for image {i} or {i+1}, skipping pair")
            continue

        # Smooth and normalize data (adjusted filter size for C3's coarser resolution)
        data1 = uniform_filter(map1.data.astype(float), 5) / map1.exposure_time.value
        data2 = uniform_filter(map2.data.astype(float), 5) / map2.exposure_time.value
        print(f"Data1 min/max: {np.nanmin(data1)}/{np.nanmax(data1)}")
        print(f"Data2 min/max: {np.nanmin(data2)}/{np.nanmax(data2)}")

        # Compute running difference
        diff_data = data2 - data1
        print(f"Diff min/max: {np.nanmin(diff_data)}/{np.nanmax(diff_data)}")

        # Create a sunpy map for the difference image
        diff_map = sunpy.map.Map(diff_data, map1.meta)

        # Apply occulting disk mask using solar center from FITS header
        h, w = diff_data.shape
        center_x = map1.meta.get('crpix1', w / 2) - 1  # FITS is 1-indexed, Python is 0-indexed
        center_y = map1.meta.get('crpix2', h / 2) - 1
        center = [center_x, center_y]
        pixel_scale_x = map1.meta.get('cdelt1', 56.0) * u.arcsec / u.pix  # C3 typical pixel scale
        solar_radius = map1.meta.get('rsun', 960) * u.arcsec
        c3_occulting_radius = 3.7 * solar_radius  # LASCO C3 occulting disk
        radius_pixels = (c3_occulting_radius / pixel_scale_x).to(u.pix).value
        mask = createCircularMask(h, w, center=center, radius=int(radius_pixels))
        diff_map.data[mask] = np.nan
        print(f"After masking, Diff min/max: {np.nanmin(diff_map.data)}/{np.nanmax(diff_map.data)}")

        # Check if there's any valid data to plot
        if np.all(np.isnan(diff_map.data)):
            print("Warning: All data is NaN after masking, skipping plot")
            continue

        # Plotting with WCS projection
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = plt.subplot(projection=diff_map)
        vmax = np.nanpercentile(np.abs(diff_map.data), 98)  # 98th percentile for contrast
        vmin = np.nanpercentile(np.abs(diff_map.data), 10)
        im = diff_map.plot(axes=ax, cmap='Greys_r', vmin=-1, vmax=1)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'LASCO C3 RUNNING DIFFERENCE {processed_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
        ax.grid(False)
        ax.coords.grid = False
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #fig.suptitle(f'LASCO C3 Running Diff {diff_map.date}', fontsize=13)

        # Save the figure without bbox_inches='tight' to maintain consistent size
        output_file = os.path.join(output_dir, f'lasco_c3_rundiff_{i:03d}.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        # Resize image to ensure consistent dimensions
        img = Image.open(output_file)
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
            img.save(output_file)
            print(f"Resized {output_file} to {target_size}")
        print(f"Saved image: {output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error processing pair at index {i}: {e}")
