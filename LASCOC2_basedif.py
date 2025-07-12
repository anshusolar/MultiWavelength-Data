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
output_dir = 'C:/Users/knadi/Desktop/lasco_c2_BASEdiff_frames'
os.makedirs(output_dir, exist_ok=True)

# Clear previous images in output directory to avoid size mismatches
for f in glob.glob(os.path.join(output_dir, 'lasco_c2_basediff_*.png')):
    os.remove(f)
    print(f"Removed old file: {f}")

# Load LASCO C2 data
lasco_c2 = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Soho_data/Lasco_C2/*.fits')
lasco_c2.sort()
print(f"Number of LASCO C2 files: {len(lasco_c2)}")

# Initialize base image data
base_data = None

# Fixed figure size and DPI for consistent image output
fig_width, fig_height = 8, 8
dpi = 300
target_size = (int(fig_width * dpi), int(fig_height * dpi))  # e.g., (2400, 2400)

for i, file in enumerate(lasco_c2):
    try:
        # Load LASCO C2 image
        map_lasco = sunpy.map.Map(file)
        print(f"Processing LASCO C2 image at {map_lasco.date}")

        # Check exposure time
        if map_lasco.exposure_time.value == 0:
            print(f"Warning: Zero exposure time for image {i}, skipping")
            base_data = None  # Reset base data to handle gaps
            continue

        # Smooth and normalize data (reduced filter size for finer details)
        data = uniform_filter(map_lasco.data.astype(float), 3) / map_lasco.exposure_time.value
        print(f"Data min/max: {np.nanmin(data)}/{np.nanmax(data)}")

        # Set the first valid image as the base image
        if base_data is None:
            base_data = data
            print(f"Set base image at {map_lasco.date}")
            continue

        # Compute base difference
        diff_data = data - base_data
        print(f"Diff min/max: {np.nanmin(diff_data)}/{np.nanmax(diff_data)}")

        # Create a sunpy map for the difference image
        diff_map = sunpy.map.Map(diff_data, map_lasco.meta)

        # Apply occulting disk mask
        h, w = diff_data.shape
        center = [int(w / 2), int(h / 2)]
        pixel_scale_x = map_lasco.meta.get('cdelt1', 11.9) * u.arcsec / u.pix
        solar_radius = map_lasco.meta.get('rsun', 960) * u.arcsec
        c2_occulting_radius = 1.8 * solar_radius  # Slightly reduced to show more coronal features
        radius_pixels = (c2_occulting_radius / pixel_scale_x).to(u.pix).value
        mask = createCircularMask(h, w, center=center, radius=int(radius_pixels))
        diff_map.data[mask] = np.nan
        print(f"After masking, Diff min/max: {np.nanmin(diff_map.data)}/{np.nanmax(diff_map.data)}")

        # Check if there's any valid data to plot
        if np.all(np.isnan(diff_map.data)):
            print("Warning: All data is NaN after masking, skipping plot")
            continue

        # Plotting with WCS projection in a subplot
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = plt.subplot(projection=diff_map)  # Subplot with WCS projection
        # Use percentile-based vmin/vmax for magma colormap to enhance fine details
        vmax = np.nanpercentile(np.abs(diff_map.data), 98)
        vmin = np.nanpercentile(np.abs(diff_map.data), 10)
        im = diff_map.plot(axes=ax, cmap='Greys_r', vmin=-1, vmax=1)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'LASCO C2 BASE DIFFERENCE {processed_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
        ax.grid(False)
        ax.coords.grid = False
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #fig.suptitle(f'LASCO C2 Base Diff {diff_map.date}', fontsize=13)

        # Save the figure without bbox_inches='tight' to maintain consistent size
        output_file = os.path.join(output_dir, f'lasco_c2_basediff_{i:03d}.png')
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
        base_data = None
    except Exception as e:
        print(f"Error processing image at index {i}: {e}")
        base_data = None
