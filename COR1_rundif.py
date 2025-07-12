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
output_dir = 'C:/Users/knadi/Desktop/cor1_rndiffR_frames'
os.makedirs(output_dir, exist_ok=True)

# Clear previous images in output directory to avoid size mismatches
for f in glob.glob(os.path.join(output_dir, 'cor1_rndiff_*.png')):
    os.remove(f)
    print(f"Removed old file: {f}")

# Load COR1 data
cor1A = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Stereo_data/stereo_20250604025622_948139/secchi/L0/a/seq/cor1/20240529/*.fts')
cor1A.sort()
print(f"Number of COR1 files: {len(cor1A)}")

# Process consecutive pairs for running difference
n_pairs = len(cor1A) - 1
print(f"Number of pairs to process: {n_pairs}")

# Fixed figure size and DPI for consistent image output
fig_width, fig_height = 8, 8
dpi = 300
target_size = (int(fig_width * dpi), int(fig_height * dpi))  # e.g., (2400, 2400)

for i in range(n_pairs):
    try:
        # Load consecutive COR1-A images
        map1 = sunpy.map.Map(cor1A[i])
        map2 = sunpy.map.Map(cor1A[i + 1])
        print(f"Processing COR1-A pair at {map1.date} and {map2.date}")

        # Check exposure time
        if map1.exposure_time.value == 0 or map2.exposure_time.value == 0:
            print(f"Warning: Zero exposure time for image {i} or {i+1}, skipping pair")
            continue

        # Smooth and normalize data
        data1 = uniform_filter(map1.data.astype(float), 7) / map1.exposure_time.value
        data2 = uniform_filter(map2.data.astype(float), 7) / map2.exposure_time.value
        print(f"Data1 min/max: {np.nanmin(data1)}/{np.nanmax(data1)}")
        print(f"Data2 min/max: {np.nanmin(data2)}/{np.nanmax(data2)}")

        # Compute running difference
        diff_data = data2 - data1
        print(f"Diff min/max: {np.nanmin(diff_data)}/{np.nanmax(diff_data)}")

        # Create a sunpy map for the difference image
        diff_map = sunpy.map.Map(diff_data, map1.meta)

        # Apply occulting disk mask
        h, w = diff_data.shape
        center = [int(w / 2), int(h / 2) - 5]
        pixel_scale_x = map1.meta.get('cdelt1', 1) * u.arcsec / u.pix
        solar_radius = 960 * u.arcsec
        cor1_occulting_radius = 1.3 * solar_radius
        radius_pixels = (cor1_occulting_radius / pixel_scale_x).to(u.pix).value
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
        im = diff_map.plot(axes=ax, clip_interval=(5, 99.9)*u.percent, cmap='stereocor1')
        diff_map.draw_limb(axes=ax)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.coords.grid = (False)
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #fig.suptitle(f'COR1-A Running Diff {diff_map.date}', fontsize=13)

        # Save the figure without bbox_inches='tight' to maintain consistent size
        output_file = os.path.join(output_dir, f'cor1_rndiff_{i:03d}.png')
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
