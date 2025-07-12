import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import sunpy.map
import glob
import astropy.units as u
import os
from moviepy import ImageSequenceClip
from IPython.display import Video, display

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
output_dir = 'C:/Users/knadi/Desktop/cor1_RAW_frames'
os.makedirs(output_dir, exist_ok=True)

# Load COR1 data
cor1A = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Stereo_data/stereo_20250604025622_948139/secchi/L0/a/seq/cor1/20240529/*.fts')
cor1A.sort()
print(f"Number of COR1 files: {len(cor1A)}")

# Process each image individually
for i, file in enumerate(cor1A):
    try:
        # Load COR1-A image
        map_cor1 = sunpy.map.Map(file)
        print(f"Processing COR1-A image at {map_cor1.date}")

        # Check exposure time
        if map_cor1.exposure_time.value == 0:
            print(f"Warning: Zero exposure time for image {i}, skipping")
            continue

        # Smooth and normalize data 
        data = uniform_filter(map_cor1.data.astype(float), 7) / map_cor1.exposure_time.value
        print(f"Data min/max: {np.nanmin(data)}/{np.nanmax(data)}")

        # Create a sunpy map for the processed image
        processed_map = sunpy.map.Map(data, map_cor1.meta)

        # Apply occulting disk mask
        h, w = data.shape
        center = [int(w / 2), int(h / 2) - 5]
        pixel_scale_x = map_cor1.meta.get('cdelt1', 1) * u.arcsec / u.pix
        solar_radius = 960 * u.arcsec
        cor1_occulting_radius = 1.3 * solar_radius
        radius_pixels = (cor1_occulting_radius / pixel_scale_x).to(u.pix).value
        mask = createCircularMask(h, w, center=center, radius=int(radius_pixels))
        processed_map.data[mask] = np.nan
        print(f"After masking, Data min/max: {np.nanmin(processed_map.data)}/{np.nanmax(processed_map.data)}")

        # Check if there's any valid data to plot
        if np.all(np.isnan(processed_map.data)):
            print("Warning: All data is NaN after masking, skipping plot")
            continue

        # Plotting with WCS projection in a subplot
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = plt.subplot(projection=processed_map)  # Subplot with WCS projection
        im = processed_map.plot(axes=ax, clip_interval=(5, 99.9)*u.percent, cmap='stereocor1')
        processed_map.draw_limb(axes=ax)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.coords.grid = False
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       # fig.suptitle(f'COR1-A Raw {processed_map.date}', fontsize=13)

        # Save the figure
        output_file = os.path.join(output_dir, f'cor1_raw_{i:03d}.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved image: {output_file}")
        plt.close(fig)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error processing image at index {i}: {e}")
