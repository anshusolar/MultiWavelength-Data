import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import sunpy.map
import glob
import astropy.units as u
import os
from moviepy import ImageSequenceClip
from IPython.display import Video, display
from astropy.visualization import ImageNormalize, LinearStretch

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
output_dir = 'C:/Users/knadi/Desktop/cor2_raw_frames'
os.makedirs(output_dir, exist_ok=True)

# Load COR2 data
cor2A = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Stereo_data/stereo_20250526052728_635708/secchi/L0/a/seq/cor2/20240529/*.fts')
cor2A.sort()
print(f"Number of COR2 files: {len(cor2A)}")

# Process each image individually
for i, file in enumerate(cor2A):
    try:
        # Load COR2-A image
        map_cor2 = sunpy.map.Map(file)
        print(f"Processing COR2-A image at {map_cor2.date}")

        # Verify metadata
        cdelt1 = map_cor2.meta.get('cdelt1', None)
        exptime = map_cor2.meta.get('exptime', None)
        
        # Validate cdelt1 (pixel scale)
        if cdelt1 is None or cdelt1 <= 0:
            print(f"Warning: Invalid or missing 'cdelt1' in metadata for image {i}, using default 14.7 arcsec/pixel")
            pixel_scale_x = 14.7 * u.arcsec / u.pix
        else:
            pixel_scale_x = cdelt1 * u.arcsec / u.pix
            print(f"Using cdelt1: {cdelt1} arcsec/pixel")

        # Validate EXPTIME (exposure time)
        if exptime is None or exptime <= 0:
            print(f"Warning: Invalid or missing 'EXPTIME' for image {i}, skipping")
            continue
        else:
            print(f"Using EXPTIME: {exptime} seconds")

        # Smooth and normalize data
        data = uniform_filter(map_cor2.data.astype(float), 7) / exptime
        print(f"Data min/max: {np.nanmin(data)}/{np.nanmax(data)}")

        # Create a sunpy map for the processed image
        processed_map = sunpy.map.Map(data, map_cor2.meta)

        # Apply occulting disk mask
        h, w = data.shape
        center = [int(w / 2), int(h / 2) - 5]  # Adjust offset if needed
        solar_radius = 960 * u.arcsec
        cor2_occulting_radius = 2.0 * solar_radius  # Larger occulting disk for COR2
        radius_pixels = (cor2_occulting_radius / pixel_scale_x).to(u.pix).value
        mask = createCircularMask(h, w, center=center, radius=int(radius_pixels))
        processed_map.data[mask] = np.nan
        print(f"After masking, Data min/max: {np.nanmin(processed_map.data)}/{np.nanmax(processed_map.data)}")

        # Check if there's any valid data to plot
        if np.all(np.isnan(processed_map.data)):
            print(f"Warning: All data is NaN after masking for image {i}, skipping plot")
            continue

        # Plotting with WCS projection in a subplot
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = plt.subplot(projection=processed_map)
        im = processed_map.plot(axes=ax, clip_interval=(5, 99.9)*u.percent, cmap='stereocor2')
        processed_map.draw_limb(axes=ax)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.coords.grid = (False)
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #fig.suptitle(f'COR2-A Raw {processed_map.date}', fontsize=13)

        # Save the figure
        output_file = os.path.join(output_dir, f'cor2_raw_{i:03d}.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved image: {output_file}")
        plt.close(fig)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error processing image at index {i}: {e}")
