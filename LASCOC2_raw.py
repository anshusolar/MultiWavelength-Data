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
output_dir = 'C:/Users/knadi/Desktop/lasco_c2_raw_frames'
os.makedirs(output_dir, exist_ok=True)

# Load LASCO C2 data
lasco_c2 = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Soho_data/Lasco_C2/*.fits')
lasco_c2.sort()
print(f"Number of LASCO C2 files: {len(lasco_c2)}")

# Process each image individually
for i, file in enumerate(lasco_c2):
    try:
        # Load LASCO C2 image
        map_lasco = sunpy.map.Map(file)
        print(f"Processing LASCO C2 image at {map_lasco.date}")

        # Check exposure time
        if map_lasco.exposure_time.value == 0:
            print(f"Warning: Zero exposure time for image {i}, skipping")
            continue

        # Smooth and normalize data 
        data = uniform_filter(map_lasco.data.astype(float), 5) / map_lasco.exposure_time.value
        print(f"Data min/max: {np.nanmin(data)}/{np.nanmax(data)}")

        # Create a sunpy map for the processed image
        processed_map = sunpy.map.Map(data, map_lasco.meta)

        # Apply occulting disk mask
        h, w = data.shape
        center = [int(w / 2), int(h / 2)]  # Adjust center if needed
        pixel_scale_x = map_lasco.meta.get('cdelt1', 11.9) * u.arcsec / u.pix
        solar_radius = map_lasco.meta.get('rsun', 960) * u.arcsec  # Use RSUN from header
        c2_occulting_radius = 2.0 * solar_radius  # LASCO C2 occulting disk ~2 solar radii
        radius_pixels = (c2_occulting_radius / pixel_scale_x).to(u.pix).value
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
        im = processed_map.plot(axes=ax, clip_interval=(5, 99.9)*u.percent, cmap='soholasco2', autoalign=True)
        processed_map.draw_limb(axes=ax)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'LASCO C2 {processed_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
        ax.grid(False)
        ax.coords.grid = False
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #fig.suptitle(f'LASCO C2 Raw {processed_map.date}', fontsize=13)
        
        # Save the figure
        output_file = os.path.join(output_dir, f'lasco_c2_raw_{i:03d}.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved image: {output_file}")
        plt.close(fig)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error processing image at index {i}: {e}")
