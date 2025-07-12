import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import sunpy.map
import glob
import astropy.units as u
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
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
output_dir = 'C:/Users/knadi/Desktop/cor1_base_diff_frames'
os.makedirs(output_dir, exist_ok=True)

# Load COR1 data
cor1A = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Stereo_data/stereo_20250604025622_948139/secchi/L0/a/seq/cor1/20240529/*.fts')
cor1A.sort()
print(f"Number of COR1 files: {len(cor1A)}")

# Load and preprocess the base image (first image)
if len(cor1A) < 2:
    print("Error: At least 2 images are required for base difference.")
else:
    try:
        base_map = sunpy.map.Map(cor1A[0])
        print(f"Base image timestamp: {base_map.date}")
        
        # Check exposure time of base image
        if base_map.exposure_time.value == 0:
            print("Error: Base image has zero exposure time, cannot proceed.")
        else:
            # Smooth and normalize base image data
            data_base = uniform_filter(base_map.data.astype(float), 7) / base_map.exposure_time.value
            print(f"Base Data min/max: {np.nanmin(data_base)}/{np.nanmax(data_base)}")

            # Process each subsequent image for base difference
            for i in range(1, len(cor1A)):
                try:
                    # Load current COR1-A image
                    map_current = sunpy.map.Map(cor1A[i])
                    print(f"Processing COR1-A image at {map_current.date}")

                    # Check exposure time
                    if map_current.exposure_time.value == 0:
                        print(f"Warning: Zero exposure time for image {i}, skipping")
                        continue

                    # Smooth and normalize current image data
                    data_current = uniform_filter(map_current.data.astype(float), 7) / map_current.exposure_time.value
                    print(f"Current Data min/max: {np.nanmin(data_current)}/{np.nanmax(data_current)}")

                    # Compute base difference
                    diff_data = data_current - data_base
                    print(f"Base Diff min/max: {np.nanmin(diff_data)}/{np.nanmax(diff_data)}")

                    # Create a sunpy map for the difference image
                    diff_map = sunpy.map.Map(diff_data, map_current.meta)

                    # Apply occulting disk mask
                    h, w = diff_data.shape
                    center = [int(w / 2), int(h / 2) - 5]
                    pixel_scale_x = map_current.meta.get('cdelt1', 1) * u.arcsec / u.pix
                    solar_radius = 960 * u.arcsec
                    cor1_occulting_radius = 1.3 * solar_radius
                    radius_pixels = (cor1_occulting_radius / pixel_scale_x).to(u.pix).value
                    mask = createCircularMask(h, w, center=center, radius=int(radius_pixels))
                    diff_map.data[mask] = np.nan
                    print(f"After masking, Base Diff min/max: {np.nanmin(diff_map.data)}/{np.nanmax(diff_map.data)}")

                    # Check if there's any valid data to plot
                    if np.all(np.isnan(diff_map.data)):
                        print("Warning: All data is NaN after masking, skipping plot")
                        continue

                    # Plotting with WCS projection in a subplot
                    fig = plt.figure(figsize=(8, 8), dpi=300)
                    ax = plt.subplot(projection=diff_map)
                    #im = diff_map.plot(axes=ax, clip_interval=(5, 99.9)*u.percent, cmap='stereocor1')
                    im = diff_map.plot(axes=ax,vmax = 30, vmin =0, cmap='stereocor1')
                    diff_map.draw_limb(axes=ax)
                    ax.set_facecolor('#7f7f7f')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.grid(False)
                    ax.coords.grid = (False)
                    #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    #fig.suptitle(f'COR1-A Base Diff {diff_map.date}', fontsize=13)

                    # Save the figure (FIX: Removed bbox_inches='tight')
                    output_file = os.path.join(output_dir, f'cor1_base_diff_{i-1:03d}.png')
                    plt.savefig(output_file)  # Changed from plt.savefig(output_file, bbox_inches='tight')
                    print(f"Saved image: {output_file}")
                    plt.close(fig)

                except FileNotFoundError as e:
                    print(f"Error: File not found: {e}")
                except Exception as e:
                    print(f"Error processing image at index {i}: {e}")

    except FileNotFoundError as e:
        print(f"Error: Base image file not found: {e}")
    except Exception as e:
        print(f"Error processing base image: {e}")
