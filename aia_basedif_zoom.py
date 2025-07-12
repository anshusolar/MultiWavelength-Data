import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import sunpy.map
from matplotlib import colors
import glob
import os
from moviepy import ImageSequenceClip
from IPython.display import Video, display
import astropy.units as u
from astropy.visualization import ImageNormalize, LinearStretch

def limb_enhance(map):
    x, y = np.meshgrid(*[np.arange(v.value) for v in map.dimensions]) * u.pix
    hpc_coords = map.pixel_to_world(x, y)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (955.98558 * u.arcsec)
    rsun_array = np.arange(1, r.max(), 0.01)
    y = np.array([map.data[(r > this_r) * (r < this_r + 0.01)].mean() for this_r in rsun_array])
    params = np.polyfit(rsun_array[rsun_array < 1.5], np.log(y[rsun_array < 1.5]), 1)
    scale_factor = np.exp((r - 1) * -params[0])
    scale_factor[r < 1] = 1
    scaled_map = sunpy.map.Map(map.data * scale_factor, map.meta)
    # Set minimal norm without vmin/vmax to avoid conflicts
    scaled_map.plot_settings['norm'] = ImageNormalize(stretch=LinearStretch())
    return scaled_map

# File paths
files = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/sdo data/*.fits')
files.sort()
output_dir = 'C:/Users/knadi/Desktop/aia_171_basediffframes'
os.makedirs(output_dir, exist_ok=True)

# Load base frame (first file)
try:
    base_map = sunpy.map.Map(files[0])
    base_map_enhanced = limb_enhance(base_map)
    base_data = uniform_filter(base_map_enhanced.data, 7) / (base_map_enhanced.exposure_time.value or 1.0)
    print(f"Base frame loaded: {base_map.date}")
except Exception as e:
    print(f"Error loading base frame {files[0]}: {e}")
    exit()

# Process
n_iterations = len(files) - 1
print(f"Number of AIA 171 files: {len(files)}")

for i, file in enumerate(files[1:]):
    try:
        # Load current frame
        map_now = sunpy.map.Map(file)
        print(f"Processing AIA map at {map_now.date}")

        # Apply limb enhancement and smoothing
        map_now_enhanced = limb_enhance(map_now)
        now = uniform_filter(map_now_enhanced.data, 7) / (map_now.exposure_time.value or 1.0)

        # Create difference map (relative to base frame)
        map_dif = sunpy.map.Map(now - base_data, map_now.meta)
        # Set normalization for the difference map
        map_dif.plot_settings['norm'] = ImageNormalize(vmin=-200, vmax=200, stretch=LinearStretch())

        # Plot
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = plt.subplot(projection=map_dif)
        im = map_dif.plot(
            axes=ax,
            cmap='Greys_r',
            title=f'AIA 171 Å Base Difference {map_now.date}'
        )
        map_dif.draw_limb(axes=ax)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.coords.grid = False
        ax.set_xlim(0,1023)  
        ax.set_ylim(768,1792)
        #fig.colorbar(im, extend='both', label='Intensity (DN/s)')
        # Save image
        output_file = os.path.join(output_dir, f'aia_{i:03d}zoomed.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved image: {output_file}")
        plt.close(fig)

    except Exception as e:
        print(f"Error at index {i}: {e}")
