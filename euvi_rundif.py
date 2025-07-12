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

def limb_enhance(map):
    x, y = np.meshgrid(*[np.arange(v.value) for v in map.dimensions]) * u.pix
    hpc_coords = map.pixel_to_world(x, y)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (955.98558 * u.arcsec)
    rsun_array = np.arange(1, r.max(), 0.01)
    y = np.array([map.data[(r > this_r) * (r < this_r + 0.01)].mean() for this_r in rsun_array])
    params = np.polyfit(rsun_array[rsun_array < 1.5], np.log(y[rsun_array < 1.5]), 1)
    scale_factor = np.exp((r - 1) * -params[0])
    scale_factor[r < 1] = 1
    return sunpy.map.Map(map.data * scale_factor, map.meta)

# File paths
files = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/Stereo_data/stereo_20250526023633_101741/secchi/L0/a/img/euvi/20240529/*.fts')
files.sort()
output_dir = 'C:/Users/knadi/Desktop/euvi_284_rundif_frames'
os.makedirs(output_dir, exist_ok=True)

# Process
n_iterations = (len(files) - 1)

for i in range(n_iterations):
    try:
        # Load and process EUVI maps
        map_before = sunpy.map.Map(files[i])
        map_now = sunpy.map.Map(files[i + 1])
        print(f"Processing EUVI map at {map_now.date}")

        # Apply limb enhancement and smoothing
        map_pre = limb_enhance(map_before)
        map_now = limb_enhance(map_now)
        pre = uniform_filter(map_pre.data, 7) / map_pre.exposure_time.value
        now = uniform_filter(map_now.data, 7) / map_now.exposure_time.value

        # Create difference map
        map_dif = sunpy.map.Map(now - pre, map_now.meta)

        # Plot
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = plt.subplot(projection=map_dif.wcs)
        map_dif.plot(axes=ax, vmin=-2, vmax=8, norm=colors.PowerNorm(gamma=0.15))
        ax.set_facecolor('#7f7f7f')
        map_dif.draw_limb(axes=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.coords.grid = False
        plt.title(f'EUVI-A 284 Å running diff {map_now.date}', fontsize=12)
        #plt.colorbar(fraction=0.046, pad=0.04)

        # Save image
        output_file = os.path.join(output_dir, f'euvi_{i:03d}.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved image: {output_file}")
        plt.close(fig)

    except Exception as e:
        print(f"Error at index {i}: {e}")
