import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import sunpy.map
from matplotlib import colors
import glob
import astropy.units as u
from astropy.visualization import ImageNormalize, LinearStretch
import os

def limb_enhance(map):
    x, y = np.meshgrid(*[np.arange(v.value) for v in map.dimensions]) * u.pix
    hpc_coords = map.pixel_to_world(x, y)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (955.98558 * u.arcsec)
    rsun_step_size = 0.01
    rsun_array = np.arange(1, r.max(), rsun_step_size)
    y = np.array([map.data[(r > this_r) * (r < this_r + rsun_step_size)].mean() for this_r in rsun_array])
    params = np.polyfit(rsun_array[rsun_array < 1.5], np.log(y[rsun_array < 1.5]), 1)
    scale_factor = np.exp((r - 1) * -params[0])
    scale_factor[r < 1] = 1
    scaled_map = sunpy.map.Map(map.data * scale_factor, map.meta)
    scaled_map.plot_settings['norm'] = ImageNormalize(stretch=map.plot_settings['norm'].stretch)
    return scaled_map
# Output directory for images
output_dir = 'C:/Users/knadi/Desktop/aia_171_frames'
os.makedirs(output_dir, exist_ok=True)

# Load AIA 171 data
st_files_171_A = glob.glob('C:/Users/knadi/OneDrive/Desktop/USO/sdo data/*.fits')
st_files_171_A.sort()
print(f"Number of AIA 171 files: {len(st_files_171_A)}")

# Process each file and save images
for i, file in enumerate(st_files_171_A):
    try:
        map_aia = sunpy.map.Map(file)
        print(f"Processing AIA map at {map_aia.date}")
        
        # Apply limb enhancement
        enhanced_map = limb_enhance(map_aia)
        smoothed_data = uniform_filter(enhanced_map.data, 7) / map_aia.exposure_time.value
        plot_map = sunpy.map.Map(smoothed_data, map_aia.meta)
        
        # Plotting
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = plt.subplot(projection=plot_map)
        im = plot_map.plot(axes=ax, clip_interval=(5, 99.9)*u.percent)
        plot_map.draw_limb(axes=ax)
        ax.set_facecolor('#7f7f7f')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.coords.grid = False
        plt.grid(False)
        #fig.colorbar(im)
        fig.suptitle(f'AIA-A 171 $\\mathrm{{Å}}$ {plot_map.date}', fontsize=13)

        # Save the figure
        output_file = os.path.join(output_dir, f'aia_{i:03d}.png')
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved image: {output_file}")
        plt.close(fig)
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error processing file {file}: {e}")
