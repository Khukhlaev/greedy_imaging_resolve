import numpy as np
import resolve as rve
import matplotlib.pyplot as plt
import h5py
import os
from pathlib import Path

from astropy.io import fits
import imageio

import pandas as pd


def get_correct_file(base_path):
    """Get the correct file path to load based on the saving strategy. If last.hdf5 exists, return it. Otherwise, return the iteration with the highest number."""
    last_file = Path(base_path) / "last.hdf5"
    if last_file.exists():
        return last_file
    
    iter_files = sorted(Path(base_path).glob("iteration_*.hdf5"))
    return iter_files[-1]   # highest iteration if names sort correctly



def border_mask(shape):
    """Create a border mask for the given shape. We assume that 1/8th of the image on each side is source-free."""
    h, w = shape
    bh = max(1, h // 8)
    bw = max(1, w // 8)
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[bh:-bh, :bw] = True
    mask[bh:-bh, -bw:] = True
    return mask

def noise_level_estimation(image):
    """Estimate noise level as 5 times rms pixel value of the image in the region without source emission. image shoud be a 2d array."""
    noise_mask = border_mask(image.shape)
    return np.sqrt(np.mean(image[noise_mask] ** 2)) * 3  # "3-rms" threshold



######## GAINS #################


def create_gain_plots(root_dir, obs, source, dir_name, seed):
    """Parent function to create and save gain plot for specific estimation"""
    uantennas = rve.unique_antennas(obs)
    station_table = obs._auxiliary_tables['ANTENNA']['STATION']

    overall_title = f"Gain plots - {source}, {dir_name}, seed={seed}"
    saving_path = os.path.join(root_dir, "images", source, dir_name, "gain_plots")

    os.makedirs(saving_path, exist_ok=True)

    time = obs.time - obs.time[0]
    dt = np.diff(time)
    breaks = np.where(dt > 5 * np.median(dt))[0]

    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, len(time) - 1]

    single_gain = obs.vis.val.shape[0] == 1

    if single_gain:
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    for ax in axs:
        for s, e in zip(starts, ends):
            ax.axvspan(time[s] / 3600, time[e] / 3600, color='0.85', alpha=0.7, zorder=0)

    # Accomodating for both saving strategies (last and all)
    base_path = os.path.join(root_dir, "output_files", source, dir_name, f"seed_{seed}", "gain_logamp")
    gain_filename = get_correct_file(base_path)

    # Amplitude gain plotter
    with h5py.File(gain_filename, "r") as hdf:

        amp_samples_list = []

        for ii in np.arange(np.array(hdf["samples"]).size):
            amp_sample = np.exp(np.array(hdf["samples"][f"{ii}"]))
            amp_samples_list.append(amp_sample)

        amp_samples_mean = np.mean(amp_samples_list, axis=0)
        amp_samples_std = np.std(amp_samples_list, axis=0)


        N = amp_samples_mean.shape[2]
        T = 10
        time_domain = np.linspace(0, N * T, N, endpoint=False)
        time_domain_hours = time_domain / 3600

        time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

        for ii in np.arange(int(len(uantennas))):
            axs[0].plot(time_domain_hours_half, amp_samples_mean[0, ii, :len(time_domain) // 2, 0],
                        label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            axs[0].fill_between(time_domain_hours_half,
                                amp_samples_mean[0, ii, :len(time_domain) // 2, 0] - amp_samples_std[0, ii,
                                                                                    :len(time_domain) // 2, 0],
                                amp_samples_mean[0, ii, :len(time_domain) // 2, 0] + amp_samples_std[0, ii,
                                                                                    :len(time_domain) // 2, 0],
                                alpha=0.5)
            
        title = "Amplitude gain" if single_gain else "Amplitude gain LCP"   
        axs[0].set_title(title)
        axs[0].set_xlabel('Relative time, hours')
        axs[0].set_ylabel('Amplitude gain')
        axs[0].legend()

        ### Amplitude gain RCP Plotter
        if not single_gain:
            for ii in np.arange(int(len(uantennas))):
                axs[1].plot(time_domain_hours_half, amp_samples_mean[1, ii, :len(time_domain) // 2, 0],
                            label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
                axs[1].fill_between(time_domain_hours_half,
                                    amp_samples_mean[1, ii, :len(time_domain) // 2, 0] - amp_samples_std[1, ii,
                                                                                        :len(time_domain) // 2, 0],
                                    amp_samples_mean[1, ii, :len(time_domain) // 2, 0] + amp_samples_std[1, ii,
                                                                                        :len(time_domain) // 2, 0],
                                    alpha=0.5)
            axs[1].set_title(f'Amplitude gain RCP')
            axs[1].set_xlabel('Relative time, hours')
            axs[1].set_ylabel('Amplitude gain')
            axs[1].legend()

    ### Phase gain plotter
    base_path = os.path.join(root_dir, "output_files", source, dir_name, f"seed_{seed}", "gain_phase")
    phase_filename = get_correct_file(base_path)

    with h5py.File(phase_filename, "r") as hdf:

        phase_samples_list = []

        for ii in np.arange(np.array(hdf["samples"]).size):
            phase_sample = np.array(hdf["samples"][f"{ii}"])
            phase_samples_list.append(phase_sample)

        phase_samples_mean = np.mean(phase_samples_list, axis=0)
        phase_samples_mean_deg = 180 / np.pi * phase_samples_mean

        phase_samples_std = np.std(phase_samples_list, axis=0)
        phase_samples_std_deg = 180 / np.pi * phase_samples_std

        N = phase_samples_mean.shape[2]
        T = 10
        time_domain = np.linspace(0, N * T, N, endpoint=False)
        time_domain_hours = time_domain / 3600

        time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

        ax_ind = 1 if single_gain else 2


        for ii in np.arange(int(len(uantennas))):
            axs[ax_ind].plot(time_domain_hours_half, phase_samples_mean_deg[0, ii, :len(time_domain) // 2, 0],
                        label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            axs[ax_ind].fill_between(time_domain_hours_half,
                                phase_samples_mean_deg[0, ii, :len(time_domain) // 2, 0] - phase_samples_std_deg[0, ii,
                                                                                        :len(time_domain) // 2, 0],
                                phase_samples_mean_deg[0, ii, :len(time_domain) // 2, 0] + phase_samples_std_deg[0, ii,
                                                                                        :len(time_domain) // 2, 0],
                                alpha=0.5)
        title = "Phase gain" if single_gain else "Phase gain LCP"
        axs[ax_ind].set_title(title)
        axs[ax_ind].set_xlabel('Relative time, hours')
        axs[ax_ind].set_ylabel('Phase gain, deg')
        axs[ax_ind].legend()

        if not single_gain:
            for ii in np.arange(int(len(uantennas))):
                axs[3].plot(time_domain_hours_half, phase_samples_mean_deg[1, ii, :len(time_domain) // 2, 0],
                            label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
                axs[3].fill_between(time_domain_hours_half,
                                    phase_samples_mean_deg[1, ii, :len(time_domain) // 2, 0] - phase_samples_std_deg[1, ii,
                                                                                            :len(time_domain) // 2, 0],
                                    phase_samples_mean_deg[1, ii, :len(time_domain) // 2, 0] + phase_samples_std_deg[1, ii,
                                                                                            :len(time_domain) // 2, 0],
                                    alpha=0.5)
            axs[3].set_title(f'Phase gain RCP')
            axs[3].set_xlabel('Relative time, hours')
            axs[3].set_ylabel('Phase gain, deg')
            axs[3].legend()

    plt.suptitle(overall_title, fontsize=16)
    plt.savefig(os.path.join(saving_path, f"{source}_{dir_name}_seed_{seed}_gains.png"), dpi=600)
    plt.close(fig)
    

########## MOVIES ##############

def load_image_from_fits(filepath):
    """Load image data from FITS file"""
    hdul = fits.open(filepath)
    data = hdul[0].data[0, 0, :, :]
    data[data <= 1e-5] = 1e-5  # Avoid log of zero or negative


    header = fits.getheader(filepath)

    # Convert from Jy/beam to Jy/mas^2
    bmaj = header.get('BMAJ', 1)  # Beam major axis in degrees
    bmin = header.get('BMIN', 1)  # Beam minor axis in degrees
    beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2)) * (3600 * 1e3) ** 2  # in mas^2
    data = data / beam_area  # Convert to Jy/mas^2

    pixscale = abs(header.get('CDELT1', 1)) * 3600 * 1e3  # Pixel scale in mas/pixel
    ny, nx = header.get('NAXIS1', 1), header.get('NAXIS2', 1)
    fovx = nx * pixscale  # in mas
    fovy = ny * pixscale  # in mas
    x = np.linspace(+fovx / 2, -fovx / 2, nx)  # RA plotted reversed
    y = np.linspace(-fovy / 2, +fovy / 2, ny)
    X, Y = np.meshgrid(x, y)

    hdul.close()

    return data, X, Y
    

def load_map_image_from_hdf5(hdf5_file):
    """Load MAP image from HDF5 file (single sample)"""
    try:
        hdf = h5py.File(hdf5_file, 'r')
        sample = np.array(hdf['samples'][f'0'])[0, 0, 0] / (206265 ** 2 * 1000 ** 2)
        hdf.close()
        return sample
    except:
        return None

def load_vi_image_from_hdf5(hdf5_file):
    """Load VI image from HDF5 file (multiple samples - averaged)"""
    try:
        hdf = h5py.File(hdf5_file, 'r')
        samples_data = np.array([hdf['samples'][f"{i}"] for i in range(len(hdf['samples']))])[:, 0, 0, 0, :, :]
        # Average all samples
        sample = np.mean(samples_data, axis=0) / (206265 ** 2 * 1000 ** 2)
        hdf.close()
        return sample
    except:
        return None


def create_movie_frames(root_dir, source_name, dir_name, pixscale=0.05, contours=False, n_contours=5):
    """
    Create movie frames with 1 plot per seed.
    
    :param root_dir: root directory of the run
    :param source_name: name of the source
    :param dir_name: name of the directory with the results
    :param pixscale: pixel scale in mas/pixel
    :param contours: whether to plot contours. Default is False
    :param n_contours: number of contours to plot
    """
    base_path = os.path.join(root_dir, "output_files", source_name, dir_name)
    seeds = [seed_dir for seed_dir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, seed_dir))]
    seeds.remove("initial_MAP")
    seeds.remove("fits_images")
    seeds.sort()

    csv_path = os.path.join(root_dir, "logs", "csv_files", f"{source_name}_{dir_name}.csv")
    info_df = pd.read_csv(csv_path)
    info_df = info_df.set_index("seed")

    save_dir = os.path.join(root_dir, "images", source_name, dir_name, "movie_frames")
    os.makedirs(save_dir, exist_ok=True)

    nx, ny = 0, 0 # Initialize image dimensions
    frame_index = 0

    for seed in seeds:
        seed_path = os.path.join(base_path, seed, "sky")

        vi_hdf5 = get_correct_file(seed_path)
        vi_image = load_vi_image_from_hdf5(vi_hdf5)
        if vi_image is None:
            continue
        nx, ny = vi_image.shape
        fovx, fovy = nx * pixscale, ny * pixscale  # in mas

        seed_num = seed[5:]  # Extract the number part from "seed_{number}"

        likelihood_val = info_df.loc[int(seed_num), "VI_likelihood"]
        map_flag = info_df.loc[int(seed_num), "MAP"]

        map_comment = "on top of MAP" if map_flag else "standalone"

        fig = plt.figure(figsize=(10, 8))

        x = np.linspace(+fovx / 2, -fovx / 2, nx)  # RA plotted reversed
        y = np.linspace(-fovy / 2, +fovy / 2, ny)
        X, Y = np.meshgrid(x, y)

        noise_level = noise_level_estimation(vi_image)
        
        im = plt.pcolormesh(X, Y, np.log10(vi_image).T, vmin=np.log10(noise_level), shading='auto', cmap='inferno')
        plt.title(f'VI {map_comment}, seed {seed_num}, likelihood={likelihood_val:.0f}', fontsize=14, pad=10, weight='bold')
        plt.colorbar(im, ax=plt.gca(), label=r"$\log_{10}[I \ (\text{Jy/mas}^2)]$")
        peak = np.nanmax(np.log10(vi_image).T)
        minimum = np.nanmin(np.log10(vi_image).T)
        pos_levels = np.linspace(minimum, peak, n_contours)
        if contours:
            plt.contour(X, Y, np.log10(vi_image).T, levels=pos_levels, colors='white', linewidths=0.8)
        plt.gca().invert_xaxis()

        plt.gca().set_aspect('equal')
        plt.xlabel("Relative RA (mas)")
        plt.ylabel("Relative Dec (mas)")

        plt.suptitle(f"{source_name}, {dir_name}", fontsize=16, weight='bold')

        # Save frame
        frame_path = os.path.join(save_dir, f"frame_{frame_index:03d}.png")
        plt.savefig(frame_path, dpi=300)
        plt.close(fig)
        frame_index += 1

    return save_dir


def create_movie(root_dir, source, dir_name, fps=1):
    """Parent function to create movie of the source"""
    frames_dir = create_movie_frames(root_dir, source, dir_name)
    files = sorted(os.listdir(frames_dir))
    out_path = os.path.join(root_dir, "images", source, dir_name, f"movie_{source}_{dir_name}.mp4")

    with imageio.get_writer(out_path, fps=fps, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
        for fn in files:
            img = imageio.imread(os.path.join(frames_dir, fn))
            writer.append_data(img)



