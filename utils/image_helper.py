import numpy as np
import resolve as rve
import matplotlib.pyplot as plt
import h5py
import os

from astropy.io import fits
from matplotlib.gridspec import GridSpec
import imageio, re




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


def plot_amplitude_gains(root_dir, source, date, uantennas, station_table, title_suffix=""):

    title = f"{source}, {date} {title_suffix}"
    saving_path = os.path.join(root_dir, "images", source, "gain_plots")

    os.makedirs(saving_path, exist_ok=True)

    
    with h5py.File(os.path.join(root_dir, "output_files", source, date, "gain_logamp", "iteration_19.hdf5"), "r") as hdf:

        ### MGVI/geoVI amp gain plotter
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

        ### Amplitude gain LCP Plotter
        plt.clf()

        time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

        for ii in np.arange(int(len(uantennas))):
            plt.plot(time_domain_hours_half, amp_samples_mean[0, ii, :len(time_domain) // 2, 0],
                        label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            plt.fill_between(time_domain_hours_half,
                                amp_samples_mean[0, ii, :len(time_domain) // 2, 0] - amp_samples_std[0, ii,
                                                                                    :len(time_domain) // 2, 0],
                                amp_samples_mean[0, ii, :len(time_domain) // 2, 0] + amp_samples_std[0, ii,
                                                                                    :len(time_domain) // 2, 0],
                                alpha=0.5)
        plt.title(f'{title} - Amplitude gain LCP')
        plt.xlabel('Relative time, hours')
        plt.ylabel('Amplitude gain')
        plt.legend()
        plt.savefig(os.path.join(saving_path, f"{source}_{date}_amp_gain_LCP.png"), dpi=600)

        plt.clf()

        ### Amplitude gain RCP Plotter

        for ii in np.arange(int(len(uantennas))):
            plt.plot(time_domain_hours_half, amp_samples_mean[1, ii, :len(time_domain) // 2, 0],
                        label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            plt.fill_between(time_domain_hours_half,
                                amp_samples_mean[1, ii, :len(time_domain) // 2, 0] - amp_samples_std[1, ii,
                                                                                    :len(time_domain) // 2, 0],
                                amp_samples_mean[1, ii, :len(time_domain) // 2, 0] + amp_samples_std[1, ii,
                                                                                    :len(time_domain) // 2, 0],
                                alpha=0.5)
        plt.title(f'{title} - Amplitude gain RCP')
        plt.xlabel('Relative time, hours')
        plt.ylabel('Amplitude gain')
        plt.legend()
        plt.savefig(os.path.join(saving_path, f"{source}_{date}_amp_gain_RCP.png"), dpi=600)
        plt.clf()
        
def plot_phase_gains(root_dir, source, date, uantennas, station_table, title_suffix=""): 

    title = f"{source}, {date} {title_suffix}"
    saving_path = os.path.join(root_dir, "images", source, "gain_plots")

    os.makedirs(saving_path, exist_ok=True)

    ### Phase gain plotter
    with h5py.File(os.path.join(root_dir, "output_files", source, date, "gain_phase", "iteration_19.hdf5"), "r") as hdf:

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

        ### MGVI/geoVI Phase gain LCP and RCP Plotter
        time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

        for ii in np.arange(int(len(uantennas))):
            plt.plot(time_domain_hours_half, phase_samples_mean_deg[0, ii, :len(time_domain) // 2, 0],
                        label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            plt.fill_between(time_domain_hours_half,
                                phase_samples_mean_deg[0, ii, :len(time_domain) // 2, 0] - phase_samples_std_deg[0, ii,
                                                                                        :len(time_domain) // 2, 0],
                                phase_samples_mean_deg[0, ii, :len(time_domain) // 2, 0] + phase_samples_std_deg[0, ii,
                                                                                        :len(time_domain) // 2, 0],
                                alpha=0.5)
        plt.title(f'{title} - Phase gain LCP')
        plt.xlabel('Relative time, hours')
        plt.ylabel('Phase gain, deg')
        plt.legend()
        plt.savefig(os.path.join(saving_path, f"{source}_{date}_phase_gain_LCP.png"), dpi=600)
        plt.clf()

        for ii in np.arange(int(len(uantennas))):
            plt.plot(time_domain_hours_half, phase_samples_mean_deg[1, ii, :len(time_domain) // 2, 0],
                        label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            plt.fill_between(time_domain_hours_half,
                                phase_samples_mean_deg[1, ii, :len(time_domain) // 2, 0] - phase_samples_std_deg[1, ii,
                                                                                        :len(time_domain) // 2, 0],
                                phase_samples_mean_deg[1, ii, :len(time_domain) // 2, 0] + phase_samples_std_deg[1, ii,
                                                                                        :len(time_domain) // 2, 0],
                                alpha=0.5)
        plt.title(f'{title} - Phase gain RCP')
        plt.xlabel('Relative time, hours')
        plt.ylabel('Phase gain, deg')
        plt.legend()
        plt.savefig(os.path.join(saving_path, f"{source}_{date}_phase_gain_RCP.png"), dpi=600)
        plt.clf()
        


def create_gain_plots(root_dir, obs, source, date):
    """Parent function to create and save all gain plots for particular source and date"""
    uantennas = rve.unique_antennas(obs)
    station_table = obs._auxiliary_tables['ANTENNA']['STATION']

    plot_amplitude_gains(root_dir, source, date, uantennas, station_table)
    plot_phase_gains(root_dir, source, date, uantennas, station_table)



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


def create_movie_frames(root_dir, source_name, pixscale=0.05, contours=False, n_contours=5):
    """
    Create movie frames with 2 subplots per date:
    [0,0] VI, [0,1] Clean
    
    :param root_dir: root directory of the run
    :param source_name: name of the run
    :param pixscale: pixel scale in mas/pixel
    :param contours: whether to plot contours. Default is False
    :param n_contours: number of contours to plot
    """
    base_path = os.path.join(root_dir, "output_files", source_name)
    dates = os.listdir(base_path)
    dates.sort()

    save_dir = os.path.join(root_dir, "images", source_name, "movie_frames")
    os.makedirs(save_dir, exist_ok=True)

    nx, ny = 0, 0 # Initialize image dimensions
    frame_index = 0

    for epoch in dates:
        date_path = os.path.join(base_path, epoch)

        vi_hdf5 = os.path.join(date_path, "sky", "iteration_19.hdf5")
        vi_image = load_vi_image_from_hdf5(vi_hdf5)
        if vi_image is None:
            continue
        nx, ny = vi_image.shape
        fovx, fovy = nx * pixscale, ny * pixscale  # in mas
        min_x, max_x = -fovx / 2, fovx / 2
        min_y, max_y = -fovy / 2, fovy / 2


        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.3)

        # ===== TOP LEFT: VI IMAGE =====
        vi_ax = fig.add_subplot(gs[0, 0])

        x = np.linspace(+fovx / 2, -fovx / 2, nx)  # RA plotted reversed
        y = np.linspace(-fovy / 2, +fovy / 2, ny)
        X, Y = np.meshgrid(x, y)

        noise_level = noise_level_estimation(vi_image)
        
        im = vi_ax.pcolormesh(X, Y, np.log10(vi_image).T, vmin=np.log10(noise_level), shading='auto', cmap='inferno')
        vi_ax.set_title('VI posterior mean', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=vi_ax, label=r"$\log_{10}[I \ (\text{Jy/mas}^2)]$")
        peak = np.nanmax(np.log10(vi_image).T)
        minimum = np.nanmin(np.log10(vi_image).T)
        pos_levels = np.linspace(minimum, peak, n_contours)
        if contours:
            vi_ax.contour(X, Y, np.log10(vi_image).T, levels=pos_levels, colors='white', linewidths=0.8)
        vi_ax.invert_xaxis()

        vi_ax.set_aspect('equal')
        vi_ax.set_xlabel("Relative RA (mas)")
        vi_ax.set_ylabel("Relative Dec (mas)")


        # ===== TOP RIGHT: CLEAN IMAGE =====
        clean_ax = fig.add_subplot(gs[0, 1])

        # Change to the path where the original CLEAN images are stored
        original_data_path = f"/aux/zeall/2cmVLBA/data/{source_name}/{epoch}/{source_name}.u.{epoch}.icn.fits.gz"

        clean_image, X, Y = load_image_from_fits(original_data_path)
        noise_level_clean = noise_level_estimation(clean_image)
        im = clean_ax.pcolormesh(X, Y, np.log10(clean_image), shading='auto', cmap='inferno', vmin=np.log10(noise_level_clean))
        clean_ax.set_title('CLEAN', fontsize=14, fontweight='bold')
        clean_ax.set_xlim(max_x, min_x)
        clean_ax.set_ylim(min_y, max_y)
        plt.colorbar(im, ax=clean_ax, label=r"$\log_{10}[I \ (\text{Jy/mas}^2)]$")
        peak = np.nanmax(np.log10(clean_image))
        minimum = np.nanmin(np.log10(clean_image))
        pos_levels = np.linspace(minimum, peak, n_contours)
        if contours:
            clean_ax.contour(X, Y, np.log10(clean_image), levels=pos_levels, colors='white', linewidths=0.8)
        clean_ax.set_aspect('equal')
        clean_ax.set_xlabel("Relative RA (mas)")
        clean_ax.set_ylabel("Relative Dec (mas)")

    
        # Add date as overall title
        fig.suptitle(f'{source_name} - {epoch}', fontsize=16, fontweight='bold', y=0.995)

        # Save frame
        frame_path = os.path.join(save_dir, f"frame_{frame_index:03d}.png")
        plt.savefig(frame_path, dpi=600)
        plt.close(fig)
        frame_index += 1

    return save_dir


def create_movie(root_dir, source, fps=1):
    """Parent function to create movie of the source"""
    frames_dir = create_movie_frames(root_dir, source)
    files = sorted(os.listdir(frames_dir))
    out_path = os.path.join(root_dir, "images", source, f"movie_{source}.mp4")

    with imageio.get_writer(out_path, fps=fps, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
        for fn in files:
            img = imageio.imread(os.path.join(frames_dir, fn))
            writer.append_data(img)



