import resolve as rve
import numpy as np
import os

from collections import OrderedDict

import fcntl

import pandas as pd

from astropy.io import fits

from .image_helper import load_vi_image_from_hdf5, get_correct_filepath

def mas_to_rad(mas: float) -> float:
    """Convert milliarcseconds to radians."""
    return mas * (np.pi / (180.0 * 3600.0 * 1000.0))

def rad_to_mas(rad: float) -> float:
    """Convert radians to milliarcseconds."""
    return rad * (180.0 * 3600.0 * 1000.0) / np.pi


def get_zeromode_offset(fov, visibility_vals, uvw):
    """
    Calculate the zeromode offset based on visibility values.  
    :param fov: tuple of (fov_x, fov_y) in mas
    :param visibility_vals: visibility values array (created from obs.vis.val)
    :param uvw: uvw values array (created from obs.uvw)

    :return: zeromode offset
    """
    
    amp_array = np.mean(np.abs(visibility_vals), axis=0)[:, 0]

    u = uvw[:, 0]
    v = uvw[:, 1]
    distance = np.sqrt(u ** 2 + v ** 2)
    min_indices = np.argsort(distance.flatten())[:10]
    mindist_10_amp_list = amp_array[min_indices]
    avg_amp = np.mean(mindist_10_amp_list)

    fov_x, fov_y = fov
    zeromode_offset = np.round(np.log(avg_amp / (mas_to_rad(fov_x) * mas_to_rad(fov_y))))

    return zeromode_offset


def get_source_date(uvf_path: str) -> tuple:
    """
    Extract source name and date from the uvf file path.
    
    :param uvf_path: path to the uvf file
    :type uvf_path: str
    :return: tuple of (source_name, date)
    :rtype: tuple
    """
    header = fits.getheader(uvf_path)

    return header['OBJECT'], header['DATE-OBS'].replace('-', '_')

def get_observation(store_dir, source, filename, polarizations="stokesi"):
    """
    Get observation, correctly averaging spectral windows. 
    The function assumes that you have already transformed the original data to the ms format.
    """

    ms_path = os.path.join(store_dir, source, f"{filename}.ms")

    if not os.path.exists(ms_path):
        raise FileNotFoundError(f"ms data file does not exist: {ms_path}")

    # First, trying ingnore_flags=False. If only one SPW was present in the data, then the flags are handled correctly (no CASA averaging)
    obs = rve.ms2observations(ms=ms_path, data_column="DATA", with_calib_info=True, spectral_window=0, polarizations=polarizations, ignore_flags=False)[0]

    if obs is None: # Happens when the data has multiple SPWs and CASA averaged them
        # Using ignore_flags=True, since we already handled flagging when combining SPWs with CASA
        obs = rve.ms2observations(ms=ms_path, data_column="DATA", with_calib_info=True, spectral_window=0, polarizations=polarizations, ignore_flags=True)[0]

    return obs


def append_message(text: str, file, other_file = None) -> None:
    """
    Append an message to the log file(s). Use file locking to prevent concurrent write issues.
    
    :param text: The message to append.
    :type text: str
    :param file: The path to the log file.
    :param other_file: optional path to the other log (error) file
    """

    with open(file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(text.rstrip() + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    if other_file is not None:
        with open(other_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(text.rstrip("\n") + "\n")
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def get_log_filename(root_dir, source):
    """
    Get a unique log filename for a given source. If a log file with the source name already exists, append an index to the filename.
    """

    log_dir = os.path.join(root_dir, "logs")
    log_filename = f"{source}.log"

    if not os.path.exists(os.path.join(log_dir, log_filename)):
        return log_filename

    i = 1
    while i < 1000:  # Arbitrary limit to prevent infinite loop
        candidate = f"{source}_{i}.log"
        if not os.path.exists(os.path.join(log_dir, candidate)):
            return candidate
        i += 1


def safe_append_row(path, row_dict):
    """Append a row to a CSV file in a thread-safe manner using file locking."""
    df_row = pd.DataFrame([row_dict])
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            df_row.to_csv(f, header=not file_exists, index=False)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def save_image_as_fits(root_dir, source_name, date, dir_name, seed, pixscale):
    """
    Save the resulting image as a FITS file in the specified directory. 

    :param root_dir: Root directory of the run.
    :param image_field: nifty.Field object, image field of the observation.
    :param source_name: Name of the source.
    :param date: Date of the observation in "YYYY_MM_DD" format.
    :param dir_name: Name of the directory of the run.
    :param seed: Seed of the specific run.
    :param pixscale: Pixel scale in mas/pixel.
    """

    fits_path = os.path.join(root_dir, "output_files", source_name, dir_name, "fits_images")
    os.makedirs(fits_path, exist_ok=True)
    fits_path = os.path.join(fits_path, f"{source_name}_{dir_name}_seed_{seed}.fits")
    hdf_path = os.path.join(root_dir, "output_files", source_name, dir_name, f"seed_{seed}", "sky")
    image_path = get_correct_filepath(hdf_path)

    image = load_vi_image_from_hdf5(image_path)
    image = image.T # Transpose to match (ny, nx) convention
    ny, nx = image.shape

    pixscale_deg = pixscale / (3600 * 1e3)  # Convert mas/pixel to deg/pixel

    h = fits.Header()
    h["BUNIT"] = "Jy/mas^2"
    h["CTYPE1"] = "RA---SIN"
    h["CRVAL1"] = 0.0
    h["CDELT1"] = -pixscale_deg
    h["CRPIX1"] = nx // 2
    h["CUNIT1"] = "deg"
    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL2"] = 0.0
    h["CDELT2"] = pixscale_deg
    h["CRPIX2"] = ny // 2
    h["CUNIT2"] = "deg"
    h["DATE-OBS"] = date.replace('_', '-')
    h["OBJECT"] = source_name
    hdu = fits.PrimaryHDU(image, header=h)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(fits_path, overwrite=True)






