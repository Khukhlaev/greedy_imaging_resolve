import resolve as rve
import numpy as np
import os
from pyuvdata import UVData

import fcntl


from astropy.io import fits

def mas_to_rad(mas: float) -> float:
    """Convert milliarcseconds to radians."""
    return mas * (np.pi / (180.0 * 3600.0 * 1000.0))

def rad_to_mas(rad: float) -> float:
    """Convert radians to milliarcseconds."""
    return rad * (180.0 * 3600.0 * 1000.0) / np.pi

def dofdex_or_none(cfg, key, total_N):
    if cfg[key] == "False":
        return None
    if cfg[key] == "True":
        return np.hstack([np.arange(total_N//2), np.arange(total_N//2)])
        #return np.arange(total_N)
    else:
        return np.fromstring(cfg[key], dtype=int, sep=',')

def get_zeromode_offset(fov, visibility_vals):
    """
    Calculate the zeromode offset based on visibility values.  
    :param fov: tuple of (fov_x, fov_y) in mas
    :param visibility_vals: visibility values array (created from obs.vis.val)

    :return: zeromode offset
    """

    amp_array = 0.5 * (abs(visibility_vals)[0,:,0] + abs(visibility_vals)[1,:,0])

    u = visibility_vals[:,0]
    v = visibility_vals[:,1]
    distance = np.sqrt(u ** 2 + v ** 2)
    min_indices = np.argsort(distance.flatten())[:10]
    mindist_10_amp_list = amp_array[min_indices]
    avg_amp = np.mean(mindist_10_amp_list)

    fov_x, fov_y = fov
    zeromode_offset = np.round(np.log(avg_amp / (mas_to_rad(fov_x) * mas_to_rad(fov_y))))

    return zeromode_offset


def combine_spectral_windows(store_dir: str, uvf_path: str, gz_suffix: str) -> str:
    """
    Combining spectral windows with ehtim.
    
    :param store_dir: path to the direcrory where ms data will be stored
    :type store_dir: str
    :param uvf_path: original uvf filepath
    :type uvf_path: str
    :param gz_suffix: suffix for gzipped files, either ".gz" or ""
    :type gz_suffix: str
    :return: temporary filepath of the uvf file with comined spectral windows
    :rtype: str
    """

    import ehtim as eh

    tmp_dir = os.path.join(store_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    if gz_suffix == ".gz":
        tmp_file = os.path.join(tmp_dir, f"combined_{os.path.basename(uvf_path)[:-3]}")
    else:
        tmp_file = os.path.join(tmp_dir, f"combined_{os.path.basename(uvf_path)}")

    eh.obsdata.load_uvfits(uvf_path).save_uvfits(tmp_file)

    return tmp_file


def get_source_date_type(uvf_path: str) -> tuple:
    """
    Extract source name and date from the uvf file path.
    
    :param uvf_path: path to the uvf file
    :type uvf_path: str
    :return: tuple of (source_name, date, visibility_type, gz_suffix)
    :rtype: tuple
    """
    header = fits.getheader(uvf_path)
    extension = os.path.basename(uvf_path).split('.')[-1]
    base_extension = os.path.basename(uvf_path).split('.')[-2]
    gz_suffix = ".gz" if extension == "gz" else ""
    visibility_type = base_extension if gz_suffix else extension

    return header['OBJECT'], header['DATE-OBS'].replace('-', '_'), visibility_type, gz_suffix


def get_ms_data_path(store_dir: str, uvf_file: str = None, source: str = None, date: str = None, visibility_type: str = None) -> str:
    """
    Get the measurement set data path. If the measurement set does not exist, convert from UVF to MS format combining spectral channels.
    In the case of uvf_raw, also average the data in time with 10s bins.

    :param store_dir: path to the direcrory where ms data will be stored
    :type store_dir: str
    :param uvf_file: path to the uvf file. If not provided, the function will use the source name and date to attempt to load MOJAVE data.
    :type uvf_file: str
    :param source: name of the source. Example: "0506+056". Required if uvf_file is not provided.
    :type source: str
    :param date: date of observation in "YYYY_MM_DD" format. Example: "2025_06_01". Required if uvf_file is not provided.
    :type date: str
    :param visibility_type: type of visibility data. Possible types: "uvf", "uvf_raw_edt". "uvf_raw" is not implemented yet. Required if uvf_file is not provided.
    :type visibility_type: str
    :return: path to the measurement set folder
    :rtype: str
    """

    if visibility_type == "uvf_raw":
        raise NotImplementedError("support for uvf_raw data is not implemented yet")

    if uvf_file is not None:
        source, date, visibility_type, gz_suffix = get_source_date_type(uvf_file)
        uvf_path = uvf_file
    else:
        gz_suffix = "" if visibility_type=="uvf" else ".gz"
        uvf_path = f"/aux/zeall/2cmVLBA/data/{source}/{date}/{source}.u.{date}.{visibility_type}{gz_suffix}"

    ms_path = os.path.join(store_dir, source, f"{source}.u.{date}.{visibility_type}.ms")

    if os.path.exists(ms_path) and visibility_type != "uvf_raw":
        return ms_path
    
    os.makedirs(os.path.dirname(ms_path), exist_ok=True)
    
    uvf_path = combine_spectral_windows(store_dir, uvf_path, gz_suffix)

    UV = UVData()
    UV.read_uvfits(uvf_path)
    UV.write_ms(ms_path, clobber=True)

    # from casatasks import importuvfits

    # importuvfits(uvf_path, vis=ms_path)

    if os.path.dirname(uvf_path) == os.path.join(store_dir, "tmp"): # Dummy check to avoid deleting original data
        os.remove(uvf_path)

    return ms_path

def get_clean_params(source: str, date: str) -> dict:
    """
    Get CLEAN image parameters from the FITS header.
    
    :param source: name of the source. Example: "0506+056"
    :type source: str
    :param date: date of observation in "YYYY_MM_DD" format. Example: "2025_06_01"
    :type date: str
    :return: dictionary with CLEAN image parameters
    :rtype: dict
    """

    fits_file = f"/aux/zeall/2cmVLBA/data/{source}/{date}/{source}.u.{date}.icn.fits.gz"
    header = fits.getheader(fits_file, ext=0)
    pixel_scale_x = abs(header['CDELT1']) * 3600 * 1e3  # mas/pixel
    pixel_scale_y = abs(header['CDELT2']) * 3600 * 1e3  # mas/pixel

    fov_x = header['NAXIS1'] * pixel_scale_x  # mas
    fov_y = header['NAXIS2'] * pixel_scale_y  # mas
    clean_params = {
        'source': source,
        'date': date,
        'imsize': (int(header['NAXIS1']), int(header['NAXIS2'])),
        'pixel scale (mas)': (pixel_scale_x, pixel_scale_y),
        'fov (mas)': (fov_x, fov_y),
        'bmaj (mas)': header['BMAJ'] * 3600 * 1e3,  
        'bmin (mas)': header['BMIN'] * 3600 * 1e3,
        'bpa (deg)': header['BPA']
    }
    return clean_params


def get_good_random_dates(source: str, num_dates: int, visibility_type: str) -> list:
    """
    Get random dates for the source. Ensure corresponding data exist, can be transformed to ms format and loaded to resolve.

    :param source: name of the source. Example: "0506+056"
    :type source: str
    :param num_dates: number of dates
    :type num_dates: int
    :param visibility_type: type of visibility data. Possible types: "uvf", "uvf_raw", "uvf_raw_edt"
    :type visibility_type: str
    :return: list of good dates
    :rtype: list
    """

    possible_dates = np.array(os.listdir(f"/aux/zeall/2cmVLBA/data/{source}/"))
    np.random.shuffle(possible_dates)
    possible_dates = possible_dates.tolist()
    good_dates = []

    assert num_dates <= len(possible_dates), "Requested more dates than available."

    while len(good_dates) < num_dates:
        date = possible_dates.pop()
        try:
            data_path = get_ms_data_path(source, date, visibility_type)
            obs = rve.ms2observations(ms=data_path, data_column="DATA", with_calib_info=True, spectral_window=0, polarizations="stokesi")
            good_dates.append(date)
        except Exception as e:
            print(f"Error processing {date}: {e}")

    return good_dates



def safe_append_file(text: str, log_file: str) -> None:
    """
    Append a message to the central log file. Uses file locking to prevent concurrent write issues.
    
    :param text: The error message to append.
    :type text: str
    :param log_file: The path to the error log file.
    :type log_file: str
    """
    with open(log_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(text)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
