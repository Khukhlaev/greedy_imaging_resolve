import resolve as rve
import numpy as np
import os

from collections import OrderedDict

import fcntl

import pandas as pd

from astropy.io import fits

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


def weighted_average_visibilities(obs_list, decimals=8):
    """
    Weighted average of visibilities, handling possible skipped uvw points for some spectral windows.

    Parameters
    ----------
    obs_list : list of rve.Observation 
        list of observations to combine

    Returns
    -------
    averaged_obs : rve.Observation
        observation with correctly averaged visibilies
    """
    groups = OrderedDict()


    for obs in obs_list:
        vis = obs.vis.val
        wgt = obs.weight.val
        uvw = obs.uvw

        if vis.shape != wgt.shape:
            raise ValueError("vis and wgt must have the same shape")
        if uvw.shape != (vis.shape[1], 3):
            raise ValueError("uvw must have shape (N, 3)")
        
        valid = np.isfinite(vis) & np.isfinite(wgt) & (wgt > 0)

        for i in range(vis.shape[1]):
            # Group by UVW, rounded
            key = tuple(np.round(uvw[i], decimals=decimals))

            if key not in groups:
                groups[key] = {
                    "sum_vw": np.zeros(vis[:, i, :].shape, dtype=vis.dtype),
                    "sum_w": np.zeros(vis[:, i, :].shape, dtype=wgt.dtype),
                    "time": 0,
                    "ant1": -1,
                    "ant2": -1
                }

            m = valid[:, i, :]               # shape (2, 1)
            if not np.any(m):
                continue

            wi = np.where(m, wgt[:, i, :], 0.0)
            vi = np.where(m, vis[:, i, :], 0.0)

            groups[key]["sum_vw"] += vi * wi
            groups[key]["sum_w"] += wi

            # Store the first valid time, ant1, ant2 for this UVW group
            if groups[key]["time"] == 0:
                groups[key]["time"] = float(obs.time[i])
                groups[key]["ant1"] = obs.ant1[i]
                groups[key]["ant2"] = obs.ant2[i]


    m = len(groups)

    vis_avg = np.zeros((obs_list[0].vis.val.shape[0], m, 1), dtype=obs_list[0].vis.val.dtype)
    wgt_new = np.zeros((obs_list[0].vis.val.shape[0], m, 1), dtype=obs_list[0].weight.val.dtype)
    uvw_new = np.zeros((m, 3), dtype=obs_list[0].uvw.dtype)
    time_new = np.zeros(m, dtype=obs_list[0].time.dtype)
    ant1_new = np.zeros(m, dtype=obs_list[0].ant1.dtype)
    ant2_new = np.zeros(m, dtype=obs_list[0].ant2.dtype)

    for j, (key, acc) in enumerate(groups.items()):
        sw = acc["sum_w"]
        svw = acc["sum_vw"]

        out = np.zeros_like(svw, dtype=obs_list[0].vis.val.dtype)
        np.divide(svw, sw, out=out, where=sw > 0)
        vis_avg[:, j, :] = out

        wgt_new[:, j, :] = sw
        uvw_new[j] = np.array(key, dtype=obs_list[0].uvw.dtype)
        time_new[j] = acc["time"]
        ant1_new[j] = acc["ant1"]
        ant2_new[j] = acc["ant2"]


    sorted_time_indices = np.argsort(time_new)
    averaged_obs = obs_list[0]
    averaged_obs._weight = wgt_new[:, sorted_time_indices, :]
    averaged_obs._vis = vis_avg[:, sorted_time_indices, :]
    averaged_obs._antpos = rve.AntennaPositions(uvw_new[sorted_time_indices, :], ant1_new[sorted_time_indices], ant2_new[sorted_time_indices], time_new[sorted_time_indices])

    return averaged_obs


def get_observation(store_dir, source, filename, polarizations="stokesi"):
    """
    Get observation, correctly averaging spectral windows. 
    The function assumes that you have already transformed the original data to the ms format.
    """

    ms_path = os.path.join(store_dir, source, f"{filename}.ms")

    if not os.path.exists(ms_path):
        raise FileNotFoundError(f"ms data file does not exist: {ms_path}")

    observations = []
    spectral_window = 0

    while spectral_window < 100: # Avoiding infinite loop
        try:
            obs = rve.ms2observations(ms=ms_path, data_column="DATA", with_calib_info=True, spectral_window=spectral_window, polarizations=polarizations, ignore_flags=True)[0]
            observations.append(obs)
            spectral_window += 1
        except Exception as e:
            break
    
    if len(observations) == 0:
        raise RuntimeError("cannot load observation")
    
    if spectral_window < 1:
        return observations[0]
    
    return weighted_average_visibilities(observations)


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