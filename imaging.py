import os
# Hardcode number of threads for Resolve and NIFTy here before importing them, setting them all to 1
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1



import resolve as rve
import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
import configparser
import argparse
import re
import sys

import shutil
from tqdm import tqdm

import datetime

from utils.utilities import get_zeromode_offset, get_ms_data_path, get_clean_params, safe_append_file
from utils.sky_model import sky_model_diffuse
from utils.calibration_operator import get_calibration_operator
from utils.image_helper import noise_level_estimation, create_gain_plots, create_movie

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Setting up argument parser to accept configuration file path
parser = argparse.ArgumentParser(description='Run imaging pipeline for specific source and date')
parser.add_argument('--config', type=str, help='path to the configuration file.')
args = parser.parse_args()

# Starting by reading and parsing the configuration file
cfg = configparser.ConfigParser()
cfg.read(args.config)

cfg_observation = cfg["observation"]
source_name = cfg_observation["source_name"].strip()
date = cfg_observation["date"].strip()

sys_error_percentage = cfg_observation.getfloat("sys_error_percentage")
spectral_window = cfg_observation.getint("spectral_window")
polarizations = cfg_observation["polarizations"]
visibility_type = cfg_observation.get("visibility_type", "uvf")

seed = cfg["base"].getint("seed")
np.random.seed(seed)
ift.random.push_sseq_from_seed(seed)


root_save_directory = cfg["base"]["root_output_directory"]
os.makedirs(root_save_directory, exist_ok=True)
os.makedirs(os.path.join(root_save_directory, "logs"), exist_ok=True)

map_flag = cfg["optimization"]["map"] == "True"
map_message = "on top of MAP" if map_flag else "standalone" 

pixscale = cfg["sky"].getfloat("pixscale", 0.05)  # in mas/pixel
n_pix_x, n_pix_y = cfg["sky"].getint("n_pixels_x"), cfg["sky"].getint("n_pixels_y")

save_strategy = cfg["base"].get("save_strategy", "last")

central_error_log = os.path.join(root_save_directory, "logs", "errors.log")
log_file = os.path.join(root_save_directory, "logs", f"{source_name}_{date}.log")

starting_time = datetime.datetime.now()

if n_pix_x == 0 or n_pix_y == 0:
    clean_params = get_clean_params(source_name, date)
    imsize = clean_params["imsize"] # Tuple of 2 image size values
else:
    imsize = (n_pix_x, n_pix_y)

fov = (imsize[0] * pixscale, imsize[1] * pixscale)  # in mas


### 1. loading data
try:
    data_path = get_ms_data_path("./ms_data", source_name, date, visibility_type)
    obs = rve.ms2observations(ms = data_path, data_column = "DATA",with_calib_info= True,spectral_window= spectral_window,polarizations= polarizations)

except Exception as e:
    safe_append_file(f"{get_current_time_str()}: Error loading data for source {source_name}, date {date}. Error: {e}.\n", central_error_log)

    sys.exit()
        
obs = obs[0]
tmin, tmax = rve.tmin_tmax(obs)
obs = obs.move_time(-tmin)
obs = obs.to_double_precision()

zeromode_offset = get_zeromode_offset(fov, obs.vis.val)

# add systematic error budget
new_weight = 1 / ((1 / np.sqrt(obs._weight)) ** 2 + (
            sys_error_percentage * abs(obs.vis.val)) ** 2)  # 1/ (sigma**2 + (sys_error_percentage*|A|)**2)
obs._weight = new_weight


# If matern kernel is used, config should provide its specific parameters!
matern = True if cfg["sky"].get("matern", "False") == "True" else False

### 2. sky prior
sky_diffuse, additional_operators = sky_model_diffuse(cfg["sky"], imsize, fov, zeromode_offset=zeromode_offset, matern=matern)
sky = sky_diffuse
additional_ops = additional_operators

### 3. calibration prior
cfg_base = cfg["base"]
epsilon = cfg_base.getfloat("epsilon")
do_wgridding = cfg_base.getboolean("do_wgridding")
nthreads_nifty = cfg_base.getint("nthreads_ift")

antenna_dct, calibration_op, phase, logamp = get_calibration_operator(cfg, obs, tmin, tmax)


### 4. likelihood
likelihood = rve.ImagingLikelihood(obs, sky, epsilon, do_wgridding, calibration_operator=calibration_op, nthreads=1)


# 5. optimization parameters setup
cfg_optimization = cfg["optimization"]
ic_sampling_iter_lim = cfg_optimization.getint("ic_samplilng_iter_lim")
ic_VL_BFGS_iter_lim_1 = cfg_optimization.getint("ic_VL_BFGS_iter_lim_1")
ic_VL_BFGS_iter_lim_2 = cfg_optimization.getint("ic_VL_BFGS_iter_lim_2")
ic_newton_iter_lim_3 = cfg_optimization.getint("ic_newton_iter_lim_3")
ic_newton_iter_lim_4 = cfg_optimization.getint("ic_newton_iter_lim_4")
ic_sampling_nl_iter_lim = cfg_optimization.getint("ic_sampling_nl_iter_lim")

ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit=ic_sampling_iter_lim)
ic_VL_BFGS_1 = ift.AbsDeltaEnergyController(deltaE=0.5,
                                        convergence_level=2, iteration_limit=ic_VL_BFGS_iter_lim_1)
ic_VL_BFGS_2 = ift.AbsDeltaEnergyController(deltaE=0.5,
                                        convergence_level=2, iteration_limit=ic_VL_BFGS_iter_lim_2)
ic_newton_3 = ift.AbsDeltaEnergyController(deltaE=0.5,
                                        convergence_level=2, iteration_limit=ic_newton_iter_lim_3)
ic_newton_4 = ift.AbsDeltaEnergyController(deltaE=0.5,
                                        convergence_level=2, iteration_limit=ic_newton_iter_lim_4)
ic_sampling_nl = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=ic_sampling_nl_iter_lim,
                                            convergence_level=2)
minimizer_1 = ift.VL_BFGS(ic_VL_BFGS_1)
minimizer_2 = ift.VL_BFGS(ic_VL_BFGS_2)
minimizer_3 = ift.NewtonCG(ic_newton_3)
minimizer_4 = ift.NewtonCG(ic_newton_4)
minimizer = lambda iiter : minimizer_1 if iiter < 5 else minimizer_2 if iiter < 10 else minimizer_3 if iiter < 15 else minimizer_4
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

# Setting up grid of image coordinates for plotting
npix_x, npix_y = imsize
fov_x, fov_y = fov
x = np.linspace(+fov_x / 2, -fov_x / 2, npix_x)  # RA plotted reversed
y = np.linspace(-fov_y / 2, +fov_y / 2, npix_y)
X, Y = np.meshgrid(x, y)

n_iterations_map = cfg["optimization"].getint("n_iterations_map")
n_samples_vi = cfg["optimization"].getint("n_samples_vi")    
n_iterations_vi = cfg["optimization"].getint("n_iterations_vi")

n_samples = lambda iiter: n_samples_vi if iiter < 15 else 2 * n_samples_vi

map_sample = None


if map_flag:

    def inspect_callback_map(sl, iglobal):
        if iglobal + 1 == n_iterations_map:

            sample_multifield = list(sl.iterator())[0]
            map_likelihood = likelihood(sample_multifield).val

            sky_map = sl.average(sky).val.T[:, :, 0, 0, 0] / (206265 ** 2 * 1000 ** 2)
            noise_level = noise_level_estimation(sky_map)
            sky_pos_mean = plt.pcolormesh(X, Y, np.log10(sky_map), cmap="inferno", vmin=np.log10(noise_level))
            plt.colorbar(sky_pos_mean, label=r"$\log_{10}[I \ (\text{Jy/mas}^2)]$")
            plt.title(f"MAP {source_name} {date} seed={seed}, likelihood={map_likelihood:.0f}", fontsize=12)
            plt.xlabel("Relative RA (mas)", fontsize=11)
            plt.ylabel("Relative Dec (mas)", fontsize=11)
            plt.gca().invert_xaxis()
            save_dir = os.path.join(root_save_directory, "images", source_name, date, "initial_MAP")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{source_name}_{date}_MAP_{seed}.png"), bbox_inches="tight", dpi=600)
            plt.close()

    output_directory = os.path.join(root_save_directory, "output_files", source_name, date, "initial_MAP", f"seed_{seed}")
    os.makedirs(output_directory, exist_ok=True)


    try:
        map_sample = ift.optimize_kl(likelihood_energy=likelihood,
                                total_iterations=n_iterations_map,
                                n_samples=0,
                                kl_minimizer=minimizer,
                                sampling_iteration_controller=ic_sampling,
                                nonlinear_sampling_minimizer=None,
                                export_operator_outputs= {
                                    "gain_phase": phase,
                                    "gain_logamp": logamp,
                                    "sky": sky,
                                },
                                output_directory=output_directory,
                                inspect_callback=inspect_callback_map,
                                comm=None,
                                resume=False,
                                save_strategy="last",
                                )
    except Exception as e:
        safe_append_file(f"{get_current_time_str()}: Error during MAP optimization for source {source_name}, date {date}, seed {seed}. Error: {e}\n", central_error_log)
        sys.exit()

    sample_multifield = list(map_sample.iterator())[0]
    final_map_likelihood = likelihood(sample_multifield).val
    map_sample = sample_multifield


def inspect_callback_vi(sl, iglobal):
    if iglobal + 1 == n_iterations_vi:
        vi_samples_multifield = list(sl.iterator())
        average_likelihood = sum([likelihood(vi_sample_multifield).val for vi_sample_multifield in vi_samples_multifield]) / len(vi_samples_multifield)

        sky_map = sl.average(sky).val.T[:, :, 0, 0, 0] / (206265 ** 2 * 1000 ** 2)
        noise_level = noise_level_estimation(sky_map)
        sky_pos_mean = plt.pcolormesh(X, Y, np.log10(sky_map), cmap="inferno", vmin=np.log10(noise_level))
        plt.colorbar(sky_pos_mean, label=r"$\log_{10}[I \ (\text{Jy/mas}^2)]$")
        plt.title(f"VI {source_name} {date}, seed {seed}, {map_message}, likelihood={average_likelihood:.0f}", fontsize=12)
        plt.xlabel("Relative RA (mas)", fontsize=11)
        plt.ylabel("Relative Dec (mas)", fontsize=11)
        plt.gca().invert_xaxis()
        save_dir = os.path.join(root_save_directory, "images", source_name, date)
        
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(os.path.join(save_dir, f"{source_name}_{date}_VI_{seed}.png"), bbox_inches="tight", dpi=600)
        plt.close()

output_directory = os.path.join(root_save_directory, "output_files", source_name, date, f"seed_{seed}")
os.makedirs(output_directory, exist_ok=True)

try:
    vi_samples = ift.optimize_kl(likelihood_energy=likelihood,
                    total_iterations=n_iterations_vi,
                    n_samples=n_samples,
                    kl_minimizer=minimizer,
                    sampling_iteration_controller=ic_sampling,
                    nonlinear_sampling_minimizer=None,
                    export_operator_outputs= {
                        "gain_phase": phase,
                        "gain_logamp": logamp,
                        "sky": sky,
                    },
                    output_directory=output_directory,
                    inspect_callback=inspect_callback_vi,
                    comm=None,
                    resume=False,
                    save_strategy=save_strategy,
                    initial_position=map_sample
                    )
except Exception as e:
    safe_append_file(f"{get_current_time_str()}: Error during VI optimization for source {source_name}, date {date}. Error: {e}\n", central_error_log)
    sys.exit()


vi_samples_multifield = list(vi_samples.iterator())
average_likelihood = sum([likelihood(vi_sample_multifield).val for vi_sample_multifield in vi_samples_multifield]) / len(vi_samples_multifield)

ending_time = datetime.datetime.now()
timedelta = ending_time - starting_time

safe_append_file(f"{get_current_time_str()}: Finished VI for source {source_name}, date {date}, seed {seed}, {map_message}. Final likelihood: {average_likelihood:.2f}. Total time taken: {(timedelta.total_seconds() / 3600):.2f} hours\n", log_file)


# Creating gain plots for every date
# create_gain_plots(root_save_directory, obs, source_name, date)

