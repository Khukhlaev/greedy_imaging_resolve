import os 

import subprocess, concurrent.futures, pathlib
from datetime import datetime
import configparser
import argparse

from utils.utilities import get_ms_data_path, get_source_date_type

import numpy as np

import sys


# Setting up argument parser to accept configuration file path
parser = argparse.ArgumentParser(description='Run imaging pipeline for multiple sources and dates. Either specify the parameters in main_conf.cfg or provide them as command line arguments. More detailed settings are availible via the config files.')
parser.add_argument('--config', type=str, help='path to the main configuration file. Default: ./main_conf.cfg', default="main_conf.cfg")
parser.add_argument('--data_file', type=str, help='path to the uvf data file. If not provided, the script will attempt to use source and date in the config to load corresponding MOJAVE data', default=argparse.SUPPRESS)
parser.add_argument('--n_threads', type=int, help='number of concurrent runs / cpu threads to use', default=argparse.SUPPRESS)
parser.add_argument('--n_map_runs', type=int, help='number of VI runs to be conducted with MAP estimations as a starting point', default=argparse.SUPPRESS)
parser.add_argument('--n_vi_standalone_runs', type=int, help='number of VI runs to be conducted \'standalone\', i.e. starting from a random point in the Gaussian prior', default=argparse.SUPPRESS)
parser.add_argument('--pixscale', type=float, help='pixelscale to be used for all runs, in mas/pixel', default=argparse.SUPPRESS)
parser.add_argument('--npix', type=int, help='number of pixels to be used (in both x and y directions)', default=argparse.SUPPRESS)


provided_arguments = vars(parser.parse_args())

if "npix" in provided_arguments:
    provided_arguments["n_pixels_x"] = provided_arguments["npix"]
    provided_arguments["n_pixels_y"] = provided_arguments["npix"]

# Starting by reading and parsing the configuration file
cfg = configparser.ConfigParser()
cfg.read(provided_arguments['config'])


MAX_CONC = provided_arguments['n_threads'] if 'n_threads' in provided_arguments else cfg['base'].getint('n_threads') # Number of concurrent runs / cpu threads to use
TEMPLATE_CONF = cfg['base'].get('imaging_config').strip() # Template imaging config to be used by imaging.py

template_cfg = configparser.ConfigParser()
template_cfg.read(TEMPLATE_CONF)

n_map_runs = provided_arguments['n_map_runs'] if 'n_map_runs' in provided_arguments else cfg['optimization'].getint('n_map_runs')
n_vi_standalone_runs = provided_arguments['n_vi_standalone_runs'] if 'n_vi_standalone_runs' in provided_arguments else cfg['optimization'].getint('n_vi_standalone_runs')

random_seeds = np.random.randint(0, 10000, size=n_map_runs+n_vi_standalone_runs)


data_store_dir = "./ms_data"
os.makedirs(data_store_dir, exist_ok=True)
data_file = provided_arguments.get('data_file', 'None') if 'data_file' in provided_arguments else cfg['observation'].get('data_file', 'None').strip()

if data_file.lower() != 'none':
    source, date, visibility_type, _ = get_source_date_type(data_file)
    get_ms_data_path(data_store_dir, uvf_file=data_file)
else:
    source, date, visibility_type, _ = cfg['observation']['source_name'].strip(), cfg['observation']['date'].strip(), cfg['observation']['visibility_type'].strip(), False
    get_ms_data_path(data_store_dir, source=source, date=date, visibility_type=visibility_type)

os.makedirs("./tmp_configs", exist_ok=True)


def submit_run(idx):
    map_flag = True if idx < n_map_runs else False

    tmp_conf = pathlib.Path(f"./tmp_configs/conf_{idx}_{source}_{date}.cfg")

    content = open(TEMPLATE_CONF).read()

    for base_key in cfg.keys():
        for attribute in cfg[base_key].keys():
            if template_cfg[base_key].get(attribute, "None").strip()[:2] == "__":
                replacement = provided_arguments.get(attribute) if attribute in provided_arguments else cfg[base_key].get(attribute)
                content = content.replace(template_cfg[base_key][attribute], str(replacement))

    content = content.replace("__SEED__", str(random_seeds[idx]))
    content = content.replace("__MAP__", str(map_flag))

    tmp_conf.write_text(content)

    proc = subprocess.run(["python","imaging.py","--config",str(tmp_conf)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tmp_conf.unlink(missing_ok=True)
    return proc.returncode

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONC) as ex:
    futures = {ex.submit(submit_run, i): seed for i, seed in enumerate(random_seeds)}
    for fut in concurrent.futures.as_completed(futures):
        seed = futures[fut]
        try:
            rc = fut.result()
        except Exception as e:
            print("FAILED", seed, e)