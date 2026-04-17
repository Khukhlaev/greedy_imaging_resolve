import os 

import subprocess, concurrent.futures
import configparser
import argparse

from pathlib import Path

from utils.utilities import get_source_date_type, get_observation
from utils.image_helper import create_movie

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

if data_file.lower() == 'none':
    source, date, visibility_type = cfg['observation']['source_name'].strip(), cfg['observation']['date'].strip(), cfg['observation']['visibility_type'].strip()
    gz_flag = "" if visibility_type == "uvf" else ".gz"
    data_file = f"/aux/zeall/2cmVLBA/data/{source}/{date}/{source}.u.{date}.{visibility_type}{gz_flag}"
else:
    source, date, visibility_type, _ = get_source_date_type(data_file)


# Transforming data to ms format
proc = subprocess.run(["python","transform_data.py","--data_file",str(data_file)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

ms_data_path = f"./ms_data/{source}/{source}.u.{date}.{visibility_type}.ms"
if os.path.exists(ms_data_path):
    print("Sucessfully transformed data to ms format. Starting the imaging.")
else:
    print("Error transforming the data to ms format. Check casa log file.")
    sys.exit()
    
# Removing casa log files
for path in Path(".").glob("casa-*.log"):
    path.unlink(missing_ok=True)

# Hardcode the polarization: "stokesi", as we want to have Stokes I images. This works for modern epochs, but sometimes fail for old ones. In that case, we will use polarizations="all"
polarizations = "stokesi"
try:
    get_observation("./ms_data", source, date, visibility_type, polarizations)

except Exception as e:
    print(f"Cannot load polarizations=\"stokesi\". Attempting to use polarizations=\"all\". Error: {e}")
    polarizations = "all"
    try:
        get_observation("./ms_data", source, date, visibility_type, polarizations)

    except Exception as e:
        print(f"Cannot load polarizations=\"all\". Cannot image the provided data. Error: {e}")

        sys.exit()

# Adding arguments to the dictionary that will be used to replace the placeholders in the template config
provided_arguments['source_name'] = source
provided_arguments['date'] = date
provided_arguments['visibility_type'] = visibility_type

os.makedirs("./tmp_configs", exist_ok=True)


def submit_run(idx):
    map_flag = True if idx < n_map_runs else False

    tmp_conf = Path(f"./tmp_configs/conf_{idx}_{source}_{date}.cfg")

    content = open(TEMPLATE_CONF).read()

    for base_key in cfg.keys():
        for attribute in cfg[base_key].keys():
            if template_cfg[base_key].get(attribute, "None").strip()[:2] == "__":
                replacement = provided_arguments.get(attribute) if attribute in provided_arguments else cfg[base_key].get(attribute)
                content = content.replace(template_cfg[base_key][attribute], str(replacement))

    content = content.replace("__SEED__", str(random_seeds[idx]))
    content = content.replace("__MAP__", str(map_flag))
    content = content.replace("__POLARIZATIONS__", polarizations)

    tmp_conf.write_text(content)

    proc = subprocess.run(["python","imaging.py","--config",str(tmp_conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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



create_movie(root_dir=cfg['base']['root_output_directory'], source=source, date=date, fps=0.5)