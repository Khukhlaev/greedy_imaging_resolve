import os 

import subprocess, concurrent.futures, pathlib
from datetime import datetime
import configparser
import argparse

from utils.utilities import get_ms_data_path

import numpy as np


# Setting up argument parser to accept configuration file path
parser = argparse.ArgumentParser(description='Run imaging pipeline for multiple sources and dates.')
parser.add_argument('--config', type=str, help='path to the main configuration file.', default="main_conf.cfg")
args = parser.parse_args()

# Starting by reading and parsing the configuration file
cfg = configparser.ConfigParser()
cfg.read(args.config)


MAX_CONC = cfg['base'].getint('n_threads') # Number of concurrent runs / cpu threads to use
TEMPLATE_CONF = cfg['base'].get('imaging_config').strip() # Template imaging config to be used by imaging.py

template_cfg = configparser.ConfigParser()
template_cfg.read(TEMPLATE_CONF)

n_map_runs = cfg['optimization'].getint('n_map_runs')
n_vi_standalone_runs = cfg['optimization'].getint('n_vi_standalone_runs')

random_seeds = np.random.randint(0, 10000, size=n_map_runs+n_vi_standalone_runs)


data_store_dir = "./ms_data"
os.makedirs(data_store_dir, exist_ok=True)
get_ms_data_path(data_store_dir, cfg['observation']['source_name'].strip(), cfg['observation']['date'].strip(), "uvf_raw_edt")

os.makedirs("./tmp_configs", exist_ok=True)


def submit_run(idx):
    map_flag = True if idx < n_map_runs else False

    tmp_conf = pathlib.Path(f"./tmp_configs/conf_{idx}_{cfg['observation']['source_name']}_{cfg['observation']['date']}.cfg")

    content = open(TEMPLATE_CONF).read()

    for base_key in cfg.keys():
        for attribute in cfg[base_key].keys():
            if template_cfg[base_key].get(attribute, "None").strip()[:2] == "__":
                content = content.replace(template_cfg[base_key][attribute], str(cfg[base_key][attribute]))

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