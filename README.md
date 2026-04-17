# greedy_imaging_resolve

Scripts for computationally expensive self-calibration and Stokes I imaging with resolve. Best used for specific source and date, with the examination of the results by eye, but (negative log-)likelihood values are provided for possible automatic comparison. Note that smaller values correspond to better fit with the data.

Main idea is running $N+M$ VIs with different random seeds, where $N$ are started from the corresponding MAP estimation, and $M$ are started from the Gaussian prior. 

## Setup

Prerequisites: Python version 3.11 or later, C++17 capable compiler (e.g. g++7 or later)

To install all required packages, run


```bash
./install.sh
```

## Run

To start, specify the desired parameters in `main_conf.cfg`, and run 


```bash
python run_greedy_imaging.py
```

Modify `imaging_conf.cfg` only if you want to change the correlation model for images and gains.

Some important parameters can alternatively be set through the command line interface:

```bash
python run_greedy_imaging.py --data_file=/path/to/uvf_file.uvf_raw_edt --n_threads=8 --pixscale=0.05 --npix=512 --n_map_runs=5 --n_vi_standalone_runs=3
```

Options (also accesible via `python run_greedy_imaging.py --help`)


- `--config`: path to the main configuration file. Default: `./main_cong.cfg`
- `--data_file`: path to the uvf data file. If not provided, the script will attempt to use source and date in the config to load corresponding MOJAVE data
- `--n_threads`: number of concurrent runs / cpu threads to use
- `--n_map_runs`: number of VI runs to be conducted with MAP estimations as a starting point
- `--n_vi_standalone_runs`: number of VI runs to be conducted 'standalone', i.e. starting from a random point in the Gaussian prior
- `--pixscale`: pixelscale to be used for all runs, in mas/pixel
- `--npix`: number of pixels to be used (in both x and y directions)

If some of these parameters are not set with the command line interface, their values will be taken from the main config file. 

Two options for providing the data are possible:

1. Specify the `data_file` field in the main config or via the command line interface with the path to the uvf or uvf_raw_edt data file 
2. Comment `data_file` field out (or set to None) and specify source name and date in the config. In that case, the script will attempt to load the corresponding MOJAVE observations (assuming you have access to the VLBI group data storage system)

Please note that if no number of pixels is provided (are equal to 0 in the config), the script will attempt to use the number of pixels from the corresponding MOJAVE observation, and fail if it does not exist.

Also note that the script will perform weighted averaging on all IF channels in the data. 

Support for the `uvf_raw` data is currently not available. 

## Analysis

The run files, useful for the subsequent analysis, are saved at `{root_output_directory}/output_files/{source_name}/{date}/seed_{seed}`.

The final images are saved at `{root_output_directory}/images/{source_name}/{date}`.

The logs are saved at `{root_output_directory}/logs`.

By default, `root_output_directory=./results`
