# greedy_imaging_resolve

Scripts for computationally expensive self-calibration and Stokes I imaging with resolve. Best used for specific source and date with careful examination of the results, (negative log-)likelihood values are provided for possible automatic comparison. Note that smaller values correspond to better fit with the data.

Main idea is running $N+M$ VIs with different random seeds, where $N$ are started from the corresponding MAP estimation, and $M$ are started from the Gaussian prior. 

## Setup

Prerequisites: Python version 3.11 or later, C++17 capable compiler (e.g. g++7 or later)

To install all required packages, run


```bash
./install.sh
```

## Run

To start, specify desired parameters in `main_conf.cfg`, and run 


```bash
python run_greedy_imaging.py
```

Modify `imaging_conf.cfg` only if you want to change the correlation model for images and gains.

Some important parameters can alternatively be set through the command line interface:

```bash
python run_greedy_imaging.py --data_file=/path/to/uvf_file.uvf_raw_edt --dir_name=2012_15GHz --n_threads=8 --pixscale=0.05 --npix=512 --n_map_runs=5 --n_vi_standalone_runs=3
```

Options (also accesible via `python run_greedy_imaging.py --help`)


- `--config`: path to the main configuration file. Default: `./main_cong.cfg`
- `--data_file`: path to the uvf data file. The data should be pre-calibrated and (ideally) coherently averaged. The file should contain the standard header with the source name and date of the observation
- `--dir_name`: name of the sub-directory where the reults of the run will be stored. Path for images is `{root_dir}/images/{source_name}/dir_name`. Path for the output files is `{root_dir}/output_files/{source_name}/{dir_name}`. If not provided, name of the data file will be used 
- `--n_threads`: number of concurrent runs / cpu threads to use
- `--n_map_runs`: number of VI runs to be conducted with MAP estimation as a starting point
- `--n_vi_standalone_runs`: number of VI runs to be conducted 'standalone', i.e. starting from a random point in the Gaussian prior
- `--pixscale`: pixelscale to be used for all runs, in mas/pixel
- `--npix`: number of pixels to be used (in both x and y directions)

If some of these parameters are not set with the command line interface, their values will be taken from the main config file. You can review and change these default values in `main_conf.cfg`.

For adequate results, please make sure that the data is pre-calibrated and (ideally) coherently averaged beforehand. This can for instance be achieved by using `uvf_raw_edt` or `uvf` formats provided by MOJAVE. Note that the script was extensively tested only for MOJAVE, using other data can lead to unexpected errors. You are welcome to raise these issues and I will try my best to include support for other data. 

Also note that the script will perform weighted averaging on all IF channels in the data.


## Analysis

The run files, useful for the subsequent analysis, are saved at `{root_output_directory}/output_files/{source_name}/{dir_name}/seed_{seed}`, with the final mean images in fits format saved individually for every run at `{root_output_directory}/output_files/{source_name}/{dir_name}/fits_images`

The final png images and movies are saved at `{root_output_directory}/images/{source_name}/{dir_name}`.

The logs are saved at `{root_output_directory}/logs`.

Some basic information about the results, as well as likelihood values, are saved as a csv file at `{root_output_directory}/logs/csv_files/{source_name}_{dir_name}.csv`

By default, `root_output_directory=./results` and `dir_name` is set to the name of the data file.
