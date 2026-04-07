# greedy_imaging_resolve

Scripts for computationally expensive and effective imaging with resolve. Best used for specific source and date, with the examination of the results by eye.

## Setup

Prerequisites: Python version 3.10 or later, C++17 capable compiler (e.g. g++7 or later)

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


## Analysis

The run files, useful for the subsequent analysis, are saved at `{root_output_directory}/output_files/{source_name}/{date}/seed_{seed}`.

The final images are saved at `{root_output_directory}/images/{source_name}/{date}`.

The logs are saved at `{root_output_directory}/logs`.
