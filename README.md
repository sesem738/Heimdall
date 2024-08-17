# Heimdall: Using LLMs as Zero-Shot Global Planners

## Project To-Do List

**Gemini: Vision Language Model**
- [ ] Telemetry Overlay Trial
- [ ] Test What Kind of Telemetry Data Produces The Best Result
- [ ] How to reduce video/audio tokens?
- [ ] What prompt engineering is needed?

**Data Processing**
- [ ] Get GPS RTK Working
- [ ] Preprocess SCAND dataset
- [ ] Find other relevant datasets
- [ ] Florida Tech Dataset

**End-to-End Navigation Algorithms**
- [ ] Diffusion Policy
- [ ] Action Chunking Transformer
- [ ] Offline Reinforcement Learning

## Overview
This repository contains code for training a GNM with your own data and pre-trained model checkpoints:

- `./train/train.py`: training script to train or fine-tune the GNM model on your custom data.
- `./train/process_*.py`: scripts to process rosbags or other formats of robot trajectories into training data.

### Pre-requisites

The codebase assumes access to a workstation running Ubuntu (tested on 18.04 and 20.04), Python 3.7+, and a GPU with CUDA 10+. It also assumes access to conda, but you can modify it to work with other virtual environment packages, or a native setup.
### Setup
Run the commands below inside the `drive-any-robot/` (topmost) directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train/environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate gnm_train
    ```
3. Install the gnm_train packages:
    ```bash
    pip install -e train/
    ```

#### Data Processing 

We provide some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s. You can also manually add your own dataset by following our structure below (if you are adding a custom dataset, please checkout the [Custom Datasets](#custom-datasets) section).
2. Run `data_split.py` on your dataset folder with the relevant args.

After step 1 of data processing, the processed dataset should have the following structure:

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

Each `*.jpg` file contains an forward-facing RGB observation from the robot, and they are temporally labeled. The `traj_data.pkl` file is the odometry data for the trajectory. It’s a pickled dictionary with the keys:
- `"position"`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
- `"yaw"`: An np.ndarray [T,] of the yaws of the robot at each image observation.


After step 2 of data processing, the processed data-split should the following structure inside `gnm_release/train/gnm_train/data/data_splits/`:

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### Training your GNM
Run this inside the `gnm_release/train` directory:
```bash
python train.py -c <path_of_train_config_file>
```
The premade config yaml files are in the `train/config` directory. 

#### Custom Config Files
You can use one of the premade yaml files as a starting point and change the values as you need. `config/gnm/gnm_public.yaml` is good choice since it has commented arguments. `config/defaults.yaml` contains the default config values (don't directly train with this config file since it does not specify any datasets for training).

#### Custom Datasets
Make sure your dataset and data-split directory follows the structures provided in the [Data Processing](#data-processing) section. Locate `train/gnm_train/data/data_config.yaml` and append the following:

```
<dataset_name>:
    metric_waypoints_distance: <average_distance_in_meters_between_waypoints_in_the_dataset>
```

Locate your training config file and add the following text under the `datasets` argument (feel free to change the values of `end_slack`, `goals_per_obs`, and `negative_mining`):
```
<dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # how many timesteps to cut off from the end of each trajectory  (in case many trajectories end in collisions)
    goals_per_obs: 1 # how many goals are sampled per observation
    negative_mining: True # negative mining from the ViNG paper (Shah et al.)
```