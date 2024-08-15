import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from gnm_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    calculate_sin_cos_for_goal,
    RandomizedClassBalancer,
    get_image_path,
    to_ego_coords,
)


class GNM_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        is_action: bool,
        transform: transforms,
        aspect_ratio: float,
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
    ):
        """
        Main GNM dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            is_action (bool): Whether to use the action dataset or the distance dataset
            transform (transforms): Transforms to apply to the image data
            aspect_ratio (float): Aspect ratio of the images (w/h)
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.is_action = is_action
        self.dataset_name = dataset_name
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.transform = transform
        self.aspect_ratio = aspect_ratio
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining and not self.is_action
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self._gen_index_to_data()

    def _gen_index_to_data(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """

        self.index_to_data = []
        label_balancer = RandomizedClassBalancer(self.distance_categories)  # -1 (if negative mine) range(min,max)
        dataset_type = "action" if self.is_action else "distance"
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_type_{dataset_type}_waypoint_spacing_{self.waypoint_spacing}_min_dist_cat_{self.min_dist_cat}_max_dist_cat_{self.max_dist_cat}_negative_mining_{self.negative_mining}_len_traj_pred_{self.len_traj_pred}_learn_angle_{self.learn_angle}_context_size_{self.context_size}_context_type_{self.context_type}_end_slack_{self.end_slack}_goals_per_obs_{self.goals_per_obs}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f1:
                self.index_to_data = pickle.load(f1)
        except:
            # if the index_to_data file doesn't exist, create it
            print(
                f"Sampling subgoals for each observation in the {self.dataset_name} {dataset_type} dataset..."
            )
            print(
                "This will take a while, but it will only be done once for each configuration per dataset."
            )
            for i in tqdm.tqdm(range(len(self.traj_names))):
                f_curr = self.traj_names[i]
                with open(
                    os.path.join(
                        os.path.join(self.data_folder, f_curr), "traj_data.pkl"
                    ),
                    "rb",
                ) as f3:
                    traj_data = pickle.load(f3)  # contains a dict of position and yaw keys
                traj_len = len(traj_data["position"])  # no. of points in position array
                # start sampling a little bit into the trajectory to give enought time to generate context
                for curr_time in range(   # typically curr_time starts at 5 as this is the context size
                    self.context_size * self.waypoint_spacing,  # 5
                    traj_len - self.end_slack,  # 195
                ):
                    max_len = min(
                        int(self.max_dist_cat * self.waypoint_spacing),
                        traj_len - curr_time - 1,
                    )
                    sampled_dists = []

                    # sample self.goals_per_obs goals per observation
                    for _ in range(self.goals_per_obs):
                        # sample a distance from the distance categories as long as it is less than the trajectory length
                        filter_func = (
                            lambda dist: int(dist * self.waypoint_spacing) <= max_len
                            and dist not in sampled_dists
                        )
                        # picks a random distance from distance_categories and ensures above condition is met
                        len_to_goal = label_balancer.sample(filter_func) 
                        sampled_dists.append(len_to_goal)   

                        # break the loop if there are no more valid distances to sample
                        if len_to_goal is None:
                            break
                        else:
                            goal_time = curr_time + int(
                                len_to_goal * self.waypoint_spacing
                            )
                            f_goal = f_curr
                        self.index_to_data += [(f_curr, f_goal, curr_time, goal_time)]   # output [["name of traj", "name of traj", 5, 11], ["name of traj", "name of traj", 5, 9], ......]]
            with open(index_to_data_path, "wb") as f2:
                pickle.dump(self.index_to_data, f2)  # dump list of tuples in pkl file


    def __len__(self) -> int:  
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation for visualization
                transf_obs_image (torch.Tensor): tensor of shape [(context_size + 1) * 3, H, W] containing the images of the context and the observation after transformation for training
                goal_pose (aka goal_image) (torch.Tensor): tensor of shape (1,2) or (1,3) (if training with angle) containing the goal pose.
                obs_pose (torch.Tensor): tensor of shape (1,2) or (1,4) containing the current pose.
                action_label (torch.Tensor): tensor of shape (num_wpts, 2) or (num_wpts, 4) (if training with angle) containing the poses from the observation to the goal 
                dataset_index (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        # Note: For us, f_curr == f_goal as we aren't using distance head so no need for negative mining
        f_curr, f_goal, curr_time, goal_time = self.index_to_data[i]
        with open(os.path.join(self.data_folder, f_curr, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        traj_len = len(traj_data["position"])
        assert curr_time < traj_len, f"{curr_time} and {traj_len}"

        transf_obs_images = []
        context = []
        if self.context_type == "temporal":  
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid type {self.context_type}")
        for f, t in context:
            obs_image_path = get_image_path(self.data_folder, f, t)
            obs_image, transf_obs_image = img_path_to_data(
                obs_image_path,
                self.transform,
                self.aspect_ratio,
            )
            transf_obs_images.append(transf_obs_image)

        transf_obs_image = torch.cat(transf_obs_images, dim=0)

        spacing = self.waypoint_spacing
        traj_len = min(self.len_traj_pred, (goal_time - curr_time) // spacing)
        pos_goal = traj_data["position"][goal_time, :2]
        pos_list = traj_data["position"][
            curr_time : curr_time + (traj_len + 1) * spacing : spacing,
            :2,
        ]
        # action_label
        if self.learn_angle:
            pos_goal = np.concatenate(
                (
                    pos_goal,
                    np.array(traj_data["yaw"][goal_time]).reshape(1),
                ),
                axis=0,
            )
            pos_list_angle = traj_data["yaw"][
                curr_time : curr_time + (traj_len + 1) * spacing : spacing
            ]
            pos_list = np.concatenate(
                (pos_list, pos_list_angle.reshape(len(pos_list_angle), 1)),
                axis=1,
            )
            param_dim = 3
        else:
            param_dim = 2  # only x,y coords

        goals_appendage = pos_goal * np.ones(
            (self.len_traj_pred - traj_len, param_dim)
        )
        pos_list = np.concatenate((pos_list, goals_appendage), axis=0)
        yaw = traj_data["yaw"][curr_time]
        positions = to_ego_coords(pos_list, pos_list[0], yaw)   # first array is the obs_pose & it will always be [0,0,0] in local coords
        obs_pose = positions[0]
        obs_pose = torch.Tensor(obs_pose.astype(float)) 
        waypoints = positions[1:]
        waypoints = torch.Tensor(waypoints.astype(float))
        goal_pose=to_ego_coords(pos_goal, pos_list[0], yaw)
        goal_pose = torch.Tensor(goal_pose.astype(float))
        if self.learn_angle:  # localize the waypoint and obs_pose angles
                #waypoints[:, 2] -= waypoints[0, 2]
                waypoints = calculate_sin_cos(waypoints)
                goal_pose = calculate_sin_cos_for_goal(goal_pose)
                obs_pose = torch.zeros_like(waypoints[0])  # all 4 values of obs_pose are zero
        if self.normalize:
            waypoints[:, :2] /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )  # only divide the dx and dy
            goal_pose[:2] /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )
        data = [
            obs_image,
            transf_obs_image,  
            goal_pose,
            obs_pose,
            waypoints,  # aka action_label
        ]
    

        data.append(torch.LongTensor([self.dataset_index]))

        return tuple(data)
