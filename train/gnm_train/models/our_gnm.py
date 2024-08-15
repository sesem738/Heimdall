import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from .modified_mobilenetv2 import MobileNetEncoder
from .base_model import BaseModel

class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        obs_pose_size: Optional[int] = 4,  # Assuming obs pose has 4 values (x, y, sin(yaw), cos(yaw))
        goal_pose_size: Optional[int] = 4,  # Assuming goal pose has 3 values
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to use for context
            len_traj_pred (int): how many waypoints to predict in the future
            observation_encoding_size (int): size of the encoding of the observation images
            obs_pose_size (int): size of the observation pose (e.g., x, y coordinates)
            goal_pose_size (int): size of the goal pose (e.g., x, y coordinates)
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.obs_pose_size = obs_pose_size
        self.goal_pose_size = goal_pose_size

        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        self.compress_obs_pose = nn.Sequential(
            nn.Linear(self.obs_pose_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.obs_pose_size),
            nn.ReLU(),
        )
        self.compress_goal_pose = nn.Sequential(
            nn.Linear(self.goal_pose_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_pose_size),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_pose_size + self.obs_pose_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, obs_pose: torch.tensor, goal_pose: torch.tensor
    ) -> torch.Tensor:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        obs_pose_encoding = self.compress_obs_pose(obs_pose)
        goal_pose_encoding = self.compress_goal_pose(goal_pose)

        z = torch.cat([obs_encoding, obs_pose_encoding, goal_pose_encoding], dim=1)
        z = self.linear_layers(z)
        action_pred = self.action_predictor(z)

        # Augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # Convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # Normalize the angle prediction
        return action_pred
