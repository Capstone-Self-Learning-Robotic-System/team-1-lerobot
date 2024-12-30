"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Loading a dataset and accessing its properties.
- Filtering data by episode number.
- Converting tensor data for visualization.
- Saving video files from dataset frames.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pathlib import Path
from pprint import pprint

import imageio
import torch

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import save_lerobot_dataset_on_disk

import h5py
import pandas as pd
import time

from datasets import Dataset

print("List of available datasets:")
print(lerobot.available_datasets)

# Let's take one for this example
repo_id = "lerobot/pusht"

# You can easily load a dataset from a Hugging Face repository
dataset = LeRobotDataset(repo_id)

# LeRobotDataset is actually a thin wrapper around an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets/index for more information).
#print(dataset)
print(dataset.hf_dataset)

save_lerobot_dataset_on_disk(dataset)

'''

df = dataset.hf_dataset.to_pandas()

print(df)

df.to_hdf('./examples/dataset.h5', key='data', mode='w')

df_new = pd.read_hdf('./examples/dataset.h5', key='data')

dataset_new = LeRobotDataset.df_new




# And provides additional utilities for robotics and compatibility with Pytorch
print(f"\naverage number of frames per episode: {dataset_new.num_samples / dataset.num_episodes:.3f}")
print(f"frames per second used during data collection: {dataset_new.fps=}")
print(f"keys to access images from cameras: {dataset_new.camera_keys=}\n")

'''