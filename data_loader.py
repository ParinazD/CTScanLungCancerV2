#!/usr/bin/env python
# coding: utf-8


folder_path = "./ImagesDICOM/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192" 

import numpy as np
# 1. Fix compatability between modern NumPy and older pylidc
np.int = int

import configparser

# 2. Fix for Python 3.12+ (SafeConfigParser removed)
if not hasattr(configparser, 'SafeConfigParser'):
    configparser.SafeConfigParser = configparser.ConfigParser


import pylidc as pl
import pylidc as pl
import pylidc.utils
import matplotlib.pyplot as plt

# CREATING HEALTHY CUBE AND NODULE CUBES

import os
import numpy as np
import random
import pandas as pd # Useful for the manifest

BASE_DIR = "LungVoxels"
POS_DIR = os.path.join(BASE_DIR, "NoduleVoxel")
NEG_DIR = os.path.join(BASE_DIR, "HealthyVoxelData")
CUBE_SIZE = 32

os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# Add a directory for masks
MASK_DIR = os.path.join(BASE_DIR, "NoduleMasks")
os.makedirs(MASK_DIR, exist_ok=True)

def generate_mask_patch(scan, cluster, centroid, cube_size=32):
    """
    Creates a 3D binary mask for a nodule centered at a specific centroid.

    @params scan: (pylidc.Scan) current CT scan object
    @params cluster: (list) pylidc annotations for nodule
    @params centroid: (tuple) (z, y, x) center of the cube.
    @params cube_size: (int)  size of the output cube (default 32).
    @returns mask_cube: (np.ndarray) Binary 3D array of shape (32, 32, 32).
    """
    half = cube_size // 2
    z, y, x = centroid
    cmask, bbox, _ = pl.utils.consensus(cluster)
    
    full_mask = np.zeros(scan.to_volume().shape, dtype=np.bool8)
    full_mask[bbox] = cmask

    # Use np.pad or manual boundary checks to ensure exactly cube_size
    # This creates a "Safety Buffer" around the scan edges
    padded_mask = np.pad(full_mask, half, mode='constant', constant_values=0)
    
    # Adjust coordinates for padding
    z, y, x = z + half, y + half, x + half
    
    mask_patch = padded_mask[z-half:z+half, 
                             y-half:y+half, 
                             x-half:x+half]
    return mask_patch.astype(np.float32)

def generate_positive_samples(vol, clusters, cube_size=32):
    half_size = cube_size // 2
    samples = []
    for cluster in clusters:
        centroid = np.mean([ann.centroid for ann in cluster], axis=0).astype(int)
        z, y, x = centroid
        cube = vol[z-half_size:z+half_size, y-half_size:y+half_size, x-half_size:x+half_size]

        if cube.shape == (cube_size, cube_size, cube_size):
            label = np.mean([ann.malignancy for ann in cluster])
            samples.append((cube, label, centroid))
    return samples


def generate_negative_samples(vol, nodules, num_samples=5, cube_size=32):
    half_size = cube_size // 2
    neg_cubes = []
    nodule_centers = [np.mean([ann.centroid for ann in cluster], axis=0) for cluster in nodules]

    attempts = 0
    while len(neg_cubes) < num_samples and attempts < 100:
        attempts += 1
        z = random.randint(half_size, vol.shape[0] - half_size)
        y = random.randint(half_size, vol.shape[1] - half_size)
        x = random.randint(half_size, vol.shape[2] - half_size)
        random_center = np.array([z, y, x])

        # Ensure we aren't near a nodule
        too_close = any(np.linalg.norm(random_center - nc) < 40 for nc in nodule_centers)

        if not too_close:
            cube = vol[z-half_size:z+half_size, y-half_size:y+half_size, x-half_size:x+half_size]
            if np.mean(cube) > -900: # Ensure we are inside the lung/body
                neg_cubes.append((cube, 0.0, random_center)) # Label 0.0 for healthy
    return neg_cubes

# pylidc is actually looking at data before the loop starts
print("GOT HERE____________")
try:
    print(f"Checking .pylidrc path: {pl.config.get('dicom', 'path')}")
except:
    print("WARNING: Could not read .pylidrc! pylidc might hang.")

manifest = [] 
for n in range(1, 123):
    patient_id = f"LIDC-IDRI-{n:04d}"
    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if not scan: 
        continue

    # Check for existing data: if this file exists, we skip the heavy volume loading
    check_file = os.path.join(NEG_DIR, f"pat{n:04d}_neg_0_lab_0.0.npy")
    if os.path.exists(check_file):
        print(f"--- Skipping {patient_id}: Already processed ---")
        continue

    print(f"--- Processing {patient_id} ---")
    vol = scan.to_volume()
    nodules = scan.cluster_annotations()

    # Positive Samples & Masks
    for i, cluster in enumerate(nodules):
        centroid = np.mean([ann.centroid for ann in cluster], axis=0).astype(int)
        z, y, x = centroid
        half = CUBE_SIZE // 2
        
        cube = vol[z-half:z+half, y-half:y+half, x-half:x+half]
        
        if cube.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE):
            label = np.mean([ann.malignancy for ann in cluster])
            fname = f"pat{n:04d}_pos_{i}_lab_{label:.1f}.npy"
            np.save(os.path.join(POS_DIR, fname), cube)

            mask = generate_mask_patch(scan, cluster, centroid, CUBE_SIZE)
            mask_fname = f"pat{n:04d}_mask_{i}.npy"
            np.save(os.path.join(MASK_DIR, mask_fname), mask)

    # Negative Samples
    neg_data = generate_negative_samples(vol, nodules, num_samples=5, cube_size=CUBE_SIZE)
    for i, (cube, label, center) in enumerate(neg_data):
        fname = f"pat{n:04d}_neg_{i}_lab_0.0.npy"
        np.save(os.path.join(NEG_DIR, fname), cube)

#NORMALIZING CUBES

import os
import glob
import numpy as np

def normalize_cube(cube):
    """
    Standardizes a 3D CT patch for Deep Learning.
    - Clips values to the Lung Window (-1000 to +400).
    - Scales values to [0.0, 1.0].
    """
    MIN_HU = -1000.0
    MAX_HU = 400.0
    cube = np.clip(cube, MIN_HU, MAX_HU)
    cube = (cube - MIN_HU) / (MAX_HU - MIN_HU)
    return cube.astype(np.float32)

# --- Define Folder Mappings ---
# Format: "Input_Folder": "Output_Folder"
# --- 2. NORMALIZATION LOOP ---
# This loop scales HU values to [0.0, 1.0] for the Neural Network
folders_to_process = {
    POS_DIR: os.path.join(BASE_DIR, "NoduleVoxel_normalized"),
    NEG_DIR: os.path.join(BASE_DIR, "HealthyVoxelData_normalized")
}

for input_dir, output_dir in folders_to_process.items():
    os.makedirs(output_dir, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_dir, "*.npy"))
    print(f"Normalizing {len(file_paths)} files in {input_dir}...")

    for path in file_paths:
        file_name = os.path.basename(path)
        save_path = os.path.join(output_dir, file_name)
        if not os.path.exists(save_path):
            raw_cube = np.load(path)
            norm_cube = normalize_cube(raw_cube)
            np.save(save_path, norm_cube)

# Final Verification 
print("\n--- Generating Final Manifest ---")
final_manifest = []

# Crawl Positive Normalized folder
norm_pos_dir = folders_to_process[POS_DIR]
for fname in os.listdir(norm_pos_dir):
    if fname.endswith(".npy"):
        # Extract patient num and index from fname: pat0001_pos_0_lab_3.0.npy
        parts = fname.split("_")
        pat_id = f"LIDC-IDRI-{parts[0].replace('pat', '')}"
        idx = parts[2]
        label = float(parts[4].replace(".npy", ""))
        mask_fname = f"{parts[0]}_mask_{idx}.npy"
        
        final_manifest.append({
            "patient": pat_id,
            "file": fname,
            "mask_file": mask_fname,
            "label": label,
            "type": "positive"
        })

# Crawl Negative Normalized folder
norm_neg_dir = folders_to_process[NEG_DIR]
for fname in os.listdir(norm_neg_dir):
    if fname.endswith(".npy"):
        parts = fname.split("_")
        pat_id = f"LIDC-IDRI-{parts[0].replace('pat', '')}"
        
        final_manifest.append({
            "patient": pat_id,
            "file": fname,
            "mask_file": None,
            "label": 0.0,
            "type": "negative"
        })

df = pd.DataFrame(final_manifest)
df.to_csv("scan_manifest.csv", index=False)
print(f"Success! scan_manifest.csv created with {len(df)} entries.")

import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset

class LungNoduleDataset(Dataset):
    """
    loads 3D lung cubes and their labels from a CSV manifest.

    Attributes:
        df (pd.DataFrame) - created from manifest
        pos_dir (str)
        neg_dir (str)
    """

    def __init__(self, csv_file, pos_dir, neg_dir, train=True):
        """
        Initializes the dataset.

        Args:
            csv_file (str) -- 'scan_manifest.csv'.
            pos_dir (str)
            neg_dir (str)
        """
        self.df = pd.read_csv(csv_file)
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.train = train # New flag

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        Returns:
            int: Total count of files.
        """
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file']
        is_positive = row['type'] == 'positive'

        # 1. Load the 3D Cube (The Input)
        folder = self.pos_dir if is_positive else self.neg_dir
        cube_array = np.load(os.path.join(folder, file_name))

        # 2. Load the 3D Mask (The Target)
        if is_positive and row['mask_file']:
            # Load the actual Nodule Mask saved in NoduleMasks directory
            mask_path = os.path.join("LungVoxels/NoduleMasks", row['mask_file'])
            mask_array = np.load(mask_path)
        else:
            # For healthy samples, create a blank (zero) mask
            mask_array = np.zeros_like(cube_array)

        # --- 3D DATA AUGMENTATION ---
        # Only apply during training to help the model generalize
        if self.train:
            # Randomly flip along the Z, Y, or X axis
            for axis in [0, 1, 2]:
                if random.random() > 0.5:
                    cube_array = np.flip(cube_array, axis=axis).copy()
                    mask_array = np.flip(mask_array, axis=axis).copy()
            
            # Random 90-degree rotations
            if random.random() > 0.5:
                k = random.randint(1, 3) # Number of 90-degree rotations
                axes = random.sample([0, 1, 2], 2) # Pick two random axes for rotation
                cube_array = np.rot90(cube_array, k, axes).copy()
                mask_array = np.rot90(mask_array, k, axes).copy()

        # Convert to PyTorch Tensors
        # unsqueeze(0) adds the 'Channel' dimension: (1, 32, 32, 32)
        cube = torch.from_numpy(cube_array).unsqueeze(0).float()
        mask = torch.from_numpy(mask_array).unsqueeze(0).float()

        return cube, mask
        

import torch
import torch.nn as nn
import torch.nn.functional as F 

class NoduleRegressionModel3D(nn.Module): 
    """
    A 3D CNN designed to predict the malignancy 
    severity score (0.0 to 5.0) of a lung volume patch.
    """

    def __init__(self):
        """
        Initializes the 3D CNN architecture layers.
        """
        super(NoduleRegressionModel3D, self).__init__()

        # feature extraction
        # input layer (1 channel, 32x32x32) -> Output (32 channels)
        # kernel_size=3 with padding=1 keeps the spatial dimensions the same.
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32) # normalizes layer outputs -- expedites training process

        # input layer (32 channels) -> Output (64 channels)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        # input layer (64 channels) -> Output (128 channels)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # max pooling - reduces spatial size by half (2x2x2)
        # 3 pools == 32x32x32 --> 4x4x4
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) #translation invariant

        # input size: 128 channels * (4 * 4 * 4 voxels) = 8192
        self.fc1 = nn.linear(128 * 4 * 4 * 4, 256)
        self.dropout = nn.Dropout(p=0.3) # PRevents overfitting by randomly 'dropping off' neurons
        self.fc2 = nn.Linear(256, 1)    # Final output: A single continuous value (the score)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): A batch of 3D cubes of shape (Batch, 1, 32, 32, 32).

        Returns:
            torch.Tensor: A batch of predicted malignancy scores.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the 3D feature maps into a 1D vector
        x = x.view(x.size(0), -1) 
        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # the raw predicted score (which could be 0.0, 2.5, 4.8, etc.)
        x = self.fc2(x)

        return x
