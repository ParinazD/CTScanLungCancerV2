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
import matplotlib.pyplot as plt

# query the database for a specific scan
# 'LIDC-IDRI-0001' -- first patient in the dataset
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0013').first()
#84, 
#import pdb; pdb.set_trace()

# cluster the annotations
# In LIDC, up to 4 radiologists looked at the same scan.
# cluster_annotations() groups their individual circles into unique 'physical nodules'.
nodules = scan.cluster_annotations()

print(f"Patient {scan.patient_id} has {len(nodules)} unique nodules.")

# access the 3D volume (The CT scan pixels converted to HU)
vol = scan.to_volume() # This is a 3D NumPy array (Z, Y, X)

# explore a specific nodule
for i, nodule in enumerate(nodules):
    # 'nodule' is a list of annotations (one from each radiologist)
    # Let's look at the first radiologist's assessment of this nodule
    ann = nodule[0]

    print(f"Nodule {i+1} Characteristics:")
    print(f" - Malignancy: {ann.malignancy} (1=Low, 5=High)")
    print(f" - Spiculation: {ann.spiculation}")

    # creates an GUI (running locally)
    #  shows the nodule with arrows and contours.
    scan.visualize(annotation_groups=nodules)
    break

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
    half_size = cube_size // 2
    z, y, x = centroid

    # 1. Get the consensus mask and its bounding box from pylidc
    # This combines the opinions of all radiologists into one "truth"
    cmask, bbox = pl.utils.consensus(cluster, ret_itrs=False)

    # 2. Create a full-volume-sized empty mask (filled with 0s)
    # Using np.bool8 to save memory before cropping
    full_mask = np.zeros(scan.to_volume().shape, dtype=np.bool8)
    full_mask[bbox] = cmask

    # 3. Crop the mask using the same centroid as the CT cube
    mask_patch = full_mask[z-half_size:z+half_size, 
                           y-half_size:y+half_size, 
                           x-half_size:x+half_size]

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

# This ensures pylidc is actually looking at your data before the loop starts
try:
    print(f"Checking .pylidrc path: {pl.config.get('dicom', 'path')}")
except:
    print("WARNING: Could not read .pylidrc! pylidc might hang.")

manifest = [] 

for n in range(1, 123):
    patient_id = f"LIDC-IDRI-{n:04d}"
    print(f"--- Processing {patient_id} ---")

    try:
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        if not scan: 
            continue

        # Check for existing data to avoid re-processing
        check_file = os.path.join(NEG_DIR, f"pat{n:04d}_neg_0_lab_0.0.npy")
        if os.path.exists(check_file):
            print(f"--- Skipping {patient_id}: Already processed ---")
            continue

        # Load volume ONCE per patient
        vol = scan.to_volume()
        nodules = scan.cluster_annotations()

        # 1. Generate Positive Samples (and their corresponding Masks)
        # pos_data should rdeturn (cube, label, center, original_nodule_index)
        pos_data = generate_positive_samples(vol, nodules, CUBE_SIZE)
        for i, (cube, label, center) in enumerate(pos_data):
            # Save the Voxel Cube
            fname = f"pat{n:04d}_pos_{i}_lab_{label:.1f}.npy"
            fpath = os.path.join(POS_DIR, fname)
            np.save(fpath, cube)

            # --- MASK SAVING LOGIC ---
            # Use the actual nodule index from the cluster to generate the mask
            # Pass 'scan' and 'vol' to avoid redundant loading inside the function
            mask = generate_mask_patch(scan, nodules[i], center, CUBE_SIZE)
            
            mask_fname = f"pat{n:04d}_mask_{i}.npy"
            mask_fpath = os.path.join(MASK_DIR, mask_fname)
            np.save(mask_fpath, mask)
            # -------------------------

            manifest.append({
                "patient": patient_id, 
                "file": fname, 
                "mask_file": mask_fname, 
                "label": label, 
                "type": "positive"
            })

        # 2. Generate and Save Negative Samples
        neg_data = generate_negative_samples(vol, nodules, num_samples=5, cube_size=CUBE_SIZE)
        
        for i, (cube, label, center) in enumerate(neg_data):
            fname = f"pat{n:04d}_neg_{i}_lab_0.0.npy"
            fpath = os.path.join(NEG_DIR, fname)
            np.save(fpath, cube)

            manifest.append({
                "patient": patient_id, 
                "file": fname, 
                "mask_file": None, 
                "label": 0.0, 
                "type": "negative"
            })

    except Exception as e:
        print(f"Skipping {patient_id} due to error: {e}")

# Save the manifest for easy training later
df = pd.DataFrame(manifest)
df.to_csv("scan_manifest.csv", index=False)
print("Done! All samples saved and scan_manifest.csv created.")


# **Feeding raw HU values directly into a Neural Network creates two significant technical hurdles:**
# 1. *Exploding Gradients:* 'Neural networks are mathematically optimized to process data
#  in a small, constrained range (typically $[0.0, 1.0]$ or $[-1.0, 1.0]$). Using large, 
# high-variance integers like $+400$ or $-1000$ can cause "exploding gradients" during 
# backpropagation, making the model's training unstable or preventing it from converging at all.
# 2. *Filtering Irrelevant Data:* A CT scan captures the entire density spectrum, from empty air to dense cortical bone. When screening for lung cancer, features like rib bones ($+700$ HU) are irrelevant noise that can distract the model.

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
folders_to_process = {
    "LungVoxels/NoduleVoxel": "LungVoxels/NoduleVoxel_normalized",
    "LungVoxels/HealthyVoxelData": "LungVoxels/HealthyVoxelData_normalized"
}


for input_dir, output_dir in folders_to_process.items():

    # Setup Directories
    os.makedirs(output_dir, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_dir, "*.npy"))

    print(f"\n--- Processing Folder: {input_dir} ---")
    print(f"Found {len(file_paths)} files. Saving to: {output_dir}")

    # Loop through every file in the current folder
    for path in file_paths:
        file_name = os.path.basename(path)
        save_path = os.path.join(output_dir, file_name)

        # Skip if already exists to save time
        if os.path.exists(save_path):
            continue
        try:
            # Load, Normalize, and Save
            raw_cube = np.load(path)
            norm_cube = normalize_cube(raw_cube)
            np.save(save_path, norm_cube)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"Finished normalizing {input_dir}!")

# Final Verification 
print("\n--- Running Final Verification ---")
for _, output_dir in folders_to_process.items():
    test_files = os.listdir(output_dir)
    if test_files:
        sample_path = os.path.join(output_dir, test_files[0])
        sample = np.load(sample_path)
        print(f"Folder: {output_dir}")
        print(f"  - Sample: {test_files[0]}")
        print(f"  - Range: [{sample.min()}, {sample.max()}]")
        print(f"  - Dtype: {sample.dtype}")


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

    def __init__(self, csv_file, pos_dir, neg_dir):
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

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        Returns:
            int: Total count of files.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads a single 3D cube and its label by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            cube (torch.Tensor): A 3D tensor of shape (1, 32, 32, 32).
            label (torch.Tensor): A scalar tensor containing the label (0 or 1).
        """
        # get cube from index (CSV row)
        row = self.df.iloc[idx]
        file_name = row['file']
        is_positive = row['type'] == 'positive'

        # which folder to look in
        folder = self.pos_dir if is_positive else self.neg_dir
        file_path = os.path.join(folder, file_name)

        # load  numpy array
        # It was saved as (32, 32, 32)
        # we need to add a "Channel" dimension (how the image "data" is quantified)
        # PyTorch expects: (Channel, Depth, Height, Width)
        array = np.load(file_path)

        # unsqueeze(0) turns (32,32,32) into (1,32,32,32)
        cube = torch.from_numpy(array).unsqueeze(0).float()

        # extract malignancy label (healthy= 0.0.) 
        # If it's a nodule, keep the 1-5 rating
        label_value = float(row['label'])
        label = torch.tensor(label_value).float()

        return cube, label


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
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 256)
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
