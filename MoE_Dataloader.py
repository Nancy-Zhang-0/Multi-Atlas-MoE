#!/usr/bin/env python3
"""
DataLoader for PyTorch
Handles loading of brain connectivity matrices derived from multiple atlases
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# Mapping from atlas name to structural (SC) and functional (FC) file keys stored in JSON
ALL_ATLAS_PATHS: Dict[str, Dict[str, str]] = {
    "AAL": {"SC": "AAL_SC_path", "FC": "AAL_FC_path"},
    "HOA": {"SC": "HOA_SC_path", "FC": "HOA_FC_path"},
    "3Hinge": {"SC": "3Hinge_SC_path", "FC": "3Hinge_FC_path"},
    "Destrieux": {"SC": "Destrieux_SC_path", "FC": "Destrieux_FC_path"},
}

class ADNIDataset(Dataset):
    """
    ADNI Dataset for loading structural and functional connectivity matrices
    """
    
    def __init__(
        self,
        json_file: str,
        atlases: Optional[List[str]] = None,
        transform=None,
    ):
        """
        Initialize ADNI Dataset
        
        Args:
            json_file: Path to JSON file containing subject data
            atlases: Optional subset of atlas names to load (defaults to all)
            transform: Optional transform to be applied on images
        """
        self.json_file = json_file
        self.transform = transform
        self.selected_atlases = self._resolve_atlases(atlases)
        self.atlas_paths = {atlas: ALL_ATLAS_PATHS[atlas] for atlas in self.selected_atlases}
        
        # Load data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Filter data to only include CN and MCI
        self.data = [item for item in self.data if item['DX'] in ['CN', 'MCI']]
        
        # Setup label encoder
        self.label_encoder = LabelEncoder()
        labels = ["CN", "MCI"]
        self.label_encoder.fit(labels)
    
    def _resolve_atlases(self, atlases: Optional[List[str]]) -> List[str]:
        """
        Validate and resolve the list of atlases that should be loaded.
        """
        if atlases is None:
            return list(ALL_ATLAS_PATHS.keys())
        
        invalid = sorted(set(atlases) - set(ALL_ATLAS_PATHS.keys()))
        if invalid:
            raise ValueError(f"Invalid atlas selection: {invalid}. "
                             f"Valid options are {sorted(ALL_ATLAS_PATHS.keys())}")
        return list(atlases)
    
    def _load_and_preprocess_image(
        self,
        file_path: str,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Load and preprocess a single connectivity matrix
        
        Args:
            file_path: Path to the .npy file
            normalize: Whether to apply log/standardization (used for SC matrices only)
            
        Returns:
            Preprocessed numpy array
        """
        try:
            image_data = np.load(file_path).astype(np.float32)

            if not normalize:
                return np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply log transformation: log(x + 1)
            image_data = np.log(image_data + 1)

            # Compute mean and standard deviation
            mean = np.mean(image_data)
            std = np.std(image_data)

            if std != 0:  # To avoid division by zero
                image_data = (image_data - mean) / std
            else:
                image_data = image_data - mean  # If std is zero, only subtract the mean

            return image_data.astype(np.float32)
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load matrix from {file_path}") from exc
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - {atlas}_SC, {atlas}_FC: Connectivity matrices for each selected atlas
                - label: Numeric label (0 for CN, 1 for MCI)
                - dx: Diagnosis string (CN, MCI)
                - mmse: MMSE score
        """
        item = self.data[idx]
        
        # Extract label and convert to numeric
        label = item['DX']
        numeric_label = self.label_encoder.transform([label])[0]
        
        # Load structural (SC) and functional (FC) matrices per selected atlas
        matrices: Dict[str, np.ndarray] = {}
        for atlas in self.selected_atlases:
            for modality, path_key in self.atlas_paths[atlas].items():
                file_path = item[path_key]
                matrices[f"{atlas}_{modality}"] = self._load_and_preprocess_image(
                    file_path,
                    normalize=(modality == "SC"),
                )
        