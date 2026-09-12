import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _list_sample_subfolders(data_dir):
    return [
        os.path.join(data_dir, subfolder)
        for subfolder in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, subfolder)) and not subfolder.startswith(".")
    ]



class PhotonSimulationDataset_GPU_Optimized(Dataset):
    def __init__(self, data_dir, same_tissue=False):
        self.data_dir = data_dir
        self.same_tissue = same_tissue

        self.subfolders = _list_sample_subfolders(data_dir)

        # Cache the tissue volume.
        self.cached_tissue = None
        if self.same_tissue:
            print(f"[Dataset] Pre-loading shared Tissue data from {data_dir}...")
            tissue_path = os.path.join(self.data_dir, "Tissue_Compressed.npz")
            tissue_np = np.load(tissue_path)['arr_0']
            self.cached_tissue = torch.from_numpy(tissue_np).long()
            print("[Dataset] Shared Tissue data loaded successfully.")

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        subfolder_path = self.subfolders[idx]

        # Load tissue as integer indices.
        if self.same_tissue:
            tissue = self.cached_tissue
        else:
            tissue_np = np.load(os.path.join(subfolder_path, "Tissue_Compressed.npz"))['arr_0']
            tissue = torch.from_numpy(tissue_np).long()

        # Load the mask as floating-point values.
        mask_np = np.load(os.path.join(subfolder_path, "Mask_Compressed.npz"))['arr_0']
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)

        # Load the label as floating-point values.
        abs_np = np.load(os.path.join(subfolder_path, "Absorption_AvgCompressed.npz"))['arr_0']
        label = torch.from_numpy(abs_np).float()

        return tissue, mask, label


def prepare_batch_on_gpu(tissue, mask, num_channels):
    """One-hot encode tissue and concatenate the mask on the device."""
    tissue_oh = F.one_hot(tissue, num_classes=num_channels)
    tissue_oh = tissue_oh.permute(0, 4, 1, 2, 3).float()
    model_input = torch.cat([tissue_oh, mask], dim=1)
    return model_input
