
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class HepaticVessel2DDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, include_empty=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.include_empty = include_empty
        self.slices = []

        for filename in sorted(os.listdir(images_dir)):
            if not filename.endswith(".nii.gz"):
                continue

            img_path = os.path.join(images_dir, filename)
            msk_path = os.path.join(masks_dir, filename)

            img = nib.load(img_path).get_fdata()
            msk = nib.load(msk_path).get_fdata()

            for i in range(img.shape[2]):
                if self.include_empty or np.max(msk[:, :, i]) > 0:
                    self.slices.append((img_path, msk_path, i))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, msk_path, slice_idx = self.slices[idx]

        img = nib.load(img_path).get_fdata()
        msk = nib.load(msk_path).get_fdata()

        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]

        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)

        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).float()
        msk_tensor = torch.from_numpy((msk_slice > 0).astype(np.float32)).unsqueeze(0)

        if self.transform:
            img_tensor = self.transform(img_tensor)
            msk_tensor = self.transform(msk_tensor)

        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        msk_tensor = F.interpolate(msk_tensor.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        return img_tensor, msk_tensor
