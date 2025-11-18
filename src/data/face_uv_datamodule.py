import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms.functional as tvf
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset

class MergedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,               
        bg_dir : str,
        split: str = 'train',                
    ):
        super().__init__()

        self.data_root = data_dir,
        dataset_names = [
            "facescape",
            "faceverse",
            "h3ds",
            "nphm",
            "polygom",
            "rp",
            "th"
        ]

        self.img_paths = []
        self.u_paths = []
        self.v_paths = []
        self.ref_path  = os.path.join(data_dir, "flame_texturemap.jpg")        

        print("Loading MergedFace paths...")
        for dataset_name in dataset_names:
            img_root = os.path.join(data_dir, dataset_name, "images")
            uv_root = os.path.join(data_dir, dataset_name, "uvs")

            subj_names = sorted(os.listdir(img_root))
            for subj_name in subj_names:
                img_dir = os.path.join(img_root, subj_name)
                uv_dir = os.path.join(uv_root, subj_name)

                if not os.path.exists(uv_dir):
                    continue

                if 2*len(os.listdir(img_dir)) != len(os.listdir(uv_dir)):
                    print(f"{img_dir} is skipped")
                    continue

                img_names = sorted(os.listdir(img_dir))
                for img_name in img_names:
                    img_path = os.path.join(img_dir, img_name)
                    u_path = os.path.join(uv_dir, img_name.split(".")[0] + "_u.png")
                    v_path = os.path.join(uv_dir, img_name.split(".")[0] + "_v.png")

                    if os.path.exists(img_path) and os.path.exists(u_path) and os.path.exists(v_path):
                        self.img_paths.append(img_path)
                        self.u_paths.append(u_path)
                        self.v_paths.append(v_path)

        bg_names = sorted(os.listdir(bg_dir))
        self.bg_paths = [os.path.join(bg_dir, bg_name) for bg_name in bg_names]

        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # For validation/test, sample fewer images
        if split != "train":
            self.img_paths = self.img_paths[::200]            
            self.u_paths = self.u_paths[::200]
            self.v_paths = self.v_paths[::200]

        assert len(self.img_paths) == len(self.u_paths)
        assert len(self.img_paths) == len(self.v_paths)
        print(f"MergedDataset : Loaded {len(self.img_paths)} samples for {split}")

    def __len__(self):
        return len(self.img_paths)

    def load_img(self, img_path, postprocess=True):
        if postprocess:
            img = cv2.imread(img_path)
            img_h, img_w = np.shape(img)[:2]
            mask_path = img_path.replace("images", "depths").replace(".jpg", ".png")
            depth = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = (depth !=0).astype(np.float32)
            bg_path = np.random.choice(self.bg_paths, 1)[0]
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, dsize=(img_w, img_h))

            invalid = mask ==0
            img[invalid] = bg[invalid]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_path)
        return img

    def load_uv(self, uv_path):
        """Load UV coordinate map"""
        # UV maps are stored as 3-channel images where:
        # - R channel: U coordinate
        # - G channel: V coordinate
        # - B channel: unused or alpha
        uv = cv2.imread(uv_path, cv2.IMREAD_UNCHANGED)
        if len(uv.shape) == 3:
            uv = uv[:, :, :2]  # Take only U and V channels
        uv = uv.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        return uv    

    def normalize_uv(self, uv, mask):
        """Normalize UV coordinates"""
        invalid = mask == 0
        # Normalize to [-1, 1] range for better training
        uv_normalized = uv * 2.0 - 1.0
        uv_normalized[invalid] = -2.0

        return uv_normalized
    
    def rand_shake(self, *things, shake_t=20):
        t = np.random.choice(range(-shake_t, shake_t + 1), size=2)
        angle = float(np.random.choice(range(-20, 20+1), size=1)[0])
        return [
            tvf.affine(thing, angle=angle, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ]

    def __getitem__(self, index):
        img_path = self.img_paths[index]        
        u_path = self.u_paths[index]
        v_path = self.v_paths[index]

        img = self.load_img(img_path, postprocess=True)        
        u = self.load_uv(u_path)
        v = self.load_uv(v_path)
        uv = np.concatenate([u[:,:,None], v[:,:,None]], axis=2)
        mask = (u != 0) * (v != 0)
        mask = mask.astype(np.float32)



        uv_normalized = self.normalize_uv(uv, mask)

        img_tensor = torch.from_numpy(np.asarray(img).astype(np.float32)/255.0).permute(2,0,1) # 3 H W
        mask_tensor = torch.from_numpy(mask).unsqueeze(dim=0) # 1 H W
        uv_tensor = torch.from_numpy(uv_normalized.astype(np.float32)).permute(2,0,1)  # 2 H W

        img_tensor, uv_tensor, mask_tensor = self.rand_shake(img_tensor, uv_tensor, mask_tensor)
        img_tensor = self.color_jitter(img_tensor)
        img_tensor = self.normalize(img_tensor)

        gts = {
            "uv" : uv_tensor,
            "mask" : mask_tensor,
        }
        return img_tensor, gts



class FaceUVDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Face UV Coordinate Estimation task.
    Uses MergedDataset for loading face UV data.
    """

    def __init__(
        self,
        data_dir: str,
        bg_dir: str,
        img_size: Tuple[int, int] = (448, 448),
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.bg_dir = bg_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augmentation = augmentation

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each split."""
        if stage == 'fit' or stage is None:
            # Setup training dataset
            self.train_dataset = MergedDataset(
                data_dir=self.data_dir,
                bg_dir=self.bg_dir,
                split='train',                                
            )

            # Setup validation dataset
            self.val_dataset = MergedDataset(
                data_dir=self.data_dir,
                bg_dir=self.bg_dir,
                split='val',               
            )

        if stage == 'test' or stage is None:
            # Setup test dataset
            self.test_dataset = MergedDataset(
                data_dir=self.data_dir,
                bg_dir=self.bg_dir,
                split='test',
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def get_output_channels(self):
        """Return the number of output channels for UV estimation."""
        return 3  # UV map has 2 channels (U, V) + 1 mask channel



if __name__ == "__main__":
    # Example usage of MergedDataset
    dataset = MergedDataset(
        data_dir="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/head_uvd_new",        
        bg_dir="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/no_humans",        
        split="train"
    )

    for data in dataset:
        print("")
    print(f"Dataset loaded with {len(dataset)} samples")
