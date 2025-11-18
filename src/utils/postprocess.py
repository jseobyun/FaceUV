import torch
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import trimesh


def bayer_matrix(n=8):
    if n == 2:
        return np.array([[0, 2],
                         [3, 1]], dtype=np.float32)
    b2 = bayer_matrix(n // 2)
    return np.block([
        [4*b2 + 0, 4*b2 + 2],
        [4*b2 + 3, 4*b2 + 1],
    ])

class FaceDepthConverter:
    def __init__(self, output_dim=512, bayer_n=8):
        self.output_dim = output_dim
        self.bayer_n = bayer_n   
    
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Handle different tensor shapes
        if len(tensor.shape) == 4:  # Batch dimension
            tensor = tensor[0]
        
        if len(tensor.shape) == 3:  # Channel dimension
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            elif tensor.shape[0] == 3:
                # Denormalize if needed (ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
                tensor = tensor.clamp(0, 1)
                tensor = tensor.permute(1, 2, 0)
        
        return tensor.numpy()
    
    def gray2dot(self, gray):
        pil = Image.fromarray((gray * 255).astype(np.uint8))
        pil = pil.resize((self.output_dim, self.output_dim), resample=Image.BILINEAR)
        gray = np.asarray(pil, dtype=np.float32) / 255.0

        B = bayer_matrix(self.bayer_n)
        T = (B + 0.5) / (self.bayer_n * self.bayer_n)  # 0~1
        Th = np.tile(T, (self.output_dim // self.bayer_n + 1, self.output_dim // self.bayer_n + 1))[:self.output_dim, :self.output_dim]

        binary = (gray >= Th).astype(np.uint8)
        return binary 
    
    def write_minimal_point_dxf(self, path, points, colors=None, layer="0", creator_comment="Created by GSI Studio"):
        """
        points: (N,3) float
        colors: (N,3) uint8 or float in 0..255 (optional). If given, writes TrueColor (group code 420).
        """
        def line(code, value=None):
            return f"  {code}\n{value}\n" if value is not None else f"  {code}\n"

        out = []
        # HEADER (minimal skeleton + comment)
        out += [line(0, "SECTION"), line(2, "HEADER"), line(999, creator_comment), line(0, "ENDSEC")]
        # TABLES / BLOCKS (empty)
        out += [line(0, "SECTION"), line(2, "TABLES"), line(0, "ENDSEC")]
        out += [line(0, "SECTION"), line(2, "BLOCKS"), line(0, "ENDSEC")]
        # ENTITIES
        out += [line(0, "SECTION"), line(2, "ENTITIES")]

        use_color = colors is not None
        if use_color:
            colors = np.asarray(colors)
            if colors.dtype != np.uint8:
                colors = np.clip(colors, 0, 255).astype(np.uint8)

        for i, p in enumerate(points):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            out += [
                line(0, "POINT"),
                line(8, layer),
                line(10, f"{x:.6f}"),
                line(20, f"{y:.6f}"),
                line(30, f"{z:.6f}"),
            ]
            if use_color:
                tc = rgb_to_int24(colors[i])
                out += [line(420, tc)]  # TrueColor

        out += [line(0, "ENDSEC"), line(0, "EOF")]

        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.writelines(out)

    def convert(
        self,
        image_path: str,
        pred_depth: torch.Tensor,    
        pred_mask: torch.Tensor,            
        save_path: Optional[str] = None,
    ) -> np.ndarray:
     
        # Convert to numpy        
        pred_depth = FaceDepthConverter.tensor_to_numpy(pred_depth)
        pred_mask = FaceDepthConverter.tensor_to_numpy(pred_mask)

        pred_depth = cv2.resize(pred_depth, dsize=(self.output_dim, self.output_dim), interpolation=cv2.INTER_LINEAR)
        pred_mask = cv2.resize(pred_mask, dsize=(self.output_dim, self.output_dim), interpolation=cv2.INTER_NEAREST)
                      
        # Load and resize input image
        gray_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                        
        gray_np = cv2.resize(gray_np, dsize=(self.output_dim, self.output_dim))  
        gray_np[pred_mask <=0.5] = 0.0
        gray_np = gray_np.astype(np.float32) / 255.0
        dot_np = self.gray2dot(gray_np)
                              
        u, v = np.meshgrid(np.arange(0, self.output_dim), np.arange(0, self.output_dim), indexing="xy")
        cx = self.output_dim/2
        cy = self.output_dim/2
        f = self.output_dim/2

        x = (u-cx) / f
        y = (v-cy) / f
        z = pred_depth        
        points = np.concatenate([x[:,:,None], y[:,:,None], z[:,:,None]], axis=-1)
        valid = dot_np > 0.5

        points = points[valid].reshape(-1, 3)

        
        dot_np = (255*dot_np).astype(np.uint8)
        cv2.imwrite(save_path.replace(".dxf", ".jpg"), dot_np)

        inner_box_edge = 50.0
        points = points * (inner_box_edge / 2.0)
        points[:, 1:] *= -1
        points[:, 2] -= points[:, 2].mean()


        pcd = trimesh.PointCloud(points)
        pcd.export(save_path.replace(".dxf", ".ply"))
        self.write_minimal_point_dxf(
            path=save_path,
            points=points,
            colors=None,       # 없애면 색 없이 저장됨
            layer="0",
            creator_comment="Created by Polygom Jongseob"
        )   


        
    
