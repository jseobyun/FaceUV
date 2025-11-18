import torch
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import trimesh

def pointmap_mask_to_mesh(points, mask, colors=None, max_edge=None):
    """
    points: (H, W, 3) float 3D 좌표 (NaN/inf 허용, 내부에서 제거)
    mask:   (H, W) bool 또는 {0,1} 유효 픽셀 마스크 (True만 사용)
    colors: (H, W, 3) uint8 RGB 선택 사항 (vertex color)
    max_edge: float | None  삼각형의 모든 변 길이가 이 값 이하일 때만 유지 (노이즈/점프 제거)
    return: trimesh.Trimesh
    """
    H, W, _ = points.shape
    mask = mask.astype(bool)

    # 유효 좌표(유한값)만 남기기
    finite = np.isfinite(points).all(axis=-1)
    mask = mask & finite

    # (i,j) 인덱스 그리드
    vidx = np.arange(H*W, dtype=np.int64).reshape(H, W)

    # 두 개의 삼각형 패턴에 대해, 마스크가 모두 True인 셀만 채택
    # tri1: (i,j)-(i+1,j)-(i,j+1)
    t1 = mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:]
    v0 = vidx[:-1,:-1][t1]
    v1 = vidx[1: ,:-1][t1]
    v2 = vidx[:-1,1: ][t1]
    faces1 = np.stack([v0, v1, v2], axis=1)

    # tri2: (i+1,j)-(i+1,j+1)-(i,j+1)
    t2 = mask[1:,:-1] & mask[1:,1:] & mask[:-1,1:]
    v0 = vidx[1: ,:-1][t2]
    v1 = vidx[1: ,1: ][t2]
    v2 = vidx[:-1,1: ][t2]
    faces2 = np.stack([v0, v1, v2], axis=1)

    faces = np.concatenate([faces1, faces2], axis=0)

    # 정점 배열
    vertices = points.reshape(-1, 3)

    # 선택: 엣지 길이 기반 클리핑 (깊이 점프/홀 경계 제거에 유용)
    if max_edge is not None and len(faces) > 0:
        tri = vertices[faces]                          # (F,3,3)
        # 세 변 길이
        e01 = np.linalg.norm(tri[:,0] - tri[:,1], axis=1)
        e12 = np.linalg.norm(tri[:,1] - tri[:,2], axis=1)
        e20 = np.linalg.norm(tri[:,2] - tri[:,0], axis=1)
        keep = (e01 <= max_edge) & (e12 <= max_edge) & (e20 <= max_edge)
        faces = faces[keep]

    # vertex color 붙이기 (선택)
    visual = None
    if colors is not None:
        vc = colors.reshape(-1, 3) # (255*colors.reshape(-1, 3)).astype(np.uint8)
        # 마스크가 False인 정점 색은 무시되지만, 어차피 faces가 참조 안 함
        visual = trimesh.visual.ColorVisuals(vertex_colors=vc)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=True)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    # o3d_mesh = o3d.geometry.TriangleMesh()
    # o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1,3).astype(np.float32))
    # o3d_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([o3d_mesh])
    return mesh

class FaceDepthVisualizer:
    """Utility class for visualizing face depth estimation results."""
    
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
    
    def visualize_prediction(
        self,
        image_path: str,
        pred_depth: torch.Tensor,    
        pred_mask: torch.Tensor,            
        save_path: Optional[str] = None
    ) -> np.ndarray:
        # Convert to numpy        
        pred_depth = FaceDepthVisualizer.tensor_to_numpy(pred_depth)
        pred_mask = FaceDepthVisualizer.tensor_to_numpy(pred_mask)
        # Get depth colormaps        
        img_h, img_w = pred_depth.shape[:2]
        
        # Load and resize input image
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)                
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
        image_np = cv2.resize(image_np, dsize=(img_w, img_h))        
        
        mask_np = image_np[:,:,3]
        image_np = image_np[:,:,:3]

        u, v = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h), indexing="xy")
        cx = img_w/2
        cy = img_h/2
        f = img_w/2

        x = (u-cx) / f
        y = (v-cy) / f
        z = pred_depth        
        points = np.concatenate([x[:,:,None], y[:,:,None], z[:,:,None]], axis=-1)
        valid = pred_mask >0.3
        # colors = image_np[valid].reshape(-1, 3).astype(np.float32)/255.0
        # points = points[valid].reshape(-1, 3)        

        mesh = pointmap_mask_to_mesh(points, valid, colors=image_np, max_edge=None)
        inner_box_edge = 50.0
        points = np.asarray(mesh.vertices)
        points = points * (inner_box_edge / 2.0)
        points[:, 1:] *= -1
        points[:, 2] -= points[:, 2].mean()
        mesh.vertices = points
        if save_path is not None:
            mesh.export(save_path)

        # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
        # pcd.colors = o3d.utility.Vector3dVector(colors)
    

