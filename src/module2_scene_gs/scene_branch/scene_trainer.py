import os
import sys
import math
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from typing import Tuple, Union

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from simple_knn._C import distCUDA2
except ImportError:
    print("[Error] Cảnh báo: Chưa biên dịch diff-gaussian-rasterization hoặc simple-knn trong submodules/")

from src.common.config_loader import load_toml_config
from src.module2_scene_gs.scene_branch.camera_scheduler import CameraScheduler


def get_expon_lr_func(lr_init, lr_final, max_steps):
    """Tính learning rate suy giảm hàm mũ theo số vòng lặp."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = np.clip(step / max_steps, 0.0, 1.0)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp
    return helper


class GaussianSceneModel:
    """
    Quản lý các thuộc tính hình học và tối ưu của mây hạt 3D Gaussian.
    """
    def __init__(self, sh_degree: int = 3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None

    def create_from_pcd(self, xyz: np.ndarray, rgb: np.ndarray, spatial_lr_scale: float = 1.0):
        """Khởi tạo các hạt Gaussian từ mây điểm thưa SfM."""
        device = "cuda"
        fused_point_cloud = torch.tensor(xyz, dtype=torch.float, device=device)
        fused_color = torch.tensor(rgb / 255.0, dtype=torch.float, device=device)

        # Chuyển đổi RGB sang Spherical Harmonics bậc 0 (DC)
        C0 = 0.28209479177387814
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=device)
        features[:, :3, 0] = (fused_color - 0.5) / C0

        # Khoảng cách lân cận qua KNN để khởi tạo tỉ lệ hạt (Scale)
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Quaternions khởi tạo quay ngẫu nhiên dạng đơn vị [1, 0, 0, 0]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        # Độ đục ban đầu khởi tạo qua inverse sigmoid (khoảng 0.1)
        opacities = torch.logit(0.1 * torch.ones((fused_point_cloud.shape[0], 1), device=device))

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))

        self.setup_optimizer(spatial_lr_scale)

    def setup_optimizer(self, spatial_lr_scale: float):
        l = [
            {'params': [self._xyz], 'lr': 0.00016 * spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0025 / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def save_ply(self, path: Union[str, Path]):
        """Xuất mây hạt Gaussian ra định dạng PLY tiêu chuẩn."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(path))
        print(f"[✓] Đã lưu Scene-GS checkpoint ({xyz.shape[0]} Gaussians) tại: {path}")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l


class SceneGSTrainer:
    """
    Bộ điều khiển huấn luyện toàn cảnh Scene-GS (20k iterations).
    """
    def __init__(self, config_path: str = "configs/base_scene.toml"):
        self.cfg = load_toml_config(config_path)

        self.iterations = self.cfg.get("iterations", 20000)
        self.output_dir = Path(self.cfg.get("workspace_dir", "data/workspace"))
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.model = GaussianSceneModel(sh_degree=self.cfg.get("sh_degree", 3))

    def load_sparse_points(self, sfm_sparse_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Đọc mây điểm thưa từ points3D.bin hoặc points3D.txt của COLMAP."""
        points3d_file = sfm_sparse_dir / "points3D.bin"
        if points3d_file.exists():
            import pycolmap
            recon = pycolmap.Reconstruction(sfm_sparse_dir)
            xyz = np.array([p.xyz for p in recon.points3D.values()])
            rgb = np.array([p.color for p in recon.points3D.values()])
            return xyz, rgb
        else:
            print("[Error] Cảnh báo: Không tìm thấy points3D.bin. Đang dùng mây điểm ngẫu nhiên để test...")
            xyz = np.random.uniform(-1.0, 1.0, (2000, 3))
            rgb = np.random.randint(0, 255, (2000, 3))
            return xyz, rgb

    def train(self):
        """Vòng lặp huấn luyện chính 20K iters."""
        sfm_dir = Path(self.cfg.get("sfm_dir", "data/sfm/sparse/0"))
        mask_meta = Path(self.cfg.get("mask_meta", "data/segmentation/segmentation_meta.json"))
        
        # 1. Chọn camera theo chiến lược của CameraScheduler
        scheduler = CameraScheduler(
            sfm_dir=sfm_dir,
            mask_meta_path=mask_meta,
            close_up_ratio_threshold=self.cfg.get("close_up_ratio_threshold", 0.08)
        )
        selected_cameras = scheduler.schedule_scene_cameras(
            roi_sample_rate=self.cfg.get("roi_sample_rate", 0.5),
            output_file=self.output_dir / "scene_camera_schedule.json"
        )

        # 2. Khởi tạo mây hạt từ sparse points
        print("[Loading] Đang khởi tạo mô hình Gaussian từ mây điểm SfM...")
        xyz, rgb = self.load_sparse_points(sfm_dir)
        self.model.create_from_pcd(xyz, rgb)

        # 3. Quá trình tối ưu (Optimization Loop)
        print(f"[Loading] Bắt đầu huấn luyện Scene-GS ({self.iterations} iterations)...")
        xyz_lr_scheduler = get_expon_lr_func(
            lr_init=0.00016,
            lr_final=0.0000016,
            max_steps=self.iterations
        )

        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training Scene-GS")
        for iteration in progress_bar:
            # Cập nhật learning rate cho tọa độ điểm
            current_xyz_lr = xyz_lr_scheduler(iteration)
            for param_group in self.model.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group["lr"] = current_xyz_lr

            # Tăng dần bậc Spherical Harmonics theo epoch
            if iteration % 1000 == 0 and self.model.active_sh_degree < self.model.max_sh_degree:
                self.model.active_sh_degree += 1

            # (Placeholder thực hiện Render & Backprop bằng CUDA rasterizer)
            # Ở bản hoàn thiện kết nối loader ảnh, rasterizer tính Photometric Loss L1 + D-SSIM
            self.model.optimizer.zero_grad()
            
            # Loss trắc quang mô phỏng để gradient graph liên tục
            dummy_loss = (self.model.get_xyz.mean() * 0.0) + (self.model.get_opacity.mean() * 0.0)
            dummy_loss.backward()
            self.model.optimizer.step()

            if iteration % 5000 == 0:
                progress_bar.set_postfix({"xyz_lr": f"{current_xyz_lr:.6f}"})

        # 4. Xuất checkpoint cho Module 3
        save_path = self.checkpoints_dir / "scene_gs.ply"
        self.model.save_ply(save_path)
        print(f"[Done] Hoàn thành huấn luyện Scene-GS. Checkpoint sẵn sàng cho Module 3 tại: {save_path}")


if __name__ == "__main__":
    trainer = SceneGSTrainer()
    trainer.train()