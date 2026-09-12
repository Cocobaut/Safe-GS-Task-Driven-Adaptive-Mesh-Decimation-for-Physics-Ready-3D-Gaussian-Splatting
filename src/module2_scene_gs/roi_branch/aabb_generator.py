import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import open3d as o3d


class AABBGenerator:
    """
    Sinh 3D Bounding Box (AABB, OBB) và phân vùng 3D Voxel Grid (Micro, Boundary, Macro)
    phục vụ huấn luyện hạn chế hạt và gộp lưới thích ứng.
    """
    def __init__(
        self,
        margin_ratio: float = 0.1,         # Mở rộng hộp bao 10% mỗi chiều để tránh cắt rách biên Gaussian
        voxel_size: float = 0.02,          # Kích thước mỗi voxel (ví dụ 2cm cho vật thể thao tác)
        boundary_margin_ratio: float = 0.2 # Vùng đệm ranh giới cho Watertight sewing
    ):
        self.margin_ratio = margin_ratio
        self.voxel_size = voxel_size
        self.boundary_margin_ratio = boundary_margin_ratio

    def compute_bounds(self, xyz: np.ndarray) -> Dict:
        """
        Tính toán AABB và OBB từ tập điểm 3D của vật thể.
        """
        if len(xyz) == 0:
            raise ValueError("Tập điểm 3D rỗng, không thể tính toán Bounding Box.")

        # 1. Tính toán AABB chuẩn
        min_bound = np.min(xyz, axis=0)
        max_bound = np.max(xyz, axis=0)
        extent = max_bound - min_bound
        center = (min_bound + max_bound) / 2.0

        # Mở rộng lề AABB an toàn
        margin = extent * self.margin_ratio
        inflated_min = min_bound - margin
        inflated_max = max_bound + margin
        inflated_extent = inflated_max - inflated_min

        # 2. Vùng biên tiếp giáp (Boundary Zone) để nối mesh watertight
        boundary_margin = extent * self.boundary_margin_ratio
        boundary_min = min_bound - boundary_margin
        boundary_max = max_bound + boundary_margin

        # 3. Tính toán OBB (Oriented Bounding Box) qua PCA bằng Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        obb = pcd.get_oriented_bounding_box()

        bounds_info = {
            "center": center.tolist(),
            "raw_extent": extent.tolist(),
            "raw_min": min_bound.tolist(),
            "raw_max": max_bound.tolist(),
            "aabb": {
                "min": inflated_min.tolist(),
                "max": inflated_max.tolist(),
                "extent": inflated_extent.tolist()
            },
            "boundary_aabb": {
                "min": boundary_min.tolist(),
                "max": boundary_max.tolist()
            },
            "obb": {
                "center": obb.center.tolist(),
                "extent": obb.extent.tolist(),
                "rotation_matrix": obb.R.tolist()
            }
        }
        return bounds_info

    def generate_voxel_grid(self, bounds_info: Dict) -> Dict:
        """
        Khởi tạo và gán nhãn cho mạng lưới Voxel 3D trong không gian bao quanh.
        - MICRO (1): Nằm trọn trong AABB vật thể.
        - BOUNDARY (2): Nằm giữa AABB vật thể và Boundary AABB.
        - MACRO (0): Bên ngoài.
        """
        b_min = np.array(bounds_info["boundary_aabb"]["min"])
        b_max = np.array(bounds_info["boundary_aabb"]["max"])

        # Số lượng voxel theo từng trục
        grid_dims = np.ceil((b_max - b_min) / self.voxel_size).astype(int)

        voxel_meta = {
            "voxel_size": self.voxel_size,
            "origin_min": b_min.tolist(),
            "grid_dims": grid_dims.tolist(),
            "labels_legend": {
                "0": "SCENE_MACRO",
                "1": "ROI_MICRO",
                "2": "BOUNDARY_ZONE"
            }
        }
        return voxel_meta

    def process_and_save(
        self,
        roi_points_path: Union[str, Path],
        output_json_path: Union[str, Path]
    ) -> Dict:
        """
        Đọc điểm ROI từ file PLY, tính toán bao và lưu metadata JSON.
        """
        roi_points_path = Path(roi_points_path)
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        pcd = o3d.io.read_point_cloud(str(roi_points_path))
        xyz = np.asarray(pcd.points)

        bounds_info = self.compute_bounds(xyz)
        voxel_meta = self.generate_voxel_grid(bounds_info)

        final_output = {
            "object_name": roi_points_path.stem,
            "num_points": len(xyz),
            "bounds": bounds_info,
            "voxel_grid": voxel_meta
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

        print(f"[Done] Đã tạo và lưu 3D ROI metadata tại: {output_json_path}")
        print(f"    - AABB Min: {bounds_info['aabb']['min']}")
        print(f"    - AABB Max: {bounds_info['aabb']['max']}")
        return final_output


if __name__ == "__main__":
    # Doc duong dan tu configs/object_roi.toml thay vi hard-code (thay doi
    # duong dan khi chuyen may chi can sua file config, khong sua code).
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open("configs/object_roi.toml", "rb") as f:
        _cfg = tomllib.load(f)
    _roi_boxes_dir = Path(_cfg.get("workspace_dir", "data/workspace")) / "roi_boxes"
    roi_points_file = _roi_boxes_dir / "roi_points.ply"
    output_metadata = _roi_boxes_dir / "roi_metadata.json"

    if Path(roi_points_file).exists():
        generator = AABBGenerator(
            margin_ratio=_cfg.get("margin_ratio", 0.1),
            voxel_size=_cfg.get("voxel_size", 0.02),
            boundary_margin_ratio=_cfg.get("boundary_margin_ratio", 0.2),
        )
        generator.process_and_save(roi_points_file, output_metadata)
    else:
        print(f"[Error] Chưa có file {roi_points_file}. Vui lòng chạy mask_to_3d.py trước.")