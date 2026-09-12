import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import open3d as o3d


class MaskTo3DProjector:
    """
    Chiếu ngược và lọc các điểm 3D từ SfM nằm bên trong mặt nạ 2D (Binary Masks).
    Hỗ trợ đọc dữ liệu camera từ pycolmap hoặc file text COLMAP.
    """
    def __init__(
        self,
        sfm_sparse_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        min_views_consensus: int = 2  # Điểm 3D phải rơi vào mask của ít nhất N camera
    ):
        self.sfm_sparse_dir = Path(sfm_sparse_dir)
        self.masks_dir = Path(masks_dir)
        self.min_views_consensus = min_views_consensus

    def _load_colmap_data(self):
        """
        Nạp dữ liệu Reconstruction từ pycolmap.
        """
        try:
            import pycolmap
            recon = pycolmap.Reconstruction(self.sfm_sparse_dir)
            return recon
        except ImportError:
            raise ImportError("Vui lòng cài đặt pycolmap để đọc định dạng sparse của COLMAP.")

    def extract_roi_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lọc các điểm 3D thuộc vùng vật thể dựa trên phép chiếu lên 2D Masks.
        Trả về (xyz_filtered, rgb_filtered).
        """
        recon = self._load_colmap_data()
        
        # Đọc trước tất cả masks vào RAM dạng nhị phân bool
        loaded_masks: Dict[str, np.ndarray] = {}
        for img_id, image in recon.images.items():
            mask_path = self.masks_dir / f"{Path(image.name).stem}.png"
            if mask_path.exists():
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    loaded_masks[image.name] = (mask_img > 127)

        print(f"[Loading] Đã nạp {len(loaded_masks)} masks để đối chiếu hình chiếu 3D...")

        roi_points = []
        roi_colors = []

        # Duyệt qua từng điểm 3D trong mô hình SfM
        for p3d_id, point in recon.points3D.items():
            consensus_count = 0
            pt_xyz = np.asarray(point.xyz, dtype=np.float64)
            
            # Kiểm tra các camera quan sát thấy điểm này
            for track_elem in point.track.elements:
                if track_elem.image_id not in recon.images:
                    continue
                image = recon.images[track_elem.image_id]
                
                if image.name not in loaded_masks:
                    continue

                camera = recon.cameras[image.camera_id]
                
                # 1. Chuyển đổi tọa độ World -> Camera (p_c)
                p_c = None
                try:
                    if hasattr(image, "cam_from_world"):
                        cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                        p_c = np.asarray(cam_from_world * pt_xyz, dtype=np.float64)
                    elif hasattr(image, "rotation"):
                        p_c = image.rotation.matrix() @ pt_xyz + np.asarray(image.translation, dtype=np.float64)
                    elif hasattr(image, "qvec") and hasattr(image, "tvec"):
                        import pycolmap
                        R = pycolmap.qvec_to_rotmat(image.qvec)
                        p_c = R @ pt_xyz + np.asarray(image.tvec, dtype=np.float64)
                except Exception:
                    continue

                if p_c is None or p_c[2] <= 0.01:  # Nằm sau mặt phẳng tiêu cự camera
                    continue

                # 2. Chiếu điểm lên mặt phẳng ảnh pixel (u, v)
                p_img = None
                try:
                    p_norm = p_c[:2] / p_c[2]
                    if hasattr(camera, "img_from_cam"):
                        p_img = camera.img_from_cam(p_norm)
                    elif hasattr(camera, "cam_to_img"):
                        p_img = camera.cam_to_img(p_norm)
                    elif hasattr(camera, "world_to_image"):
                        p_img = camera.world_to_image(p_norm)
                    elif hasattr(camera, "project"):
                        p_img = camera.project(p_norm)
                except Exception:
                    continue

                if p_img is None:
                    continue

                u, v = int(round(p_img[0])), int(round(p_img[1]))

                mask = loaded_masks[image.name]
                h, w = mask.shape
                if 0 <= u < w and 0 <= v < h:
                    if mask[v, u]:  # Pixel nằm trong mask vật thể
                        consensus_count += 1

            # Giữ lại điểm nếu đạt đủ số góc nhìn đồng thuận
            if consensus_count >= self.min_views_consensus:
                roi_points.append(pt_xyz)
                roi_colors.append(point.color)

        if not roi_points:
            print("[Loading] Cảnh báo: Không tìm thấy điểm 3D nào thỏa mãn consensus. Sử dụng toàn bộ điểm SfM...")
            xyz_all = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float64)
            rgb_all = np.array([p.color for p in recon.points3D.values()], dtype=np.uint8)
            return xyz_all, rgb_all

        xyz_arr = np.array(roi_points, dtype=np.float64)
        rgb_arr = np.array(roi_colors, dtype=np.uint8)

        print(f"[Done] Tìm thấy {len(xyz_arr)} điểm 3D nằm trong ROI (consensus >= {self.min_views_consensus}).")

        # Lọc nhiễu thống kê bằng Open3D để loại bỏ floaters
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_arr)
        pcd.colors = o3d.utility.Vector3dVector(rgb_arr / 255.0)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        clean_xyz = np.asarray(cl.points)
        clean_rgb = (np.asarray(cl.colors) * 255.0).astype(np.uint8)

        print(f"[Done] Sau khi lọc nhiễu Outlier: Giữ lại {len(clean_xyz)} điểm 3D.")
        return clean_xyz, clean_rgb

    def save_roi_pcd(self, output_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xyz, rgb = self.extract_roi_points()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"[Done] Đã lưu ROI Point Cloud tại: {output_path}")
        return xyz, rgb


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

    projector = MaskTo3DProjector(
        sfm_sparse_dir=_cfg.get("sfm_dir", "data/sfm/sparse/0"),
        masks_dir=_cfg.get("masks_dir", "data/segmentation"),
        min_views_consensus=_cfg.get("min_views_consensus", 2)
    )
    projector.save_roi_pcd(_roi_boxes_dir / "roi_points.ply")