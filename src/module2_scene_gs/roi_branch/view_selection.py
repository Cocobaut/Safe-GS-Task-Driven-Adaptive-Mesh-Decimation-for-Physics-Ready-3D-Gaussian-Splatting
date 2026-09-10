import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pycolmap


def get_cam_pose(image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trích xuất R (3x3), t (3,), và cam_center (3,) bằng đại số ma trận thuần NumPy.
    """
    # 1. pycolmap bản mới: cam_from_world (Rigid3d)
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        rot = cfw.rotation() if callable(cfw.rotation) else cfw.rotation
        R = rot.matrix() if hasattr(rot, "matrix") else np.array(rot)
        t = np.array(cfw.translation() if callable(cfw.translation) else cfw.translation).flatten()
        center = -R.T @ t
        return R, t, center

    # 2. pycolmap có image.rotation
    if hasattr(image, "rotation"):
        rot = image.rotation() if callable(image.rotation) else image.rotation
        R = rot.matrix() if hasattr(rot, "matrix") else np.array(rot)
        trans = image.translation() if callable(image.translation) else image.translation
        t = np.array(trans).flatten()
        center = -R.T @ t
        return R, t, center

    # 3. pycolmap bản cũ dùng qvec và tvec chuẩn COLMAP
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        qvec = image.qvec
        w, x, y, z = qvec[0], qvec[1], qvec[2], qvec[3]
        R = np.array([
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]
        ])
        t = np.array(image.tvec).flatten()
        center = -R.T @ t
        return R, t, center

    raise AttributeError("Không thể đọc pose từ đối tượng Image của pycolmap.")


def project_point_to_cam(camera, R: np.ndarray, t: np.ndarray, pt_world: np.ndarray) -> Optional[np.ndarray]:
    """
    Chiếu điểm 3D từ World Space sang Pixel ảnh tương thích với pycolmap mới.
    """
    # 1. Chuyển sang Camera Space: p_c = R * p_w + t
    p_c = R @ pt_world + t
    if p_c[2] <= 0.05:  # Nằm sau mặt phẳng camera
        return None

    # 2. Sử dụng trực tiếp params [fx, fy, cx, cy] của PINHOLE / OPENCV (nhanh và chuẩn xác nhất)
    if hasattr(camera, "params"):
        params = camera.params
        norm_x = p_c[0] / p_c[2]
        norm_y = p_c[1] / p_c[2]

        if len(params) >= 4:  # PINHOLE, OPENCV: [fx, fy, cx, cy, ...]
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            u = norm_x * fx + cx
            v = norm_y * fy + cy
            return np.array([u, v], dtype=np.float64)
        elif len(params) == 3:  # SIMPLE_PINHOLE / SIMPLE_RADIAL: [f, cx, cy]
            f, cx, cy = params[0], params[1], params[2]
            u = norm_x * f + cx
            v = norm_y * f + cy
            return np.array([u, v], dtype=np.float64)

    # 3. Fallback dùng API pycolmap: img_from_cam nhận vector 3D [X, Y, Z]
    if hasattr(camera, "img_from_cam"):
        try:
            pix = camera.img_from_cam(p_c.astype(np.float64))
            return np.array(pix, dtype=np.float64) if pix is not None else None
        except Exception:
            pass

    if hasattr(camera, "world_to_image"):
        norm_coord = (p_c[:2] / p_c[2]).astype(np.float64)
        return np.array(camera.world_to_image(norm_coord), dtype=np.float64)

    return None


class ROIViewSelector:
    """
    Thuật toán chọn góc nhìn tập trung vào vật thể (Object-Focused Camera Selection)
    dựa trên ROI-GS (Bui et al., 2025).
    """
    def __init__(
        self,
        sfm_sparse_dir: Union[str, Path],
        roi_metadata_path: Union[str, Path],
        min_keypoints_visible: int = 5,
        min_projected_area_ratio: float = 0.01
    ):
        self.sfm_sparse_dir = Path(sfm_sparse_dir)
        self.roi_metadata_path = Path(roi_metadata_path)
        self.min_keypoints_visible = min_keypoints_visible
        self.min_projected_area_ratio = min_projected_area_ratio

        with open(self.roi_metadata_path, "r", encoding="utf-8") as f:
            self.roi_meta = json.load(f)

        self.aabb_min = np.array(self.roi_meta["bounds"]["aabb"]["min"])
        self.aabb_max = np.array(self.roi_meta["bounds"]["aabb"]["max"])
        self.roi_center = np.array(self.roi_meta["bounds"]["center"])

    def _get_aabb_corners(self) -> np.ndarray:
        """Sinh 8 đỉnh của hộp bao 3D AABB."""
        corners = []
        for x in [self.aabb_min[0], self.aabb_max[0]]:
            for y in [self.aabb_min[1], self.aabb_max[1]]:
                for z in [self.aabb_min[2], self.aabb_max[2]]:
                    corners.append([x, y, z])
        return np.array(corners)

    def select_cameras(self, max_cameras: Optional[int] = 150) -> List[Dict]:
        """
        Lọc và xếp hạng các góc nhìn quan sát tốt nhất cho ROI (bản sửa lỗi lọc rỗng).
        """
        recon = pycolmap.Reconstruction(str(self.sfm_sparse_dir))
        aabb_corners = self._get_aabb_corners()

        # Tạo trước tập hợp ID của các điểm 3D nằm trong AABB để tra cứu O(1)
        roi_point3d_ids = set()
        for p3d_id, pt in recon.points3D.items():
            if np.all(pt.xyz >= self.aabb_min) and np.all(pt.xyz <= self.aabb_max):
                roi_point3d_ids.add(p3d_id)

        print(f"[*] Tổng số điểm SfM 3D nằm trong AABB: {len(roi_point3d_ids)}")

        candidate_views = []

        for img_id, image in recon.images.items():
            camera = recon.cameras[image.camera_id]
            R, t, cam_center = get_cam_pose(image)
            distance = float(np.linalg.norm(cam_center - self.roi_center))

            # 1. Chiếu tọa độ 8 đỉnh sang Camera Space
            corners_cam = (R @ aabb_corners.T + t[:, None]).T
            
            # SỬA LỖI 1: Chỉ bỏ qua nếu TẤT CẢ các đỉnh nằm sau camera
            if np.all(corners_cam[:, 2] <= 0.05):
                continue

            # Giữ lại các đỉnh có Z > 0 để chiếu lên ảnh
            valid_corners = aabb_corners[corners_cam[:, 2] > 0.05]
            if len(valid_corners) == 0:
                continue

            proj_pixels = []
            for pt in valid_corners:
                pix = project_point_to_cam(camera, R, t, pt)
                if pix is not None:
                    proj_pixels.append(pix)

            if len(proj_pixels) < 2:
                continue

            proj_pixels = np.array(proj_pixels)
            u_min, v_min = np.min(proj_pixels, axis=0)
            u_max, v_max = np.max(proj_pixels, axis=0)

            # Giới hạn trong kích thước ảnh (Frustum intersection)
            w, h = camera.width, camera.height
            box_u_min = max(0, u_min)
            box_v_min = max(0, v_min)
            box_u_max = min(w, u_max)
            box_v_max = min(h, v_max)

            if box_u_max <= box_u_min or box_v_max <= box_v_min:
                continue  # Nằm hoàn toàn ngoài khung nhìn

            projected_area = (box_u_max - box_u_min) * (box_v_max - box_v_min)
            area_ratio = projected_area / (w * h)

            # SỬA LỖI 2: Hạ ngưỡng diện tích nếu chụp quá xa hoặc nới lỏng kiểm tra
            if area_ratio < self.min_projected_area_ratio:
                continue

            # SỬA LỖI 3: Đếm điểm nhìn thấy qua track elements hoặc point2D tương thích pycolmap
            visible_roi_pts = 0
            if hasattr(image, "points2D"):
                for pt2d in image.points2D:
                    has_p3d = pt2d.has_point3D() if callable(pt2d.has_point3D) else pt2d.has_point3D
                    if has_p3d:
                        p3d_id = pt2d.point3D_id
                        if p3d_id in roi_point3d_ids:
                            visible_roi_pts += 1

            # Nới lỏng: Nếu mây điểm ROI quá ít (dưới ngưỡng), vẫn cho phép lọt qua nếu AABB chiếu tốt
            if len(roi_point3d_ids) >= self.min_keypoints_visible:
                if visible_roi_pts < self.min_keypoints_visible:
                    continue

            score = (area_ratio * 10.0) + math.log(visible_roi_pts + 1) - (0.05 * distance)

            candidate_views.append({
                "image_name": image.name,
                "distance": distance,
                "projected_area_ratio": float(area_ratio),
                "visible_keypoints": int(visible_roi_pts),
                "cam_center": cam_center.tolist(),
                "score": float(score)
            })

        candidate_views.sort(key=lambda x: x["score"], reverse=True)

        if max_cameras is not None and len(candidate_views) > max_cameras:
            candidate_views = candidate_views[:max_cameras]

        print(f"[✓] Đã chọn lọc được {len(candidate_views)} góc máy tối ưu cho ROI.")
        return candidate_views

    def export_schedule(
        self,
        output_path: Union[str, Path],
        max_cameras: Optional[int] = 150
    ) -> List[str]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_data = self.select_cameras(max_cameras=max_cameras)
        camera_names = [item["image_name"] for item in selected_data]

        export_payload = {
            "num_selected": len(camera_names),
            "camera_list": camera_names,
            "detailed_scores": selected_data
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=4)

        print(f"[✓] Đã lưu danh sách Camera ROI tại: {output_path}")
        return camera_names


if __name__ == "__main__":
    sfm_directionary = r"E:\Hcmut material\Project_Safe_GS\tmp\sfm\sparse\0"
    roi_meta = r"E:\Hcmut material\Project_Safe_GS\tmp\workspace\roi_boxes\roi_metadata.json"
    output_schedule = r"E:\Hcmut material\Project_Safe_GS\tmp\workspace\roi_boxes\roi_camera_schedule.json"

    if Path(roi_meta).exists():
        selector = ROIViewSelector(
            sfm_sparse_dir=sfm_directionary,
            roi_metadata_path=roi_meta,
            min_keypoints_visible=1,
            min_projected_area_ratio=0.01
        )
        selector.export_schedule(output_schedule, max_cameras=150)
    else:
        print(f"[!] Chưa có file {roi_meta}. Vui lòng chạy aabb_generator.py trước.")