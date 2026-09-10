import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.common.config_loader import load_toml_config
from src.module2_scene_gs.roi_branch.mask_to_3d import MaskTo3DProjector
from src.module2_scene_gs.roi_branch.aabb_generator import AABBGenerator
from src.module2_scene_gs.roi_branch.view_selection import ROIViewSelector


def main():
    print("=" * 60)
    print(">>> BẮT ĐẦU CHẠY MODULE 2: NHÁNH ROI 3D & SPACE PARTITIONING <<<")
    print("=" * 60)

    cfg_path = "configs/object_roi.toml"
    cfg = load_toml_config(cfg_path)

    workspace = Path(cfg.get("workspace_dir", "data/workspace"))
    roi_boxes_dir = workspace / "roi_boxes"
    roi_boxes_dir.mkdir(parents=True, exist_ok=True)

    roi_pcd_path = roi_boxes_dir / "roi_points.ply"
    roi_meta_path = roi_boxes_dir / "roi_metadata.json"
    roi_camera_schedule = roi_boxes_dir / "roi_camera_schedule.json"

    # Bước 1: Chiếu ngược 2D Mask lọc mây điểm 3D
    print("\n[BƯỚC 1/3] Chiếu ngược 2D Mask lên mây điểm thưa SfM...")
    projector = MaskTo3DProjector(
        sfm_sparse_dir=cfg["sfm_dir"],
        masks_dir=cfg["masks_dir"],
        min_views_consensus=cfg.get("min_views_consensus", 2)
    )
    projector.save_roi_pcd(roi_pcd_path)

    # Bước 2: Tạo 3D AABB & Voxel Grid
    print("\n[BƯỚC 2/3] Ước lượng 3D AABB, OBB và phân chia Voxel Grid...")
    generator = AABBGenerator(
        margin_ratio=cfg.get("margin_ratio", 0.1),
        voxel_size=cfg.get("voxel_size", 0.02),
        boundary_margin_ratio=cfg.get("boundary_margin_ratio", 0.2)
    )
    generator.process_and_save(roi_pcd_path, roi_meta_path)

    # Bước 3: Lọc danh sách góc chụp tối ưu cho ROI
    print("\n[BƯỚC 3/3] Chọn lọc tập góc máy tối ưu cho vùng ROI...")
    selector = ROIViewSelector(
        sfm_sparse_dir=cfg["sfm_dir"],
        roi_metadata_path=roi_meta_path,
        min_keypoints_visible=cfg.get("min_keypoints_visible", 3),
        min_projected_area_ratio=cfg.get("min_projected_area_ratio", 0.01)
    )
    selector.export_schedule(roi_camera_schedule, max_cameras=cfg.get("max_roi_cameras", 150))

    print("\n" + "=" * 60)
    print(">>> HOÀN THÀNH XỬ LÝ NHÁNH ROI 3D THÀNH CÔNG <<<")
    print(f"Metadata sẵn sàng cho Module 3 tại: {roi_meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()