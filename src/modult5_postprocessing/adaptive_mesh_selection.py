import json
import os
import sys
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roi_config import (
    ADAPTIVE_DIR,
    CURRENT_TASK_ROI_MIN,
    CURRENT_TASK_ROI_MAX,
    SELECTION_DIR,
    SELECTED_MESH,
    SELECTION_REPORT,
    ensure_dir,
    create_subrois,
)


# ============================================================
# STEP 7: ADAPTIVE MESH SELECTION
# Xác định thứ tự mesh cần kiểm tra từ COARSE → FINE
# ============================================================

def box_volume(box_min, box_max):
    size = np.maximum(
        np.asarray(box_max) - np.asarray(box_min),
        0.0
    )
    return float(np.prod(size))


def intersection_volume(min1, max1, min2, max2):
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    return box_volume(inter_min, inter_max)


def main():

    print("=" * 60)
    print("STEP 7 - ADAPTIVE MESH SELECTION")
    print("=" * 60)

    ensure_dir(SELECTION_DIR)

    # --------------------------------------------------------
    # 1. Lấy Task ROI hiện tại
    # --------------------------------------------------------
    task_min = np.asarray(CURRENT_TASK_ROI_MIN)
    task_max = np.asarray(CURRENT_TASK_ROI_MAX)

    task_volume = box_volume(
        task_min,
        task_max
    )

    # --------------------------------------------------------
    # 2. Tạo các Sub-ROI
    # --------------------------------------------------------
    subrois = create_subrois()

    # --------------------------------------------------------
    # 3. Tính mức độ overlap giữa Task ROI và từng Sub-ROI
    # --------------------------------------------------------
    results = []

    for subroi_name, box in subrois.items():

        subroi_min = np.asarray(box["min"])
        subroi_max = np.asarray(box["max"])

        overlap = intersection_volume(
            task_min,
            task_max,
            subroi_min,
            subroi_max
        )

        overlap_ratio = (
            overlap / task_volume
            if task_volume > 0
            else 0.0
        )

        results.append({
            "subroi": subroi_name,
            "ratio": box["ratio"],
            "detail_level": box["detail_level"],
            "overlap_ratio": overlap_ratio,
        })

    # --------------------------------------------------------
    # 4. Xác định Sub-ROI phù hợp nhất
    # --------------------------------------------------------
    # Ưu tiên Sub-ROI có overlap lớn nhất.
    # Nếu bằng nhau → ưu tiên vùng có ratio nhỏ hơn
    # (tức độ phân giải cao hơn).
    selected_subroi = max(
        results,
        key=lambda r: (
            r["overlap_ratio"],
            -r["ratio"]
        )
    )

    print(
        f"\nTarget Sub-ROI: "
        f"{selected_subroi['subroi']}"
    )

    # --------------------------------------------------------
    # 5. Tạo thứ tự candidate COARSE → FINE
    # --------------------------------------------------------
    # Ví dụ:
    #
    # M3 → M2 → M1 → M0
    #
    # Step 8 sẽ kiểm tra lần lượt.
    # Nếu M3 lỗi → thử M2 → M1 → M0.
    # --------------------------------------------------------

    candidate_order = [
        "M3_Coarse.ply",
        "M2_Medium.ply",
        "M1_MediumHigh.ply",
        "M0_HighRes.ply",
    ]

    print("\nCandidate order:")
    for mesh_name in candidate_order:
        print("  →", mesh_name)

    # --------------------------------------------------------
    # 6. Kiểm tra file candidate tồn tại
    # --------------------------------------------------------
    candidates = []

    for mesh_name in candidate_order:

        mesh_path = os.path.join(
            ADAPTIVE_DIR,
            mesh_name
        )

        if os.path.exists(mesh_path):
            candidates.append(mesh_name)

    if not candidates:
        raise RuntimeError(
            "No adaptive mesh candidate found."
        )

    # --------------------------------------------------------
    # 7. Lưu thông tin để Step 8 sử dụng
    # --------------------------------------------------------
    report = {
        "task_roi": {
            "min": task_min.tolist(),
            "max": task_max.tolist(),
            "volume": task_volume,
        },

        "selected_subroi": selected_subroi,

        "candidate_order": candidates,

        "selection_rule":
            "Coarse-to-fine validation: M3 -> M2 -> M1 -> M0",

        "note":
            "Step 7 defines candidate order. "
            "Step 8 performs quality validation."
    }

    with open(
        SELECTION_REPORT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4
        )

    print("\nSelection report saved:")
    print(SELECTION_REPORT)

    print("\nSTEP 7 COMPLETED")


if __name__ == "__main__":
    main()