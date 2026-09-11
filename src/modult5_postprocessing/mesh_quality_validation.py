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
    SELECTION_REPORT,
    SELECTION_DIR,
    SELECTED_MESH,
    VALIDATION_DIR,
    VALIDATED_MESH,
    VALIDATION_REPORT,
    CHAMFER_SAMPLES,
    TAU_ROI,
    ensure_dir,
)


# ============================================================
# STEP 8: MESH QUALITY VALIDATION
# Kiểm tra mesh từ COARSE → FINE
# ============================================================

def check_mesh_structure(mesh):

    # Mesh rỗng
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return False, "EMPTY_MESH"

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Vertex không hợp lệ
    if not np.all(np.isfinite(vertices)):
        return False, "NON_FINITE_VERTICES"

    # Triangle index không hợp lệ
    if np.any(triangles < 0):
        return False, "INVALID_TRIANGLE_INDEX"

    if np.any(triangles >= len(vertices)):
        return False, "INVALID_TRIANGLE_INDEX"

    # Kiểm tra triangle suy biến
    tri_vertices = vertices[triangles]

    cross = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0]
    )

    areas = 0.5 * np.linalg.norm(
        cross,
        axis=1
    )

    degenerate_ratio = (
        np.sum(areas <= 1e-12)
        / len(triangles)
    )

    if degenerate_ratio > 0.05:
        return False, "TOO_MANY_DEGENERATE_TRIANGLES"

    return True, "VALID"


def crop_to_roi(mesh):

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return o3d.geometry.TriangleMesh()

    # Tính centroid của từng triangle
    centroids = np.mean(
        vertices[triangles],
        axis=1
    )

    # Chỉ lấy triangle nằm trong Task ROI
    mask = np.all(
        (centroids >= CURRENT_TASK_ROI_MIN)
        &
        (centroids <= CURRENT_TASK_ROI_MAX),
        axis=1
    )

    roi_triangles = triangles[mask]

    if len(roi_triangles) == 0:
        return o3d.geometry.TriangleMesh()

    roi_mesh = o3d.geometry.TriangleMesh()

    roi_mesh.vertices = o3d.utility.Vector3dVector(
        vertices
    )

    roi_mesh.triangles = o3d.utility.Vector3iVector(
        roi_triangles
    )

    roi_mesh.remove_unreferenced_vertices()

    return roi_mesh


def calculate_chamfer(mesh1, mesh2):

    pcd1 = mesh1.sample_points_uniformly(
        number_of_points=CHAMFER_SAMPLES
    )

    pcd2 = mesh2.sample_points_uniformly(
        number_of_points=CHAMFER_SAMPLES
    )

    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        return None

    d1 = np.asarray(
        pcd1.compute_point_cloud_distance(pcd2)
    )

    d2 = np.asarray(
        pcd2.compute_point_cloud_distance(pcd1)
    )

    return float(
        (np.mean(d1) + np.mean(d2)) / 2.0
    )


def main():

    print("=" * 65)
    print("STEP 8 - MESH QUALITY VALIDATION")
    print("=" * 65)

    ensure_dir(VALIDATION_DIR)

    # --------------------------------------------------------
    # 1. Đọc candidate order từ Step 7
    # --------------------------------------------------------
    with open(
        SELECTION_REPORT,
        "r",
        encoding="utf-8"
    ) as f:
        step7_report = json.load(f)

    candidates = step7_report["candidate_order"]

    print("\nTesting order:")
    print(" → ".join(candidates))

    # --------------------------------------------------------
    # 2. M0 là reference mesh chất lượng cao
    # --------------------------------------------------------
    reference_path = os.path.join(
        ADAPTIVE_DIR,
        "M0_HighRes.ply"
    )

    reference_mesh = o3d.io.read_triangle_mesh(
        reference_path
    )

    valid, status = check_mesh_structure(
        reference_mesh
    )

    if not valid:
        raise RuntimeError(
            f"M0 reference mesh is invalid: {status}"
        )

    reference_roi = crop_to_roi(
        reference_mesh
    )

    if reference_roi.is_empty():
        raise RuntimeError(
            "M0 has no geometry inside Task ROI."
        )

    # --------------------------------------------------------
    # 3. Kiểm tra lần lượt M3 → M2 → M1 → M0
    # --------------------------------------------------------
    results = []

    final_mesh = None
    final_result = None

    for mesh_name in candidates:

        print("\n" + "-" * 55)
        print("Testing:", mesh_name)
        print("-" * 55)

        mesh_path = os.path.join(
            ADAPTIVE_DIR,
            mesh_name
        )

        mesh = o3d.io.read_triangle_mesh(
            mesh_path
        )

        # ----------------------------------------------------
        # 3.1 Kiểm tra cấu trúc mesh
        # ----------------------------------------------------
        valid, status = check_mesh_structure(mesh)

        print("Structure:", status)

        if not valid:

            results.append({
                "mesh": mesh_name,
                "status": status,
                "passed": False,
            })

            print("→ REJECT")

            continue

        # ----------------------------------------------------
        # 3.2 Lấy phần mesh nằm trong Task ROI
        # ----------------------------------------------------
        candidate_roi = crop_to_roi(mesh)

        if candidate_roi.is_empty():

            results.append({
                "mesh": mesh_name,
                "status": "NO_ROI_GEOMETRY",
                "passed": False,
            })

            print("→ REJECT: No ROI geometry")

            continue

        # ----------------------------------------------------
        # 3.3 So sánh ROI với M0 bằng Chamfer Distance
        # ----------------------------------------------------
        chamfer = calculate_chamfer(
            candidate_roi,
            reference_roi
        )

        passed = (
            chamfer is not None
            and chamfer <= TAU_ROI
        )

        print(
            f"ROI Chamfer: "
            f"{chamfer:.6f}"
        )

        print(
            f"Threshold: "
            f"{TAU_ROI:.6f}"
        )

        print(
            "→",
            "PASS" if passed else "FAIL"
        )

        result = {
            "mesh": mesh_name,
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.triangles)),
            "structurally_valid": True,
            "validation_status": status,
            "chamfer": (
                None
                if chamfer is None
                else float(chamfer)
            ),
            "threshold": float(TAU_ROI),
            "passed": bool(passed),
        }

        results.append(result)

        # ----------------------------------------------------
        # 3.4 Mesh đầu tiên PASS sẽ được chọn
        # ----------------------------------------------------
        if passed:

            final_mesh = mesh
            final_result = result

            print(
                f"\n✓ SELECTED: {mesh_name}"
            )

            break

    # --------------------------------------------------------
    # 4. Không có mesh nào đạt
    # --------------------------------------------------------
    if final_mesh is None:

        raise RuntimeError(
            "No candidate mesh passed validation."
        )

    # --------------------------------------------------------
    # 5. Lưu mesh cuối cùng
    # --------------------------------------------------------
    final_mesh.remove_unreferenced_vertices()
    final_mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        SELECTED_MESH,
        final_mesh
    )

    o3d.io.write_triangle_mesh(
        VALIDATED_MESH,
        final_mesh
    )

    # --------------------------------------------------------
    # 6. Lưu report
    # --------------------------------------------------------
    report = {
        "candidate_order": candidates,

        "selected_mesh":
            final_result["mesh"],

        "validation_results":
            results,

        "reference_mesh":
            "M0_HighRes.ply",

        "chamfer_threshold":
            float(TAU_ROI),

        "output":
            VALIDATED_MESH,
    }

    with open(
        VALIDATION_REPORT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4
        )

    print("\n" + "=" * 65)
    print("STEP 8 COMPLETED")
    print("=" * 65)

    print(
        "Final mesh:",
        final_result["mesh"]
    )

    print(
        "Saved:",
        VALIDATED_MESH
    )


if __name__ == "__main__":
    main()