import os
import json
import numpy as np
import open3d as o3d

from roi_config import (
    CLEANED_MESH,
    COMPONENT_MESH,
    COMPONENT_LABELS,
    COMPONENT_REPORT,
    SMALL_COMPONENT_TRIANGLES,
    SMALL_COMPONENT_AREA,
    ensure_dir,
)


def main():

    # ========================================================
    # STEP 2: CONNECTED COMPONENT FILTERING
    # Loại bỏ các component nhỏ không mong muốn khỏi ROI mesh
    # ========================================================

    # --------------------------------------------------------
    # Bước 2.1: Đọc mesh sau khi Cleaning từ Step 1
    # --------------------------------------------------------
    mesh = o3d.io.read_triangle_mesh(CLEANED_MESH)

    if mesh.is_empty():
        raise RuntimeError(f"Cannot load mesh: {CLEANED_MESH}")

    print(f"Vertices : {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")


    # --------------------------------------------------------
    # Bước 2.2: Tìm các Connected Components
    # Các triangle liên thông sẽ được gom thành một component
    # --------------------------------------------------------
    triangle_clusters, cluster_n_triangles, cluster_area = \
        mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print(f"Components: {len(cluster_n_triangles)}")


    # --------------------------------------------------------
    # Bước 2.3: Xác định component cần giữ / loại bỏ
    # Component < 2900 triangles → REMOVE
    # Component >= 2900 triangles → KEEP
    # --------------------------------------------------------
    keep_mask = (
        cluster_n_triangles >= SMALL_COMPONENT_TRIANGLES
    )

    for i in range(len(cluster_n_triangles)):
        status = "KEEP" if keep_mask[i] else "REMOVE"

        print(
            f"Component {i}: "
            f"{cluster_n_triangles[i]} triangles, "
            f"area={cluster_area[i]:.6f} -> {status}"
        )


    # --------------------------------------------------------
    # Bước 2.4: Loại bỏ các triangle thuộc component nhỏ
    # --------------------------------------------------------
    triangle_keep = keep_mask[triangle_clusters]

    filtered_mesh = o3d.geometry.TriangleMesh(mesh)

    filtered_mesh.remove_triangles_by_mask(
        ~triangle_keep
    )

    # Xóa vertex không còn được triangle nào sử dụng
    filtered_mesh.remove_unreferenced_vertices()

    # Tính lại normal cho mesh sau khi lọc
    filtered_mesh.compute_vertex_normals()


    # --------------------------------------------------------
    # Bước 2.5: Lưu mesh sau khi lọc
    # --------------------------------------------------------
    ensure_dir(os.path.dirname(COMPONENT_MESH))

    o3d.io.write_triangle_mesh(
        COMPONENT_MESH,
        filtered_mesh
    )


    # --------------------------------------------------------
    # Bước 2.6: Lưu label của các triangle còn lại
    # --------------------------------------------------------
    np.save(
        COMPONENT_LABELS,
        triangle_clusters[triangle_keep]
    )


    # --------------------------------------------------------
    # Bước 2.7: Lưu thống kê kết quả
    # --------------------------------------------------------
    report = {
        "input_mesh": CLEANED_MESH,
        "output_mesh": COMPONENT_MESH,

        # Ngưỡng dùng để quyết định REMOVE / KEEP
        "triangle_threshold": SMALL_COMPONENT_TRIANGLES,

        # Chỉ ghi vào report, không dùng để filter
        "area_threshold_report_only": SMALL_COMPONENT_AREA,

        "num_components": int(len(cluster_n_triangles)),
        "triangles_before": int(len(mesh.triangles)),
        "triangles_after": int(len(filtered_mesh.triangles)),
        "triangles_removed": int(
            len(mesh.triangles)
            - len(filtered_mesh.triangles)
        ),
    }

    with open(
        COMPONENT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)


    # --------------------------------------------------------
    # Kết thúc Step 2
    # --------------------------------------------------------
    print("\n=== STEP 2 COMPLETED ===")
    print(f"Triangles after : {len(filtered_mesh.triangles)}")
    print(
        f"Triangles removed: "
        f"{report['triangles_removed']}"
    )


if __name__ == "__main__":
    main()