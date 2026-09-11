import json
import os
import sys

import numpy as np
import open3d as o3d


# ============================================================
# IMPORT CONFIG
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from roi_config import (
    ADAPTIVE_DIR,
    LINKING_DIR,
    MESH_SUBROI_PAIRS,
    create_subrois,
    ensure_dir,
    triangle_centroids,
    point_inside_aabb,
)


# ============================================================
# PROCESS ONE MESH - SUBROI PAIR
# ============================================================

def process_pair(mesh_name, subroi_name, subrois):

    print("\n" + "=" * 65)
    print(f"PAIR: ({mesh_name}, {subroi_name})")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load adaptive mesh
    # --------------------------------------------------------

    mesh_path = os.path.join(
        ADAPTIVE_DIR,
        f"{mesh_name}.ply"
    )

    print("Mesh:", mesh_path)

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"Mesh not found:\n{mesh_path}"
        )

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if mesh.is_empty():
        raise RuntimeError(
            f"Cannot load mesh:\n{mesh_path}"
        )

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print("Vertices :", len(vertices))
    print("Triangles:", len(triangles))

    # --------------------------------------------------------
    # 2. Get corresponding Sub-ROI
    # --------------------------------------------------------

    if subroi_name not in subrois:
        raise KeyError(
            f"Sub-ROI '{subroi_name}' not found."
        )

    box = subrois[subroi_name]

    print("\nSub-ROI:", subroi_name)
    print("Ratio:", box["ratio"])
    print("Detail level:", box["detail_level"])
    print("Min:", box["min"])
    print("Max:", box["max"])

    # --------------------------------------------------------
    # 3. Calculate triangle centroids
    # --------------------------------------------------------

    centroids = triangle_centroids(mesh)

    # --------------------------------------------------------
    # 4. Determine triangles inside Sub-ROI
    # --------------------------------------------------------

    inside = np.array(
        [
            point_inside_aabb(
                centroid,
                box["min"],
                box["max"]
            )
            for centroid in centroids
        ],
        dtype=bool
    )

    triangles_inside = int(
        np.sum(inside)
    )

    total_triangles = len(triangles)

    percentage = (
        triangles_inside
        / total_triangles
        * 100.0
        if total_triangles > 0
        else 0.0
    )

    print(
        "\nTriangles inside Sub-ROI:",
        f"{triangles_inside} / {total_triangles}"
    )

    print(
        f"Percentage: {percentage:.2f}%"
    )

    # --------------------------------------------------------
    # 5. Save triangle-to-SubROI labels
    # --------------------------------------------------------

    labels_path = os.path.join(
        LINKING_DIR,
        f"{mesh_name}_{subroi_name}_InsideSubROI.npy"
    )

    np.save(
        labels_path,
        inside
    )

    print(
        "Triangle labels:",
        labels_path
    )

    # --------------------------------------------------------
    # 6. Create report
    # --------------------------------------------------------

    report = {
        "mesh": mesh_name,

        "subroi": subroi_name,

        "pair": [
            mesh_name,
            subroi_name
        ],

        "mesh_path": mesh_path,

        "mesh_statistics": {
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles))
        },

        "subroi_definition": {
            "ratio": float(box["ratio"]),
            "detail_level": box["detail_level"],
            "min": box["min"].tolist(),
            "max": box["max"].tolist()
        },

        "linking_statistics": {
            "triangles_inside_subroi": triangles_inside,
            "total_triangles": int(total_triangles),
            "triangle_percentage": float(percentage)
        },

        "label_file": labels_path
    }

    report_path = os.path.join(
        LINKING_DIR,
        f"{mesh_name}_{subroi_name}_Report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        "Report:",
        report_path
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STEP 6 - MESH <-> SUB-ROI LINKING")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Prepare output directory
    # --------------------------------------------------------

    ensure_dir(LINKING_DIR)

    # --------------------------------------------------------
    # 2. Create Sub-ROIs from roi_config.py
    # --------------------------------------------------------

    print("\n[1] Creating Sub-ROIs...")

    subrois = create_subrois()

    for name, box in subrois.items():

        print(
            f"{name}: "
            f"ratio={box['ratio']}, "
            f"detail={box['detail_level']}"
        )

    # --------------------------------------------------------
    # 3. Process predefined mesh-SubROI pairs
    # --------------------------------------------------------

    print("\n[2] Processing mesh-SubROI pairs...")

    all_reports = {}

    for pair in MESH_SUBROI_PAIRS:

        mesh_name = pair["mesh"]
        subroi_name = pair["subroi"]

        report = process_pair(
            mesh_name,
            subroi_name,
            subrois
        )

        key = (
            f"({mesh_name}, {subroi_name})"
        )

        all_reports[key] = report

    # --------------------------------------------------------
    # 4. Save global mapping
    # --------------------------------------------------------

    print("\n[3] Saving mesh-SubROI mapping...")

    mapping = {
        "mesh_set": [
            pair["mesh"]
            for pair in MESH_SUBROI_PAIRS
        ],

        "subroi_set": [
            pair["subroi"]
            for pair in MESH_SUBROI_PAIRS
        ],

        "pairs": MESH_SUBROI_PAIRS,

        "mapping": {
            pair["subroi"]: pair["mesh"]
            for pair in MESH_SUBROI_PAIRS
        },

        "purpose": (
            "Link each pre-generated adaptive mesh "
            "to its corresponding Sub-ROI. "
            "Mesh selection is performed in Step 7."
        )
    }

    mapping_path = os.path.join(
        LINKING_DIR,
        "MeshSubROIMapping_ROI1.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mapping,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 5. Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        LINKING_DIR,
        "MeshSubROILinkingSummary_ROI1.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_reports,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 6. Final output
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 6 COMPLETED")
    print("=" * 65)

    print("\nMapping:")
    print(mapping_path)

    print("\nSummary:")
    print(summary_path)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()