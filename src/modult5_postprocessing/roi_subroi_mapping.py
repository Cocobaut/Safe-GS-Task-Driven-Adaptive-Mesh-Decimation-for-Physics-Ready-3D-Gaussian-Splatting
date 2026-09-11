import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roi_config import (
    OUTPUT_DIR,
    REPAIRED_MESH,
    ROI_MAPPING_REPORT,
    ROI_MAX,
    ROI_MIN,
    SPATIAL_INFLUENCE_SIGMA,
    SUBROIS,
    create_subrois,
    distance_to_aabb,
    ensure_dir,
    point_inside_aabb,
    roi_center_size,
    triangle_centroids,
)


def classify_centroid(centroid, subrois):
    for name in SUBROIS:
        box = subrois[name]
        if point_inside_aabb(centroid, box["min"], box["max"]):
            return name
    return "NON_ROI"


def main():
    print("=" * 70)
    print("STEP 4 - ROI / SUB-ROI MAPPING + SPATIAL INFLUENCE")
    print("=" * 70)
    print("\n[1] Loading mesh...")
    print("Input:", REPAIRED_MESH)

    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)
    if mesh.is_empty():
        raise RuntimeError("Mesh could not be loaded.")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    print(f"Vertices : {len(vertices):,}")
    print(f"Triangles: {len(triangles):,}")

    roi_center, roi_size = roi_center_size()
    print("\n[2] ROI information")
    print("ROI min   :", ROI_MIN)
    print("ROI max   :", ROI_MAX)
    print("ROI center:", roi_center)
    print("ROI size  :", roi_size)
    print("(AABB is the current stand-in until Module 2 is available.)")

    subrois = create_subrois()
    print("\n[3] Creating Sub-ROIs...")
    for name, box in subrois.items():
        print(
            f"{name}: ratio={box['ratio']:.2f}, detail={box['detail_level']}"
        )

    centroids = triangle_centroids(mesh)
    triangle_labels = np.array(
        [classify_centroid(c, subrois) for c in centroids],
        dtype="<U10",
    )
    distances = np.array(
        [distance_to_aabb(c, ROI_MIN, ROI_MAX) for c in centroids],
        dtype=np.float64,
    )
    spatial_influence = np.exp(-distances / SPATIAL_INFLUENCE_SIGMA)

    print("\n[8] Triangle classification result")
    total = len(triangle_labels)
    statistics = {}
    for name in ["R0", "R1", "R2", "R3", "NON_ROI"]:
        count = int(np.sum(triangle_labels == name))
        percentage = count / total * 100.0 if total else 0.0
        statistics[name] = {"triangles": count, "percentage": percentage}
        print(f"{name:8s}: {count:8,d} triangles ({percentage:6.2f}%)")

    ensure_dir(OUTPUT_DIR)
    label_path = os.path.join(OUTPUT_DIR, "TriangleSubROILabels_ROI1.npy")
    centroid_path = os.path.join(OUTPUT_DIR, "TriangleCentroids_ROI1.npy")
    distance_path = os.path.join(OUTPUT_DIR, "TriangleROIDistances_ROI1.npy")
    influence_path = os.path.join(OUTPUT_DIR, "TriangleSpatialInfluence_ROI1.npy")
    np.save(label_path, triangle_labels)
    np.save(centroid_path, centroids)
    np.save(distance_path, distances)
    np.save(influence_path, spatial_influence)

    subroi_report = {
        name: {
            "ratio": box["ratio"],
            "detail_level": box["detail_level"],
            "min": box["min"].tolist(),
            "max": box["max"].tolist(),
        }
        for name, box in subrois.items()
    }

    report = {
        "input_mesh": REPAIRED_MESH,
        "roi_source": "manual_aabb_standin_for_module2",
        "mesh_statistics": {
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles)),
        },
        "roi": {
            "min": ROI_MIN.tolist(),
            "max": ROI_MAX.tolist(),
            "center": roi_center.tolist(),
            "size": roi_size.tolist(),
        },
        "subroi_definition": subroi_report,
        "classification_rule":
            "Triangle centroid priority: R0 > R1 > R2 > R3 > NON_ROI.",
        "spatial_influence": {
            "formula": "I = exp(-d / sigma)",
            "sigma": SPATIAL_INFLUENCE_SIGMA,
        },
        "statistics": statistics,
        "outputs": {
            "triangle_labels": label_path,
            "triangle_centroids": centroid_path,
            "triangle_roi_distances": distance_path,
            "triangle_spatial_influence": influence_path,
        },
    }

    with open(ROI_MAPPING_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Report:", ROI_MAPPING_REPORT)
    print("\nSTEP 4 COMPLETED")


if __name__ == "__main__":
    main()
