import os
import json
import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_MESH = os.path.join(
    BASE_DIR,
    "Output",
    "TopologyRepaired_Object_ROI1.ply"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "Output")


# ============================================================
# CURRENT ROI
# ============================================================

ROI_MIN = np.array([
    1.31638556,
    -1.97638444,
    -1.12399122
])

ROI_MAX = np.array([
    2.24038578,
    -0.92432044,
    -0.49570042
])


# ============================================================
# SUB-ROI CONFIGURATION
# ============================================================
# Nested Sub-ROIs are created around the same ROI center.
#
# R0 -> 40% -> highest detail
# R1 -> 60% -> medium-high detail
# R2 -> 80% -> medium detail
# R3 -> 100% -> coarse detail
#
# Classification priority:
# R0 > R1 > R2 > R3 > NON_ROI
# ============================================================

SUBROIS = {
    "R0": {
        "ratio": 0.40,
        "detail_level": "HIGH"
    },
    "R1": {
        "ratio": 0.60,
        "detail_level": "MEDIUM_HIGH"
    },
    "R2": {
        "ratio": 0.80,
        "detail_level": "MEDIUM"
    },
    "R3": {
        "ratio": 1.00,
        "detail_level": "COARSE"
    }
}


# ============================================================
# SPATIAL INFLUENCE
# ============================================================
# I = exp(-d / sigma)
#
# d     : distance from triangle centroid to ROI
# sigma : influence range
# ============================================================

SIGMA = 0.10


# ============================================================
# FUNCTIONS
# ============================================================

def create_subrois(roi_min, roi_max):
    """Create nested Sub-ROI AABBs."""

    center = (roi_min + roi_max) / 2.0
    half_size = (roi_max - roi_min) / 2.0

    subrois = {}

    for name, config in SUBROIS.items():
        ratio = config["ratio"]

        sub_min = center - half_size * ratio
        sub_max = center + half_size * ratio

        subrois[name] = {
            "ratio": ratio,
            "detail_level": config["detail_level"],
            "min": sub_min,
            "max": sub_max
        }

    return subrois


def point_inside_aabb(point, box_min, box_max):
    """Check whether a point is inside an AABB."""

    return np.all(
        (point >= box_min) &
        (point <= box_max)
    )


def classify_centroid(centroid, subrois):
    """
    Assign triangle to the first Sub-ROI containing its centroid.

    Priority:
    R0 > R1 > R2 > R3 > NON_ROI
    """

    for name in SUBROIS.keys():
        box = subrois[name]

        if point_inside_aabb(
            centroid,
            box["min"],
            box["max"]
        ):
            return name

    return "NON_ROI"


def distance_to_aabb(point, box_min, box_max):
    """Calculate Euclidean distance from point to AABB."""

    delta = np.maximum(
        np.maximum(box_min - point, 0),
        point - box_max
    )

    return np.linalg.norm(delta)


def compute_spatial_influence(distances, sigma):
    """Calculate exponential spatial influence."""

    return np.exp(-distances / sigma)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 4 - ROI / SUB-ROI MAPPING + SPATIAL INFLUENCE")
    print("=" * 70)

    # ========================================================
    # 1. Load mesh
    # ========================================================

    print("\n[1] Loading mesh...")
    print("Input:", INPUT_MESH)

    mesh = o3d.io.read_triangle_mesh(INPUT_MESH)

    if mesh.is_empty():
        raise RuntimeError("Mesh could not be loaded.")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"Vertices : {len(vertices):,}")
    print(f"Triangles: {len(triangles):,}")


    # ========================================================
    # 2. ROI information
    # ========================================================

    roi_center = (ROI_MIN + ROI_MAX) / 2.0
    roi_size = ROI_MAX - ROI_MIN

    print("\n[2] ROI information")
    print("ROI min   :", ROI_MIN)
    print("ROI max   :", ROI_MAX)
    print("ROI center:", roi_center)
    print("ROI size  :", roi_size)


    # ========================================================
    # 3. Create nested Sub-ROIs
    # ========================================================

    print("\n[3] Creating Sub-ROIs...")

    subrois = create_subrois(
        ROI_MIN,
        ROI_MAX
    )

    for name, box in subrois.items():
        print(
            f"{name}: "
            f"ratio={box['ratio']:.2f}, "
            f"detail={box['detail_level']}"
        )
        print("  min:", box["min"])
        print("  max:", box["max"])


    # ========================================================
    # 4. Calculate triangle centroids
    # ========================================================

    print("\n[4] Calculating triangle centroids...")

    triangle_vertices = vertices[triangles]

    centroids = np.mean(
        triangle_vertices,
        axis=1
    )

    print(
        f"Centroids calculated: {len(centroids):,}"
    )


    # ========================================================
    # 5. Classify triangles into Sub-ROIs
    # ========================================================

    print("\n[5] Classifying triangles...")

    triangle_labels = np.array(
        [
            classify_centroid(
                centroid,
                subrois
            )
            for centroid in centroids
        ],
        dtype="<U10"
    )


    # ========================================================
    # 6. Calculate distance to ROI
    # ========================================================

    print("\n[6] Calculating distance to ROI...")

    distances = np.array([
        distance_to_aabb(
            centroid,
            ROI_MIN,
            ROI_MAX
        )
        for centroid in centroids
    ])

    print(
        f"Min : {distances.min():.6f}"
    )
    print(
        f"Max : {distances.max():.6f}"
    )
    print(
        f"Mean: {distances.mean():.6f}"
    )


    # ========================================================
    # 7. Calculate spatial influence
    # ========================================================

    print("\n[7] Calculating spatial influence...")

    spatial_influence = compute_spatial_influence(
        distances,
        SIGMA
    )

    print(
        f"Min : {spatial_influence.min():.6f}"
    )
    print(
        f"Max : {spatial_influence.max():.6f}"
    )
    print(
        f"Mean: {spatial_influence.mean():.6f}"
    )


    # ========================================================
    # 8. Triangle statistics
    # ========================================================

    print("\n[8] Triangle classification result")
    print("-" * 55)

    total = len(triangle_labels)
    statistics = {}

    for name in ["R0", "R1", "R2", "R3", "NON_ROI"]:

        count = int(
            np.sum(triangle_labels == name)
        )

        percentage = (
            count / total * 100.0
        )

        statistics[name] = {
            "triangles": count,
            "percentage": percentage
        }

        print(
            f"{name:8s}: "
            f"{count:8,d} triangles "
            f"({percentage:6.2f}%)"
        )


    # ========================================================
    # 9. Validation
    # ========================================================

    print("\n[9] Validation...")

    if not (
        len(triangles)
        == len(centroids)
        == len(triangle_labels)
        == len(distances)
        == len(spatial_influence)
    ):
        raise RuntimeError(
            "Triangle-level data size mismatch."
        )

    if not np.all(np.isfinite(centroids)):
        raise RuntimeError(
            "Invalid centroid values detected."
        )

    if not np.all(np.isfinite(distances)):
        raise RuntimeError(
            "Invalid distance values detected."
        )

    if not np.all(np.isfinite(spatial_influence)):
        raise RuntimeError(
            "Invalid spatial influence values detected."
        )

    print("PASS: Triangle-level data is aligned.")
    print("PASS: R0, R1, R2 and R3 are defined.")


    # ========================================================
    # 10. Save NumPy outputs
    # ========================================================

    print("\n[10] Saving NumPy outputs...")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_files = {
        "triangle_labels":
            "TriangleSubROILabels_ROI1.npy",

        "centroids":
            "TriangleCentroids_ROI1.npy",

        "distances":
            "TriangleROIDistances_ROI1.npy",

        "spatial_influence":
            "TriangleSpatialInfluence_ROI1.npy"
    }

    label_path = os.path.join(
        OUTPUT_DIR,
        output_files["triangle_labels"]
    )

    centroid_path = os.path.join(
        OUTPUT_DIR,
        output_files["centroids"]
    )

    distance_path = os.path.join(
        OUTPUT_DIR,
        output_files["distances"]
    )

    influence_path = os.path.join(
        OUTPUT_DIR,
        output_files["spatial_influence"]
    )

    np.save(label_path, triangle_labels)
    np.save(centroid_path, centroids)
    np.save(distance_path, distances)
    np.save(influence_path, spatial_influence)

    print("Labels            :", label_path)
    print("Centroids         :", centroid_path)
    print("Distances         :", distance_path)
    print("Spatial influence :", influence_path)


    # ========================================================
    # 11. Save JSON report
    # ========================================================

    print("\n[11] Saving JSON report...")

    subroi_report = {}

    for name, box in subrois.items():
        subroi_report[name] = {
            "ratio": box["ratio"],
            "detail_level": box["detail_level"],
            "min": box["min"].tolist(),
            "max": box["max"].tolist()
        }

    report = {
        "input_mesh": INPUT_MESH,

        "mesh_statistics": {
            "vertices": len(vertices),
            "triangles": len(triangles)
        },

        "roi": {
            "min": ROI_MIN.tolist(),
            "max": ROI_MAX.tolist(),
            "center": roi_center.tolist(),
            "size": roi_size.tolist()
        },

        "subroi_definition": subroi_report,

        "classification_rule":
            "Triangle centroid priority: "
            "R0 > R1 > R2 > R3 > NON_ROI.",

        "spatial_influence": {
            "formula": "I = exp(-d / sigma)",
            "sigma": SIGMA,
            "distance_statistics": {
                "min": float(distances.min()),
                "max": float(distances.max()),
                "mean": float(distances.mean())
            },
            "influence_statistics": {
                "min": float(spatial_influence.min()),
                "max": float(spatial_influence.max()),
                "mean": float(spatial_influence.mean())
            }
        },

        "statistics": statistics,

        "outputs": {
            "triangle_labels": label_path,
            "triangle_centroids": centroid_path,
            "triangle_roi_distances": distance_path,
            "triangle_spatial_influence": influence_path
        }
    }

    report_path = os.path.join(
        OUTPUT_DIR,
        "ROISubROIMappingReport_ROI1.json"
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

    print("Report:", report_path)


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETED")
    print("=" * 70)

    print(
        f"Total triangles: {len(triangles):,}"
    )

    for name in ["R0", "R1", "R2", "R3", "NON_ROI"]:
        print(
            f"{name:8s}: "
            f"{statistics[name]['triangles']:8,d} "
            f"({statistics[name]['percentage']:6.2f}%)"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()