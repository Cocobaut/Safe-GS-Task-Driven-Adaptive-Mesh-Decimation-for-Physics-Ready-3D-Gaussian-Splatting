import open3d as o3d
import numpy as np
import os
import json


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output"
)

ADAPTIVE_DIR = os.path.join(
    OUTPUT_DIR,
    "AdaptiveMeshes_ROI1"
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "MeshSubROILinking_ROI1"
)


# ============================================================
# ROI
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
# MESH LEVELS
#
# M0 = Raw / Highest resolution
# M1 = Medium-High
# M2 = Medium
# M3 = Coarse
# ============================================================

MESH_LEVELS = [
    "M0_HighRes",
    "M1_MediumHigh",
    "M2_Medium",
    "M3_Coarse"
]


# ============================================================
# SUB-ROI
#
# Each mesh Mi has exactly one corresponding Sub-ROI Ri.
#
# R0 -> High-detail region
# R1 -> Medium-High region
# R2 -> Medium region
# R3 -> Low-detail region
# ============================================================

ROI_CENTER = (ROI_MIN + ROI_MAX) / 2.0
ROI_SIZE = ROI_MAX - ROI_MIN


# Ratios define the nested Sub-ROI boundaries.
RATIOS = [
    0.40,   # R0
    0.60,   # R1
    0.80,   # R2
    1.00    # R3
]


def create_aabb_from_ratio(ratio):
    """
    Create a centered AABB occupying 'ratio' of the
    original ROI size along each dimension.
    """

    sub_min = (
        ROI_CENTER
        - ROI_SIZE * ratio / 2.0
    )

    sub_max = (
        ROI_CENTER
        + ROI_SIZE * ratio / 2.0
    )

    return sub_min, sub_max


# Create Sub-ROI boundaries
SUBROI_BOUNDS = {}

for i, ratio in enumerate(RATIOS):

    sub_min, sub_max = create_aabb_from_ratio(ratio)

    SUBROI_BOUNDS[f"R{i}"] = {
        "min": sub_min,
        "max": sub_max,
        "ratio": ratio
    }


# ============================================================
# MESH <-> SUB-ROI PAIRS
#
# This is the core of Step 6:
#
# (M0, R0)
# (M1, R1)
# (M2, R2)
# (M3, R3)
# ============================================================

MESH_SUBROI_PAIRS = [
    {
        "mesh": "M0_HighRes",
        "subroi": "R0"
    },
    {
        "mesh": "M1_MediumHigh",
        "subroi": "R1"
    },
    {
        "mesh": "M2_Medium",
        "subroi": "R2"
    },
    {
        "mesh": "M3_Coarse",
        "subroi": "R3"
    }
]


# ============================================================
# POINT INSIDE AABB
# ============================================================

def point_inside_aabb(point, box_min, box_max):

    return np.all(
        point >= box_min
    ) and np.all(
        point <= box_max
    )


# ============================================================
# DISTANCE FROM POINT TO SUB-ROI
# ============================================================

def point_to_aabb_distance(
    point,
    box_min,
    box_max
):

    delta = np.maximum(
        np.maximum(
            box_min - point,
            0
        ),
        point - box_max
    )

    return float(
        np.linalg.norm(delta)
    )


# ============================================================
# SPATIAL INFLUENCE
# ============================================================

SIGMA = 0.10


def compute_spatial_influence(
    distance,
    sigma
):

    return float(
        np.exp(
            -distance / sigma
        )
    )


# ============================================================
# PROCESS ONE MESH-SUBROI PAIR
# ============================================================

def process_pair(
    mesh_name,
    subroi_name
):

    print("\n" + "=" * 60)

    print(
        f"PAIR: ({mesh_name}, {subroi_name})"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # LOAD MESH
    # --------------------------------------------------------

    mesh_path = os.path.join(
        ADAPTIVE_DIR,
        f"{mesh_name}.ply"
    )

    mesh = o3d.io.read_triangle_mesh(
        mesh_path
    )

    if mesh.is_empty():

        raise RuntimeError(
            f"Cannot load {mesh_path}"
        )


    vertices = np.asarray(
        mesh.vertices
    )

    triangles = np.asarray(
        mesh.triangles
    )


    print(
        "Vertices:",
        len(vertices)
    )

    print(
        "Triangles:",
        len(triangles)
    )


    # --------------------------------------------------------
    # GET SUB-ROI
    # --------------------------------------------------------

    subroi = SUBROI_BOUNDS[
        subroi_name
    ]

    subroi_min = subroi["min"]
    subroi_max = subroi["max"]


    print(
        "\nSub-ROI:",
        subroi_name
    )

    print(
        "Min:",
        subroi_min
    )

    print(
        "Max:",
        subroi_max
    )


    # --------------------------------------------------------
    # COMPUTE TRIANGLE CENTROIDS
    # --------------------------------------------------------

    triangle_vertices = (
        vertices[triangles]
    )

    centroids = np.mean(
        triangle_vertices,
        axis=1
    )


    # --------------------------------------------------------
    # CHECK TRIANGLES INSIDE CORRESPONDING SUB-ROI
    # --------------------------------------------------------

    inside_subroi = []

    distances = []

    influences = []


    for centroid in centroids:

        inside = point_inside_aabb(
            centroid,
            subroi_min,
            subroi_max
        )

        distance = point_to_aabb_distance(
            centroid,
            subroi_min,
            subroi_max
        )

        influence = compute_spatial_influence(
            distance,
            SIGMA
        )

        inside_subroi.append(
            inside
        )

        distances.append(
            distance
        )

        influences.append(
            influence
        )


    inside_subroi = np.asarray(
        inside_subroi,
        dtype=bool
    )

    distances = np.asarray(
        distances,
        dtype=np.float64
    )

    influences = np.asarray(
        influences,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    triangle_count = len(triangles)

    subroi_triangle_count = int(
        np.sum(inside_subroi)
    )

    subroi_triangle_percentage = (
        subroi_triangle_count
        / triangle_count
        * 100
        if triangle_count > 0
        else 0.0
    )


    if subroi_triangle_count > 0:

        mean_distance = float(
            np.mean(
                distances[
                    inside_subroi
                ]
            )
        )

        mean_influence = float(
            np.mean(
                influences[
                    inside_subroi
                ]
            )
        )

    else:

        mean_distance = 0.0
        mean_influence = 0.0


    # --------------------------------------------------------
    # SAVE ARRAYS
    # --------------------------------------------------------

    np.save(
        os.path.join(
            RESULT_DIR,
            f"{mesh_name}_{subroi_name}_TriangleCentroids.npy"
        ),
        centroids
    )


    np.save(
        os.path.join(
            RESULT_DIR,
            f"{mesh_name}_{subroi_name}_InsideSubROI.npy"
        ),
        inside_subroi
    )


    np.save(
        os.path.join(
            RESULT_DIR,
            f"{mesh_name}_{subroi_name}_Distances.npy"
        ),
        distances
    )


    np.save(
        os.path.join(
            RESULT_DIR,
            f"{mesh_name}_{subroi_name}_SpatialInfluence.npy"
        ),
        influences
    )


    # --------------------------------------------------------
    # PAIR REPORT
    # --------------------------------------------------------

    report = {

        "mesh": mesh_name,

        "subroi": subroi_name,

        "pair": [
            mesh_name,
            subroi_name
        ],

        "mesh_path": mesh_path,

        "vertices": int(
            len(vertices)
        ),

        "triangles": int(
            len(triangles)
        ),

        "subroi": {

            "min": subroi_min.tolist(),

            "max": subroi_max.tolist(),

            "ratio": float(
                subroi["ratio"]
            )
        },

        "linking_statistics": {

            "triangles_inside_subroi":
                subroi_triangle_count,

            "triangle_percentage":
                float(
                    subroi_triangle_percentage
                ),

            "mean_distance":
                mean_distance,

            "mean_spatial_influence":
                mean_influence
        }
    }


    report_path = os.path.join(
        RESULT_DIR,
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


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        "\nMesh <-> Sub-ROI:"
    )

    print(
        f"({mesh_name}, {subroi_name})"
    )

    print(
        "\nTriangles inside Sub-ROI:"
    )

    print(
        f"{subroi_triangle_count} "
        f"/ {triangle_count}"
    )

    print(
        f"Percentage: "
        f"{subroi_triangle_percentage:.2f}%"
    )

    print(
        "\nMean distance:"
    )

    print(
        f"{mean_distance:.6f}"
    )

    print(
        "\nMean spatial influence:"
    )

    print(
        f"{mean_influence:.6f}"
    )

    print(
        "\nSaved report:"
    )

    print(
        report_path
    )


    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "STEP 6 - MESH <-> SUB-ROI LINKING"
    )

    print("=" * 60)


    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )


    all_reports = {}


    # --------------------------------------------------------
    # PROCESS EACH (Mi, Ri) PAIR
    # --------------------------------------------------------

    for pair in MESH_SUBROI_PAIRS:

        mesh_name = pair["mesh"]

        subroi_name = pair["subroi"]


        report = process_pair(
            mesh_name,
            subroi_name
        )


        all_reports[
            f"({mesh_name}, {subroi_name})"
        ] = report


    # --------------------------------------------------------
    # SAVE MESH-SUBROI MAPPING
    # --------------------------------------------------------

    mapping = {

        "mesh_set": [
            pair["mesh"]
            for pair in MESH_SUBROI_PAIRS
        ],

        "subroi_set": [
            pair["subroi"]
            for pair in MESH_SUBROI_PAIRS
        ],

        "pairs": MESH_SUBROI_PAIRS
    }


    mapping_path = os.path.join(
        RESULT_DIR,
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
    # SUMMARY
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULT_DIR,
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
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "STEP 6 COMPLETED"
    )

    print("=" * 60)


    print(
        "\nMesh <-> Sub-ROI pairs:"
    )


    for pair in MESH_SUBROI_PAIRS:

        print(
            f"({pair['mesh']}, "
            f"{pair['subroi']})"
        )


    print(
        "\nMapping:"
    )

    print(
        mapping_path
    )


    print(
        "\nSummary:"
    )

    print(
        summary_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()