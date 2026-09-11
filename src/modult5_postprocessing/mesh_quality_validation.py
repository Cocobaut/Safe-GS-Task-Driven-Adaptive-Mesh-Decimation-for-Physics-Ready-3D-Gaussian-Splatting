import os
import json
import numpy as np
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"src\modult5_postprocessing"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "MeshQualityValidation_ROI1"
)

ADAPTIVE_MESH_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "AdaptiveMeshes_ROI1"
)

SELECTION_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "AdaptiveMeshSelection_ROI1"
)

STEP7_REPORT = os.path.join(
    SELECTION_DIR,
    "AdaptiveMeshSelectionReport_ROI1.json"
)

REFERENCE_MESH = os.path.join(
    BASE_DIR,
    "Output",
    "TopologyRepaired_Object_ROI1.ply"
)

OUTPUT_MESH = os.path.join(
    OUTPUT_DIR,
    "Validated_Mesh_ROI1.ply"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "MeshQualityValidationReport_ROI1.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Number of sampled points
N_SAMPLES = 30000

# ROI Chamfer threshold
TAU_ROI = 0.015


# ============================================================
# CURRENT TASK ROI
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
# FUNCTIONS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if mesh.is_empty():
        raise RuntimeError(
            f"Could not load mesh: {mesh_path}"
        )

    if len(mesh.triangles) == 0:
        raise RuntimeError(
            f"Mesh contains no triangles: {mesh_path}"
        )

    pcd = mesh.sample_points_uniformly(
        number_of_points=N_SAMPLES
    )

    return pcd


def crop_roi(points, roi_min, roi_max):
    points = np.asarray(points)

    mask = np.all(
        (points >= roi_min) &
        (points <= roi_max),
        axis=1
    )

    return points[mask]


def chamfer_distance(points_a, points_b):

    if len(points_a) == 0 or len(points_b) == 0:
        return None

    pcd_a = o3d.geometry.PointCloud()
    pcd_b = o3d.geometry.PointCloud()

    pcd_a.points = o3d.utility.Vector3dVector(points_a)
    pcd_b.points = o3d.utility.Vector3dVector(points_b)

    # A -> B
    dist_a = np.asarray(
        pcd_a.compute_point_cloud_distance(pcd_b)
    )

    # B -> A
    dist_b = np.asarray(
        pcd_b.compute_point_cloud_distance(pcd_a)
    )

    chamfer = (
        np.mean(dist_a) +
        np.mean(dist_b)
    ) / 2.0

    return float(chamfer)


# ============================================================
# MAIN
# ============================================================

print("\nSTEP 8 - GEOMETRIC ERROR VALIDATION")
print("=" * 60)


# ------------------------------------------------------------
# Validate ROI
# ------------------------------------------------------------

if not np.all(np.isfinite(ROI_MIN)):
    raise RuntimeError("ROI_MIN contains invalid values.")

if not np.all(np.isfinite(ROI_MAX)):
    raise RuntimeError("ROI_MAX contains invalid values.")

if np.any(ROI_MIN > ROI_MAX):
    raise RuntimeError(
        "Invalid ROI: ROI_MIN must be <= ROI_MAX."
    )


print("\nCurrent ROI:")
print("MIN:", ROI_MIN)
print("MAX:", ROI_MAX)


# ------------------------------------------------------------
# Load Step 7 result
# ------------------------------------------------------------

step7 = load_json(STEP7_REPORT)

initial_mesh = step7["selected_mesh"]

# Step 7 stores "M3_Coarse"
# Candidate files use "M3_Coarse.ply"
if not initial_mesh.endswith(".ply"):
    initial_mesh = initial_mesh + ".ply"


print("\nInitial selected mesh:")
print(initial_mesh)


# ------------------------------------------------------------
# Candidate meshes
# ------------------------------------------------------------

candidates = [
    "M0_HighRes.ply",
    "M1_MediumHigh.ply",
    "M2_Medium.ply",
    "M3_Coarse.ply"
]


if initial_mesh not in candidates:
    raise RuntimeError(
        f"Selected mesh '{initial_mesh}' "
        f"is not one of the adaptive candidates."
    )


# Start from selected mesh.
# If it fails, move toward higher resolution.
initial_index = candidates.index(initial_mesh)

test_candidates = candidates[
    initial_index::-1
]


# ------------------------------------------------------------
# Load reference mesh
# ------------------------------------------------------------

print("\nLoading reference mesh...")

reference_pcd = sample_mesh(
    REFERENCE_MESH
)

reference_points = np.asarray(
    reference_pcd.points
)

reference_roi_points = crop_roi(
    reference_points,
    ROI_MIN,
    ROI_MAX
)

print(
    "Reference ROI points:",
    len(reference_roi_points)
)


if len(reference_roi_points) == 0:
    raise RuntimeError(
        "Reference mesh has no points inside current ROI."
    )


# ============================================================
# TEST CANDIDATE MESHES
# ============================================================

results = []

validated_mesh_name = None


for mesh_name in test_candidates:

    print("\n" + "-" * 60)
    print("Testing:", mesh_name)

    mesh_path = os.path.join(
        ADAPTIVE_MESH_DIR,
        mesh_name
    )

    if not os.path.exists(mesh_path):
        print("[SKIP] Mesh file not found.")

        results.append({
            "mesh": mesh_name,
            "roi_chamfer": None,
            "threshold": TAU_ROI,
            "passed": False,
            "status": "FILE_NOT_FOUND"
        })

        continue


    # --------------------------------------------------------
    # Sample candidate mesh
    # --------------------------------------------------------

    candidate_pcd = sample_mesh(
        mesh_path
    )

    candidate_points = np.asarray(
        candidate_pcd.points
    )


    # --------------------------------------------------------
    # Crop candidate to ROI
    # --------------------------------------------------------

    candidate_roi_points = crop_roi(
        candidate_points,
        ROI_MIN,
        ROI_MAX
    )

    print(
        "Candidate ROI points:",
        len(candidate_roi_points)
    )


    if len(candidate_roi_points) == 0:

        print(
            "[FAIL] Candidate has no points inside ROI."
        )

        results.append({
            "mesh": mesh_name,
            "roi_chamfer": None,
            "threshold": TAU_ROI,
            "passed": False,
            "status": "NO_ROI_POINTS"
        })

        continue


    # --------------------------------------------------------
    # Chamfer Distance
    # --------------------------------------------------------

    error_roi = chamfer_distance(
        candidate_roi_points,
        reference_roi_points
    )


    if error_roi is None:

        print(
            "[FAIL] Could not calculate Chamfer Distance."
        )

        results.append({
            "mesh": mesh_name,
            "roi_chamfer": None,
            "threshold": TAU_ROI,
            "passed": False,
            "status": "CHAMFER_FAILED"
        })

        continue


    print(
        f"ROI Chamfer Distance: {error_roi:.8f}"
    )

    print(
        f"Threshold: {TAU_ROI:.8f}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    passed = error_roi <= TAU_ROI


    if passed:

        print(
            "[PASS] ROI geometric error is acceptable."
        )

        status = "PASS"

    else:

        print(
            "[FAIL] ROI geometric error is too high."
        )

        status = "FAIL"


    results.append({
        "mesh": mesh_name,
        "roi_chamfer": error_roi,
        "threshold": TAU_ROI,
        "passed": passed,
        "status": status
    })


    # --------------------------------------------------------
    # Stop when a valid mesh is found
    # --------------------------------------------------------

    if passed:

        validated_mesh_name = mesh_name
        break


# ============================================================
# FINAL RESULT
# ============================================================

if validated_mesh_name is None:

    print("\n" + "=" * 60)
    print("STEP 8 FAILED")
    print("=" * 60)

    print(
        "No candidate mesh satisfies the ROI Chamfer threshold."
    )

    report = {
        "reference_mesh": REFERENCE_MESH,

        "initial_selected_mesh": initial_mesh,

        "validated_mesh": None,

        "roi": {
            "min": ROI_MIN.tolist(),
            "max": ROI_MAX.tolist()
        },

        "metric": "Chamfer Distance",

        "threshold": TAU_ROI,

        "tested_candidates": results,

        "status": "FAIL"
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    raise RuntimeError(
        "Step 8 validation failed."
    )


# ============================================================
# SAVE VALIDATED MESH
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 RESULT")
print("=" * 60)

print(
    "Validated Mesh:",
    validated_mesh_name
)


validated_path = os.path.join(
    ADAPTIVE_MESH_DIR,
    validated_mesh_name
)


validated_mesh = o3d.io.read_triangle_mesh(
    validated_path
)


if validated_mesh.is_empty():
    raise RuntimeError(
        "Validated mesh could not be loaded."
    )


if len(validated_mesh.triangles) == 0:
    raise RuntimeError(
        "Validated mesh contains no triangles."
    )


success = o3d.io.write_triangle_mesh(
    OUTPUT_MESH,
    validated_mesh
)


if not success:
    raise RuntimeError(
        "Failed to save validated mesh."
    )


print("\nValidated mesh saved:")
print(OUTPUT_MESH)


# ============================================================
# SAVE REPORT
# ============================================================

report = {

    "reference_mesh":
        REFERENCE_MESH,

    "initial_selected_mesh":
        initial_mesh,

    "validated_mesh":
        validated_mesh_name,

    "roi": {
        "min": ROI_MIN.tolist(),
        "max": ROI_MAX.tolist()
    },

    "metric":
        "Chamfer Distance",

    "threshold":
        TAU_ROI,

    "tested_candidates":
        results,

    "decision_rule":
        "Accept the first candidate whose ROI Chamfer "
        "Distance is less than or equal to tau_ROI.",

    "status":
        "PASS"
}


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


print("\nValidation report saved:")
print(OUTPUT_REPORT)

print("\nSTEP 8 COMPLETED")