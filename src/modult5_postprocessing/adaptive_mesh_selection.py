import os
import json
import numpy as np
import open3d as o3d


# ============================================================
# STEP 7 - ADAPTIVE MESH SELECTION
# ============================================================
#
# Purpose:
# Select the most suitable pre-generated mesh according to
# the overlap between the CURRENT TASK ROI and the predefined
# Sub-ROIs (R0, R1, R2, R3).
#
# R0 -> Highest detail      -> M0_HighRes
# R1 -> Medium-high detail -> M1_MediumHigh
# R2 -> Medium detail      -> M2_Medium
# R3 -> Coarse detail      -> M3_Coarse
#
# The QEM simplification is NOT executed again in this step.
# Step 7 only evaluates the overlap and selects an existing
# candidate mesh.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"src\modult5_postprocessing"

# Step 4 output:
# Contains the definitions of R0, R1, R2 and R3.
STEP4_REPORT = os.path.join(
    BASE_DIR,
    "Output",
    "ROISubROIMappingReport_ROI1.json"
)

# Step 5 output:
# Contains the pre-generated candidate meshes.
ADAPTIVE_MESH_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "AdaptiveMeshes_ROI1"
)

# Step 7 output directory.
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "AdaptiveMeshSelection_ROI1"
)

OUTPUT_MESH = os.path.join(
    OUTPUT_DIR,
    "Selected_Mesh_ROI1.ply"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "AdaptiveMeshSelectionReport_ROI1.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. CURRENT TASK ROI
# ============================================================
#
# This is the ROI required by the CURRENT task.
#
# If the task changes, only CURRENT_ROI_MIN and
# CURRENT_ROI_MAX need to be changed.
#
# The candidate meshes generated in Step 5 remain unchanged.
# ============================================================

CURRENT_ROI_MIN = np.array([
    1.31638556,
    -1.97638444,
    -1.12399122
])

CURRENT_ROI_MAX = np.array([
    2.24038578,
    -0.92432044,
    -0.49570042
])


# ============================================================
# 3. MESH <-> SUB-ROI MAPPING
# ============================================================
#
# This mapping was established in Step 6.
#
# R0 -> M0_HighRes
# R1 -> M1_MediumHigh
# R2 -> M2_Medium
# R3 -> M3_Coarse
# ============================================================

MESH_MAPPING = {
    "R0": "M0_HighRes.ply",
    "R1": "M1_MediumHigh.ply",
    "R2": "M2_Medium.ply",
    "R3": "M3_Coarse.ply"
}

SUBROI_ORDER = [
    "R0",
    "R1",
    "R2",
    "R3"
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def box_volume(roi_min, roi_max):
    """
    Calculate the volume of an axis-aligned bounding box.
    """

    size = np.maximum(
        roi_max - roi_min,
        0
    )

    return float(np.prod(size))


def intersection_volume(
    min_a,
    max_a,
    min_b,
    max_b
):
    """
    Calculate the intersection volume between two AABBs.
    """

    overlap_min = np.maximum(
        min_a,
        min_b
    )

    overlap_max = np.minimum(
        max_a,
        max_b
    )

    size = np.maximum(
        overlap_max - overlap_min,
        0
    )

    return float(np.prod(size))


def calculate_iou(
    current_min,
    current_max,
    subroi_min,
    subroi_max
):
    """
    Calculate 3D Intersection over Union (IoU).

    IoU = Intersection Volume / Union Volume
    """

    intersection = intersection_volume(
        current_min,
        current_max,
        subroi_min,
        subroi_max
    )

    current_volume = box_volume(
        current_min,
        current_max
    )

    subroi_volume = box_volume(
        subroi_min,
        subroi_max
    )

    union = (
        current_volume
        + subroi_volume
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# 5. START STEP 7
# ============================================================

print("\n" + "=" * 75)
print("STEP 7 - ADAPTIVE MESH SELECTION")
print("=" * 75)


# ============================================================
# 6. DISPLAY CURRENT TASK ROI
# ============================================================

print("\n[1] CURRENT TASK ROI")
print("-" * 75)

print("ROI Min:")
print(CURRENT_ROI_MIN)

print("ROI Max:")
print(CURRENT_ROI_MAX)


current_roi_volume = box_volume(
    CURRENT_ROI_MIN,
    CURRENT_ROI_MAX
)

print(
    f"\nCurrent ROI Volume : "
    f"{current_roi_volume:.6f}"
)


# ============================================================
# 7. LOAD SUB-ROI INFORMATION FROM STEP 4
# ============================================================

print("\n[2] LOAD SUB-ROI INFORMATION")
print("-" * 75)

if not os.path.exists(STEP4_REPORT):
    raise FileNotFoundError(
        f"Step 4 report not found:\n{STEP4_REPORT}"
    )


with open(
    STEP4_REPORT,
    "r",
    encoding="utf-8"
) as f:
    step4_data = json.load(f)


if "subroi_definition" not in step4_data:
    raise KeyError(
        "Cannot find 'subroi_definition' in Step 4 report."
    )


subrois = step4_data["subroi_definition"]


print("Loaded Sub-ROIs:")

for name in SUBROI_ORDER:

    if name not in subrois:
        raise KeyError(
            f"Cannot find {name} in Step 4 report."
        )

    print(
        f"  {name} -> "
        f"{MESH_MAPPING[name]} | "
        f"Ratio = {subrois[name]['ratio']:.2f} | "
        f"Detail = {subrois[name]['detail_level']}"
    )


# ============================================================
# 8. CALCULATE OVERLAP AND IoU
# ============================================================
#
# For every Sub-ROI we calculate:
#
# 1. Ratio
#    Size ratio used to construct the Sub-ROI.
#
# 2. Sub-ROI Volume
#    Volume of the Sub-ROI.
#
# 3. Overlap Volume
#    Intersection volume between the current Task ROI
#    and the Sub-ROI.
#
# 4. Overlap Ratio
#    How much of the CURRENT TASK ROI is covered by
#    the Sub-ROI.
#
#    Overlap Ratio =
#        Overlap Volume / Current ROI Volume
#
# 5. IoU
#    Measures the geometric similarity between the
#    current Task ROI and the Sub-ROI.
#
#    IoU =
#        Overlap Volume / Union Volume
# ============================================================

print("\n[3] CALCULATE OVERLAP AND IoU")
print("-" * 75)

results = []


for name in SUBROI_ORDER:

    # --------------------------------------------------------
    # Get Sub-ROI information
    # --------------------------------------------------------

    subroi_min = np.array(
        subrois[name]["min"],
        dtype=float
    )

    subroi_max = np.array(
        subrois[name]["max"],
        dtype=float
    )

    ratio = float(
        subrois[name]["ratio"]
    )

    detail_level = subrois[name]["detail_level"]

    mesh_name = MESH_MAPPING[name]


    # --------------------------------------------------------
    # Calculate Sub-ROI volume
    # --------------------------------------------------------

    subroi_volume = box_volume(
        subroi_min,
        subroi_max
    )


    # --------------------------------------------------------
    # Calculate overlap volume
    # --------------------------------------------------------

    overlap_volume = intersection_volume(
        CURRENT_ROI_MIN,
        CURRENT_ROI_MAX,
        subroi_min,
        subroi_max
    )


    # --------------------------------------------------------
    # Calculate overlap ratio
    # --------------------------------------------------------

    if current_roi_volume > 0:

        overlap_ratio = (
            overlap_volume
            / current_roi_volume
        )

    else:

        overlap_ratio = 0.0


    # --------------------------------------------------------
    # Calculate IoU
    # --------------------------------------------------------

    iou = calculate_iou(
        CURRENT_ROI_MIN,
        CURRENT_ROI_MAX,
        subroi_min,
        subroi_max
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "subroi": name,

        "ratio": ratio,

        "detail_level": detail_level,

        "mesh": mesh_name,

        "subroi_volume": float(
            subroi_volume
        ),

        "overlap_volume": float(
            overlap_volume
        ),

        "overlap_ratio": float(
            overlap_ratio
        ),

        "iou": float(
            iou
        )
    }

    results.append(result)


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(f"\n{name}")
    print(f"  Ratio          : {ratio:.2f}")
    print(f"  Detail level   : {detail_level}")
    print(f"  Sub-ROI volume : {subroi_volume:.6f}")
    print(f"  Overlap volume : {overlap_volume:.6f}")
    print(f"  Overlap ratio  : {overlap_ratio:.6f}")
    print(f"  IoU            : {iou:.6f}")
    print(f"  Mesh           : {mesh_name}")


# ============================================================
# 9. SELECT BEST SUB-ROI
# ============================================================
#
# The Sub-ROI with the highest IoU is selected.
#
# If two Sub-ROIs have the same IoU, the finer Sub-ROI
# is preferred.
#
# Priority:
#     R0 > R1 > R2 > R3
# ============================================================

print("\n[4] SELECT BEST SUB-ROI")
print("-" * 75)

priority = {
    "R0": 0,
    "R1": 1,
    "R2": 2,
    "R3": 3
}


selected = max(
    results,
    key=lambda x: (
        x["iou"],
        -priority[x["subroi"]]
    )
)


selected_subroi = selected["subroi"]
selected_mesh = selected["mesh"]
selected_detail = selected["detail_level"]
selected_iou = selected["iou"]


print(
    f"Selected Sub-ROI : "
    f"{selected_subroi}"
)

print(
    f"Selected Mesh    : "
    f"{selected_mesh}"
)

print(
    f"Detail Level     : "
    f"{selected_detail}"
)

print(
    f"Best IoU         : "
    f"{selected_iou:.6f}"
)


# ============================================================
# 10. LOAD SELECTED MESH
# ============================================================

print("\n[5] LOAD SELECTED MESH")
print("-" * 75)


selected_mesh_path = os.path.join(
    ADAPTIVE_MESH_DIR,
    selected_mesh
)


print("Mesh path:")
print(selected_mesh_path)


if not os.path.exists(
    selected_mesh_path
):
    raise FileNotFoundError(
        f"Selected mesh not found:\n"
        f"{selected_mesh_path}"
    )


mesh = o3d.io.read_triangle_mesh(
    selected_mesh_path
)


if mesh.is_empty():
    raise RuntimeError(
        "Selected mesh is empty."
    )


print("\nMesh loaded successfully.")

print(
    f"Vertices  : "
    f"{len(mesh.vertices)}"
)

print(
    f"Triangles : "
    f"{len(mesh.triangles)}"
)


# ============================================================
# 11. SAVE SELECTED MESH
# ============================================================

print("\n[6] SAVE SELECTED MESH")
print("-" * 75)


success = o3d.io.write_triangle_mesh(
    OUTPUT_MESH,
    mesh
)


if not success:
    raise RuntimeError(
        f"Failed to save selected mesh:\n"
        f"{OUTPUT_MESH}"
    )


print("Selected mesh saved:")
print(OUTPUT_MESH)


# ============================================================
# 12. SAVE SELECTION REPORT
# ============================================================
#
# The report stores the complete selection process so that
# the result can be inspected later without rerunning Step 7.
# ============================================================

print("\n[7] SAVE SELECTION REPORT")
print("-" * 75)


report = {

    "step": 7,

    "current_task_roi": {

        "min": CURRENT_ROI_MIN.tolist(),

        "max": CURRENT_ROI_MAX.tolist(),

        "volume": current_roi_volume
    },

    "subroi_candidates": results,

    "selected_subroi": selected_subroi,

    "selected_mesh": selected_mesh,

    "selected_detail_level": selected_detail,

    "selected_iou": selected_iou,

    "selected_mesh_statistics": {

        "vertices": len(mesh.vertices),

        "triangles": len(mesh.triangles)
    },

    "selected_mesh_path": OUTPUT_MESH,

    "selection_rule":
        "Select the Sub-ROI with the highest 3D IoU "
        "with the current Task ROI. "
        "If IoU is equal, prefer the finer Sub-ROI."
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


print("Report saved:")
print(OUTPUT_REPORT)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 7 COMPLETED")
print("=" * 75)

print("\nCurrent Task ROI:")
print("  Min:", CURRENT_ROI_MIN)
print("  Max:", CURRENT_ROI_MAX)
print(
    f"  Volume: {current_roi_volume:.6f}"
)

print("\nCandidate comparison:")

for result in results:

    print(
        f"  {result['subroi']} | "
        f"Ratio={result['ratio']:.2f} | "
        f"Overlap={result['overlap_volume']:.6f} | "
        f"OverlapRatio={result['overlap_ratio']:.6f} | "
        f"IoU={result['iou']:.6f} | "
        f"{result['mesh']}"
    )


print("\nFinal selection:")
print(
    f"  Sub-ROI : {selected_subroi}"
)

print(
    f"  Mesh    : {selected_mesh}"
)

print(
    f"  Detail  : {selected_detail}"
)

print(
    f"  IoU     : {selected_iou:.6f}"
)

print(
    f"  Vertices: {len(mesh.vertices)}"
)

print(
    f"  Triangles: {len(mesh.triangles)}"
)

print("\nOutput:")
print(f"  {OUTPUT_MESH}")

print("\nReport:")
print(f"  {OUTPUT_REPORT}")

print("=" * 75)