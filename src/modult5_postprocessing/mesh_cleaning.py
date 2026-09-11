import open3d as o3d
import numpy as np
import os


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = r"src\modult5_postprocessing\Output\Object_ROI1.ply"

OUTPUT_DIR = r"src\modult5_postprocessing\Output"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "Cleaned_Object_ROI1.ply"
)

# Threshold for degenerate triangle area
EPSILON = 1e-12


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("STEP 1 - MESH CLEANING")
print("=" * 60)

print("\nLoading:", INPUT_PATH)

mesh = o3d.io.read_triangle_mesh(INPUT_PATH)

if mesh.is_empty():
    raise RuntimeError(
        "Mesh is empty or cannot be loaded."
    )


# ============================================================
# BEFORE CLEANING
# ============================================================

print("\nBefore cleaning:")

print("Vertices :", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))

vertices_before = np.asarray(mesh.vertices).copy()
triangles_before = np.asarray(mesh.triangles).copy()

num_vertices_before = len(vertices_before)
num_triangles_before = len(triangles_before)


# ============================================================
# [1] REMOVE NON-FINITE VERTICES
# ============================================================

print("\n[1] Checking non-finite vertices...")

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

finite_mask = np.isfinite(vertices).all(axis=1)

invalid_vertex_indices = np.where(
    ~finite_mask
)[0]

invalid_vertices = len(
    invalid_vertex_indices
)

print("Invalid vertices:", invalid_vertices)

if invalid_vertices > 0:

    # Mapping:
    # old vertex index -> new vertex index
    vertex_map = -np.ones(
        len(vertices),
        dtype=np.int64
    )

    valid_indices = np.where(
        finite_mask
    )[0]

    vertex_map[valid_indices] = np.arange(
        len(valid_indices)
    )

    # Keep only triangles whose three vertices
    # are valid
    triangle_valid_mask = finite_mask[
        triangles
    ].all(axis=1)

    triangles = triangles[
        triangle_valid_mask
    ]

    # Update triangle indices
    triangles = vertex_map[
        triangles
    ]

    vertices = vertices[
        finite_mask
    ]

    mesh.vertices = o3d.utility.Vector3dVector(
        vertices
    )

    mesh.triangles = o3d.utility.Vector3iVector(
        triangles
    )

    print(
        "Removed triangles containing "
        "invalid vertices:",
        np.sum(~triangle_valid_mask)
    )


# ============================================================
# [2] REMOVE DUPLICATED TRIANGLES
# ============================================================

print("\n[2] Removing duplicated triangles...")

triangles = np.asarray(
    mesh.triangles
)

# Sort vertex indices inside each triangle.
#
# Example:
# [10, 5, 7] -> [5, 7, 10]
#
# This allows triangles with different
# vertex ordering to be detected as duplicates.

sorted_triangles = np.sort(
    triangles,
    axis=1
)

_, unique_indices = np.unique(
    sorted_triangles,
    axis=0,
    return_index=True
)

unique_indices = np.sort(
    unique_indices
)

mesh.triangles = (
    o3d.utility.Vector3iVector(
        triangles[unique_indices]
    )
)

removed_duplicate_triangles = (
    len(triangles)
    - len(unique_indices)
)

print(
    "Duplicated triangles removed:",
    removed_duplicate_triangles
)


# ============================================================
# [3] REMOVE DEGENERATED TRIANGLES
# ============================================================

print("\n[3] Removing degenerated triangles...")

triangles = np.asarray(
    mesh.triangles
)

vertices = np.asarray(
    mesh.vertices
)

if len(triangles) > 0:

    # Triangle vertices
    v0 = vertices[
        triangles[:, 0]
    ]

    v1 = vertices[
        triangles[:, 1]
    ]

    v2 = vertices[
        triangles[:, 2]
    ]

    # Triangle area:
    #
    # A = 0.5 * |(v1-v0) x (v2-v0)|

    cross = np.cross(
        v1 - v0,
        v2 - v0
    )

    area = 0.5 * np.linalg.norm(
        cross,
        axis=1
    )

    valid_area = (
        area > EPSILON
    )

    invalid_area_count = np.sum(
        ~valid_area
    )

    mesh.triangles = (
        o3d.utility.Vector3iVector(
            triangles[valid_area]
        )
    )

else:

    invalid_area_count = 0

print(
    "Degenerated triangles removed:",
    invalid_area_count
)


# ============================================================
# [4] REMOVE UNREFERENCED VERTICES
# ============================================================

print("\n[4] Removing unreferenced vertices...")

vertices_before_cleanup = len(
    mesh.vertices
)

mesh.remove_unreferenced_vertices()

vertices_after_cleanup = len(
    mesh.vertices
)

print(
    "Unreferenced vertices removed:",
    vertices_before_cleanup
    - vertices_after_cleanup
)


# ============================================================
# [5] REMOVE DUPLICATED VERTICES
# ============================================================

print("\n[5] Removing duplicated vertices...")

vertices_before_cleanup = len(
    mesh.vertices
)

mesh.remove_duplicated_vertices()

vertices_after_cleanup = len(
    mesh.vertices
)

print(
    "Duplicated vertices removed:",
    vertices_before_cleanup
    - vertices_after_cleanup
)


# ============================================================
# [6] FINAL DEGENERATE TRIANGLE CLEANUP
# ============================================================

print(
    "\n[6] Final degenerate triangle cleanup..."
)

triangles_before_cleanup = len(
    mesh.triangles
)

mesh.remove_degenerate_triangles()

triangles_after_cleanup = len(
    mesh.triangles
)

print(
    "Degenerated triangles removed:",
    triangles_before_cleanup
    - triangles_after_cleanup
)


# ============================================================
# [7] FINAL DUPLICATED TRIANGLE CLEANUP
# ============================================================

print(
    "\n[7] Final duplicated triangle cleanup..."
)

triangles_before_cleanup = len(
    mesh.triangles
)

mesh.remove_duplicated_triangles()

triangles_after_cleanup = len(
    mesh.triangles
)

print(
    "Duplicated triangles removed:",
    triangles_before_cleanup
    - triangles_after_cleanup
)


# ============================================================
# [8] FINAL UNREFERENCED VERTEX CLEANUP
# ============================================================

print(
    "\n[8] Final unreferenced vertex cleanup..."
)

vertices_before_cleanup = len(
    mesh.vertices
)

mesh.remove_unreferenced_vertices()

vertices_after_cleanup = len(
    mesh.vertices
)

print(
    "Unreferenced vertices removed:",
    vertices_before_cleanup
    - vertices_after_cleanup
)


# ============================================================
# [9] COMPUTE NORMALS
# ============================================================

print("\n[9] Computing vertex normals...")

mesh.compute_vertex_normals()


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("CLEANING RESULT")
print("=" * 60)

num_vertices_after = len(
    mesh.vertices
)

num_triangles_after = len(
    mesh.triangles
)

print(
    "Vertices before:",
    num_vertices_before
)

print(
    "Triangles before:",
    num_triangles_before
)

print(
    "Vertices after:",
    num_vertices_after
)

print(
    "Triangles after:",
    num_triangles_after
)

vertex_reduction = (
    (num_vertices_before - num_vertices_after)
    / num_vertices_before * 100
    if num_vertices_before > 0
    else 0
)

triangle_reduction = (
    (num_triangles_before - num_triangles_after)
    / num_triangles_before * 100
    if num_triangles_before > 0
    else 0
)

print(
    f"Vertex reduction : "
    f"{vertex_reduction:.2f}%"
)

print(
    f"Triangle reduction: "
    f"{triangle_reduction:.2f}%"
)


# ============================================================
# CHECK EMPTY
# ============================================================

if mesh.is_empty():

    raise RuntimeError(
        "Mesh became empty after cleaning."
    )


# ============================================================
# SAVE
# ============================================================

print("\nSaving cleaned mesh...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

success = o3d.io.write_triangle_mesh(
    OUTPUT_PATH,
    mesh
)

if not success:

    raise RuntimeError(
        "Failed to save cleaned mesh."
    )

print("\nSaved:")
print(OUTPUT_PATH)


# ============================================================
# VISUALIZATION
# ============================================================

print("\nOpening cleaned mesh...")

mesh.paint_uniform_color(
    [0.7, 0.7, 0.7]
)

o3d.visualization.draw_geometries(
    [mesh],
    window_name="Cleaned Object Mesh",
    width=1400,
    height=900
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 COMPLETED")
print("=" * 60)