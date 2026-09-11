import os
import json
import numpy as np
import open3d as o3d


# ============================================================
# STEP 9 - COLLISION MESH GENERATION
# ============================================================

BASE_DIR = r"src\modult5_postprocessing"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "Output",
    "MeshQualityValidation_ROI1",
    "Validated_Mesh_ROI1.ply"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "CollisionMesh_ROI1"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "Collision_Mesh_ROI1.ply"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "CollisionMeshReport_ROI1.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# COLLISION MESH TARGET
# ============================================================

# Initial target for collision mesh.
# The collision mesh should be lighter than the selected
# visual/adaptive mesh.

TARGET_TRIANGLES = 6000


# ============================================================
# TOPOLOGY FUNCTIONS
# ============================================================

def build_edge_map(triangles):

    edge_map = {}

    for tri_id, tri in enumerate(triangles):

        v0, v1, v2 = map(int, tri)

        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0))),
        ]

        for edge in edges:

            if edge not in edge_map:
                edge_map[edge] = []

            edge_map[edge].append(tri_id)

    return edge_map


def topology_statistics(mesh):

    triangles = np.asarray(mesh.triangles)

    edge_map = build_edge_map(triangles)

    boundary_edges = 0
    non_manifold_edges = 0
    manifold_edges = 0

    for incident_triangles in edge_map.values():

        count = len(incident_triangles)

        if count == 1:
            boundary_edges += 1

        elif count == 2:
            manifold_edges += 1

        elif count > 2:
            non_manifold_edges += 1

    return {
        "edges": len(edge_map),
        "boundary_edges": boundary_edges,
        "manifold_edges": manifold_edges,
        "non_manifold_edges": non_manifold_edges,
        "watertight": (
            boundary_edges == 0
            and non_manifold_edges == 0
        ),
    }


# ============================================================
# LOAD INPUT
# ============================================================

print("=" * 60)
print("STEP 9 - COLLISION MESH GENERATION")
print("=" * 60)

print()
print("Input mesh:")
print(INPUT_PATH)

if not os.path.exists(INPUT_PATH):

    raise FileNotFoundError(
        f"Input mesh not found:\n{INPUT_PATH}"
    )


mesh = o3d.io.read_triangle_mesh(
    INPUT_PATH
)

if mesh.is_empty():

    raise RuntimeError(
        "Input mesh is empty."
    )


mesh.compute_vertex_normals()


# ============================================================
# ORIGINAL STATISTICS
# ============================================================

original_vertices = len(mesh.vertices)
original_triangles = len(mesh.triangles)

original_topology = topology_statistics(mesh)


print()
print("-" * 60)
print("INPUT MESH")
print("-" * 60)

print(
    f"Vertices        : {original_vertices}"
)

print(
    f"Triangles       : {original_triangles}"
)

print(
    f"Boundary edges  : "
    f"{original_topology['boundary_edges']}"
)

print(
    f"Non-manifold    : "
    f"{original_topology['non_manifold_edges']}"
)

print(
    f"Watertight      : "
    f"{original_topology['watertight']}"
)


# ============================================================
# COLLISION MESH SIMPLIFICATION
# ============================================================

print()
print("-" * 60)
print("COLLISION MESH SIMPLIFICATION")
print("-" * 60)

if original_triangles > TARGET_TRIANGLES:

    print(
        f"Target triangles: {TARGET_TRIANGLES}"
    )

    collision_mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=TARGET_TRIANGLES
    )

else:

    print(
        "Input mesh is already below target."
    )

    collision_mesh = mesh


# ============================================================
# CLEANUP
# ============================================================

print()
print("-" * 60)
print("CLEANUP")
print("-" * 60)

collision_mesh.remove_degenerate_triangles()
collision_mesh.remove_duplicated_triangles()
collision_mesh.remove_unreferenced_vertices()

collision_mesh.compute_vertex_normals()


# ============================================================
# FINAL STATISTICS
# ============================================================

final_vertices = len(
    collision_mesh.vertices
)

final_triangles = len(
    collision_mesh.triangles
)

final_topology = topology_statistics(
    collision_mesh
)


polygon_reduction = (
    1.0
    -
    final_triangles / original_triangles
) * 100.0


print()
print("-" * 60)
print("COLLISION MESH RESULT")
print("-" * 60)

print(
    f"Vertices        : {final_vertices}"
)

print(
    f"Triangles       : {final_triangles}"
)

print(
    f"Polygon reduction: "
    f"{polygon_reduction:.2f}%"
)

print(
    f"Boundary edges  : "
    f"{final_topology['boundary_edges']}"
)

print(
    f"Non-manifold    : "
    f"{final_topology['non_manifold_edges']}"
)

print(
    f"Watertight      : "
    f"{final_topology['watertight']}"
)


# ============================================================
# VALIDITY CHECK
# ============================================================

valid_mesh = (
    final_vertices > 0
    and
    final_triangles > 0
    and
    final_topology["non_manifold_edges"] == 0
)


print()

if valid_mesh:
    print("Collision mesh validity: PASS")
else:
    print("Collision mesh validity: FAIL")


# ============================================================
# SAVE
# ============================================================

success = o3d.io.write_triangle_mesh(
    OUTPUT_PATH,
    collision_mesh
)

if not success:

    raise RuntimeError(
        "Failed to save collision mesh."
    )


# ============================================================
# REPORT
# ============================================================

report = {

    "step": 9,

    "input_mesh": INPUT_PATH,

    "output_mesh": OUTPUT_PATH,

    "method":
        "Quadratic Error Metrics (QEM) simplification",

    "target_triangles":
        TARGET_TRIANGLES,

    "input": {

        "vertices":
            original_vertices,

        "triangles":
            original_triangles,

        "boundary_edges":
            original_topology["boundary_edges"],

        "non_manifold_edges":
            original_topology["non_manifold_edges"],

        "watertight":
            original_topology["watertight"]
    },

    "output": {

        "vertices":
            final_vertices,

        "triangles":
            final_triangles,

        "boundary_edges":
            final_topology["boundary_edges"],

        "non_manifold_edges":
            final_topology["non_manifold_edges"],

        "watertight":
            final_topology["watertight"],

        "polygon_reduction_percent":
            polygon_reduction
    },

    "valid":
        valid_mesh
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("STEP 9 COMPLETED")
print("=" * 60)

print()
print("Collision mesh:")
print(OUTPUT_PATH)

print()
print("Report:")
print(REPORT_PATH)

print()
print("=" * 60)