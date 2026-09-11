import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roi_config import (
    CLEANED_MESH,
    DEGENERATE_AREA_EPS,
    OBJECT_MESH,
    OUTPUT_DIR,
    VERTEX_MERGE_EPS,
    ensure_dir,
    maybe_visualize,
)


def clean_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    finite_mask = np.isfinite(vertices).all(axis=1)
    invalid_vertices = int(np.sum(~finite_mask))
    print("Invalid vertices:", invalid_vertices)

    if invalid_vertices > 0:
        vertex_map = -np.ones(len(vertices), dtype=np.int64)
        valid_indices = np.where(finite_mask)[0]
        vertex_map[valid_indices] = np.arange(len(valid_indices))

        triangle_valid_mask = finite_mask[triangles].all(axis=1)
        triangles = vertex_map[triangles[triangle_valid_mask]]
        vertices = vertices[finite_mask]
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        print(
            "Removed triangles containing invalid vertices:",
            int(np.sum(~triangle_valid_mask)),
        )

    print("Merging vertices within epsilon:", VERTEX_MERGE_EPS)
    before = len(mesh.vertices)
    mesh.merge_close_vertices(VERTEX_MERGE_EPS)
    print("Duplicated / near-duplicate vertices removed:", before - len(mesh.vertices))

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    if len(triangles) > 0:
        sorted_triangles = np.sort(triangles, axis=1)
        _, unique_indices = np.unique(sorted_triangles, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        removed = len(triangles) - len(unique_indices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles[unique_indices])
        print("Duplicated triangles removed:", removed)

        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        valid_area = area > DEGENERATE_AREA_EPS
        print("Degenerated triangles removed:", int(np.sum(~valid_area)))
        mesh.triangles = o3d.utility.Vector3iVector(triangles[valid_area])
    else:
        print("Degenerated triangles removed: 0")

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def main():
    print("=" * 60)
    print("STEP 1 - MESH CLEANING")
    print("=" * 60)
    print("\nLoading:", OBJECT_MESH)

    mesh = o3d.io.read_triangle_mesh(OBJECT_MESH)
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty or cannot be loaded.")

    num_vertices_before = len(mesh.vertices)
    num_triangles_before = len(mesh.triangles)
    print("\nBefore cleaning:")
    print("Vertices :", num_vertices_before)
    print("Triangles:", num_triangles_before)

    mesh = clean_mesh(mesh)

    print("\n" + "=" * 60)
    print("CLEANING RESULT")
    print("=" * 60)
    print("Vertices before:", num_vertices_before)
    print("Triangles before:", num_triangles_before)
    print("Vertices after:", len(mesh.vertices))
    print("Triangles after:", len(mesh.triangles))

    if mesh.is_empty():
        raise RuntimeError("Mesh became empty after cleaning.")

    ensure_dir(OUTPUT_DIR)
    if not o3d.io.write_triangle_mesh(CLEANED_MESH, mesh):
        raise RuntimeError("Failed to save cleaned mesh.")

    print("\nSaved:", CLEANED_MESH)
    maybe_visualize(mesh, "Cleaned Object Mesh")
    print("\nSTEP 1 COMPLETED")


if __name__ == "__main__":
    main()
