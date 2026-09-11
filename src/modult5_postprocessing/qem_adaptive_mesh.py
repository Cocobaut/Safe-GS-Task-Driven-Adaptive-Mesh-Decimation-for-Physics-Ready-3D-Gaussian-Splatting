import json
import os
import sys

import open3d as o3d

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from roi_config import (
    ADAPTIVE_DIR,
    QEM_REPORT,
    REPAIRED_MESH,
    TARGET_TRIANGLES,
    ensure_dir,
)


def mesh_statistics(mesh):
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
    }


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def simplify_mesh(mesh, target_triangles):
    """
    Simplify the mesh using QEM.
    """

    if (
        target_triangles is None
        or target_triangles <= 0
        or len(mesh.triangles) <= target_triangles
    ):
        return o3d.geometry.TriangleMesh(mesh)

    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_triangles)
    )

    return clean_mesh(simplified)


def main():

    print("=" * 70)
    print("STEP 5 - QEM ADAPTIVE MESH GENERATION")
    print("=" * 70)

    ensure_dir(ADAPTIVE_DIR)

    # ------------------------------------------------------------
    # 1. Load topology-repaired mesh
    # ------------------------------------------------------------
    print("\n[1] Loading topology-repaired mesh...")

    print("Input:", REPAIRED_MESH)

    mesh = o3d.io.read_triangle_mesh(REPAIRED_MESH)

    if mesh.is_empty():
        raise RuntimeError(
            "Input mesh is empty or cannot be loaded."
        )

    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    original_stats = mesh_statistics(mesh)

    print("Vertices :", original_stats["vertices"])
    print("Triangles:", original_stats["triangles"])

    # ------------------------------------------------------------
    # 2. Generate multiple QEM mesh levels
    # ------------------------------------------------------------
    print("\n[2] Generating QEM mesh levels...")

    report = {
        "input": REPAIRED_MESH,
        "original_mesh": original_stats,
        "method": "Global QEM simplification",
        "meshes": {},
    }

    for mesh_name, target in TARGET_TRIANGLES.items():

        print("\n" + "-" * 60)
        print(mesh_name)
        print("-" * 60)

        # M0 = original high-resolution mesh
        if target is None:
            current_mesh = o3d.geometry.TriangleMesh(mesh)

        else:
            current_mesh = simplify_mesh(
                mesh,
                target
            )

        stats = mesh_statistics(current_mesh)

        # Triangle reduction
        if original_stats["triangles"] > 0:
            reduction = (
                (original_stats["triangles"]
                 - stats["triangles"])
                / original_stats["triangles"]
                * 100.0
            )
        else:
            reduction = 0.0

        output_path = os.path.join(
            ADAPTIVE_DIR,
            f"{mesh_name}.ply"
        )

        success = o3d.io.write_triangle_mesh(
            output_path,
            current_mesh
        )

        if not success:
            raise RuntimeError(
                f"Failed to save {mesh_name}"
            )

        print("Saved:", output_path)
        print("Vertices :", stats["vertices"])
        print("Triangles:", stats["triangles"])
        print(f"Reduction: {reduction:.2f}%")

        report["meshes"][mesh_name] = {
            "target_triangles": (
                None
                if target is None
                else int(target)
            ),
            "actual_vertices": stats["vertices"],
            "actual_triangles": stats["triangles"],
            "triangle_reduction_percent": float(
                reduction
            ),
            "output": output_path,
        }

    # ------------------------------------------------------------
    # 3. Save report
    # ------------------------------------------------------------
    with open(
        QEM_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("\nReport saved:", QEM_REPORT)

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()