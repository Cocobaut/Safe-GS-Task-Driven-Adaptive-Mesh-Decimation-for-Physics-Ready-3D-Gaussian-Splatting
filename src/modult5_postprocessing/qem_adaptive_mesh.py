import open3d as o3d
import numpy as np
import os
import json


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    r"src\modult5_postprocessing\Output"
    r"\TopologyRepaired_Object_ROI1.ply"
)

OUTPUT_DIR = (
    r"src\modult5_postprocessing\Output"
    r"\AdaptiveMeshes_ROI1"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "QEMAdaptiveMeshReport_ROI1.json"
)


# ============================================================
# TARGET TRIANGLE COUNTS
# ============================================================

TARGET_TRIANGLES = {

    # Original mesh
    "M0_HighRes": None,

    # QEM simplification levels
    "M1_MediumHigh": 18000,

    "M2_Medium": 12000,

    "M3_Coarse": 6000
}


# ============================================================
# MESH STATISTICS
# ============================================================

def mesh_statistics(mesh):

    vertices = np.asarray(
        mesh.vertices
    )

    triangles = np.asarray(
        mesh.triangles
    )

    return {
        "vertices": int(
            len(vertices)
        ),

        "triangles": int(
            len(triangles)
        )
    }


# ============================================================
# CLEAN MESH
# ============================================================

def clean_mesh(mesh):

    # Remove invalid triangles
    mesh.remove_degenerate_triangles()

    # Remove duplicated triangles
    mesh.remove_duplicated_triangles()

    # Remove vertices that are not referenced
    mesh.remove_unreferenced_vertices()

    # Recalculate normals
    mesh.compute_vertex_normals()

    return mesh


# ============================================================
# QEM SIMPLIFICATION
# ============================================================

def simplify_mesh(
    mesh,
    target_triangles
):

    print(
        f"    Target triangles: "
        f"{target_triangles}"
    )

    simplified = (
        mesh.simplify_quadric_decimation(
            target_number_of_triangles=
            target_triangles
        )
    )

    simplified = clean_mesh(
        simplified
    )

    return simplified


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STEP 5 - QEM ADAPTIVE MESH GENERATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 2. LOAD INPUT MESH
    # --------------------------------------------------------

    print(
        "\n[1] Loading topology-repaired mesh..."
    )

    mesh = o3d.io.read_triangle_mesh(
        INPUT_PATH
    )

    if mesh.is_empty():

        raise RuntimeError(
            "Input mesh is empty or cannot be loaded."
        )

    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    original_stats = mesh_statistics(
        mesh
    )

    print(
        "Vertices :",
        original_stats["vertices"]
    )

    print(
        "Triangles:",
        original_stats["triangles"]
    )

    # --------------------------------------------------------
    # 3. GENERATE ADAPTIVE MESH LEVELS
    # --------------------------------------------------------

    print(
        "\n[2] Generating QEM mesh levels..."
    )

    report = {

        "input": INPUT_PATH,

        "original_mesh": original_stats,

        "meshes": {}
    }

    for mesh_name, target in (
        TARGET_TRIANGLES.items()
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            mesh_name
        )

        print(
            "-" * 60
        )

        # ----------------------------------------------------
        # M0 - ORIGINAL
        # ----------------------------------------------------

        if target is None:

            current_mesh = (
                o3d.geometry.TriangleMesh(
                    mesh
                )
            )

        # ----------------------------------------------------
        # M1/M2/M3 - QEM
        # ----------------------------------------------------

        else:

            # Never request a target larger
            # than the original mesh.

            if target >= len(mesh.triangles):

                print(
                    "Target is larger than "
                    "original mesh."
                )

                print(
                    "Using original mesh."
                )

                current_mesh = (
                    o3d.geometry.TriangleMesh(
                        mesh
                    )
                )

            else:

                current_mesh = simplify_mesh(
                    mesh,
                    target
                )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        stats = mesh_statistics(
            current_mesh
        )

        original_triangles = (
            original_stats[
                "triangles"
            ]
        )

        reduction = (

            (
                original_triangles
                - stats["triangles"]
            )
            / original_triangles
            * 100

            if original_triangles > 0
            else 0
        )

        stats[
            "triangle_reduction_percent"
        ] = float(reduction)

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{mesh_name}.ply"
        )

        success = (
            o3d.io.write_triangle_mesh(
                output_path,
                current_mesh
            )
        )

        if not success:

            raise RuntimeError(
                f"Failed to save {mesh_name}"
            )

        print(
            "Saved:",
            output_path
        )

        print(
            "Vertices:",
            stats["vertices"]
        )

        print(
            "Triangles:",
            stats["triangles"]
        )

        print(
            f"Reduction: "
            f"{reduction:.2f}%"
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        report["meshes"][mesh_name] = {

            "target_triangles":
                None
                if target is None
                else int(target),

            "actual_vertices":
                stats["vertices"],

            "actual_triangles":
                stats["triangles"],

            "triangle_reduction_percent":
                stats[
                    "triangle_reduction_percent"
                ],

            "output":
                output_path
        }

    # --------------------------------------------------------
    # 4. SAVE REPORT
    # --------------------------------------------------------

    print(
        "\n[3] Saving QEM report..."
    )

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

    print(
        "Report saved:"
    )

    print(
        REPORT_PATH
    )

    # --------------------------------------------------------
    # 5. SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "QEM ADAPTIVE MESH SUMMARY"
    )

    print(
        "=" * 65
    )

    for name, info in (
        report["meshes"].items()
    ):

        print(
            f"{name:20s} "
            f"{info['actual_triangles']:8d} triangles "
            f"reduction = "
            f"{info['triangle_reduction_percent']:.2f}%"
        )

    print(
        "\n" + "=" * 65
    )

    print(
        "STEP 5 COMPLETED"
    )

    print(
        "=" * 65
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()