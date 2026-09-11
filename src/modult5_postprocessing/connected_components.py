import open3d as o3d
import numpy as np
import os
import json


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    r"src\modult5_postprocessing\Output"
    r"\Cleaned_Object_ROI1.ply"
)

OUTPUT_DIR = (
    r"src\modult5_postprocessing\Output"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "ComponentFiltered_Object_ROI1.ply"
)

LABEL_PATH = os.path.join(
    OUTPUT_DIR,
    "ConnectedComponents_ROI1.npy"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "ConnectedComponentReport_ROI1.json"
)


# ============================================================
# THRESHOLD
# ============================================================

# Component có số triangle nhỏ hơn threshold
# sẽ được xem là noise / small component.
#
# Mục tiêu hiện tại:
# dùng threshold 2900 để loại component của tường.

SMALL_TRIANGLES = 2900

# Chỉ dùng để thống kê trong report.
# Không dùng trực tiếp để quyết định REMOVE.
SMALL_AREA = 0.005


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 2 - CONNECTED COMPONENT ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD MESH
    # --------------------------------------------------------

    print("\n[1] Loading mesh...")

    mesh = o3d.io.read_triangle_mesh(
        INPUT_PATH
    )

    if mesh.is_empty():
        raise RuntimeError(
            "Mesh is empty or cannot be loaded."
        )

    print(
        "Vertices :",
        len(mesh.vertices)
    )

    print(
        "Triangles:",
        len(mesh.triangles)
    )

    # Remove vertices that are not used.
    # This does not change triangle topology/order.
    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------------
    # 2. CONNECTED COMPONENT ANALYSIS
    # --------------------------------------------------------

    print(
        "\n[2] Analyzing connected components..."
    )

    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area
    ) = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(
        triangle_clusters
    )

    cluster_n_triangles = np.asarray(
        cluster_n_triangles
    )

    cluster_area = np.asarray(
        cluster_area
    )

    num_components = len(
        cluster_n_triangles
    )

    print(
        "\nNumber of components:",
        num_components
    )

    # --------------------------------------------------------
    # 3. COMPONENT INFORMATION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "COMPONENT INFORMATION"
    )

    print(
        "=" * 75
    )

    print(
        f"{'ID':>6} "
        f"{'Triangles':>12} "
        f"{'Area':>15} "
        f"{'Decision':>12}"
    )

    print("-" * 75)

    keep_components = []
    remove_components = []

    # Sort components from largest to smallest
    order = np.argsort(
        -cluster_n_triangles
    )

    for component_id in order:

        n_triangles = int(
            cluster_n_triangles[
                component_id
            ]
        )

        area = float(
            cluster_area[
                component_id
            ]
        )

        # ----------------------------------------------------
        # Noise criterion
        # ----------------------------------------------------
        #
        # Component is removed ONLY when
        # number of triangles < 2900.
        #
        # Area is reported but not used for filtering.
        #

        is_small = (
            n_triangles < SMALL_TRIANGLES
        )

        if is_small:

            decision = "REMOVE"

            remove_components.append(
                int(component_id)
            )

        else:

            decision = "KEEP"

            keep_components.append(
                int(component_id)
            )

        print(
            f"{component_id:6d} "
            f"{n_triangles:12d} "
            f"{area:15.6f} "
            f"{decision:>12}"
        )

    # --------------------------------------------------------
    # 4. MAIN COMPONENT
    # --------------------------------------------------------

    largest_component = int(
        order[0]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MAIN COMPONENT"
    )

    print(
        "=" * 60
    )

    print(
        "Largest component ID:",
        largest_component
    )

    print(
        "Triangles:",
        int(
            cluster_n_triangles[
                largest_component
            ]
        )
    )

    print(
        "Area:",
        float(
            cluster_area[
                largest_component
            ]
        )
    )

    # --------------------------------------------------------
    # 5. BUILD TRIANGLE MASK
    # --------------------------------------------------------

    print(
        "\n[3] Filtering noisy components..."
    )

    keep_mask = np.isin(
        triangle_clusters,
        keep_components
    )

    remove_mask = ~keep_mask

    removed_triangles = int(
        np.sum(remove_mask)
    )

    kept_triangles = int(
        np.sum(keep_mask)
    )

    print(
        "Triangles removed:",
        removed_triangles
    )

    print(
        "Triangles kept:",
        kept_triangles
    )

    # --------------------------------------------------------
    # 6. CREATE FILTERED MESH
    # --------------------------------------------------------

    print(
        "\n[4] Creating filtered mesh..."
    )

    triangles = np.asarray(
        mesh.triangles
    )

    vertices = np.asarray(
        mesh.vertices
    )

    filtered_triangles = (
        triangles[keep_mask]
    )

    filtered_mesh = (
        o3d.geometry.TriangleMesh()
    )

    filtered_mesh.vertices = (
        o3d.utility.Vector3dVector(
            vertices
        )
    )

    filtered_mesh.triangles = (
        o3d.utility.Vector3iVector(
            filtered_triangles
        )
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Do NOT run:
    #
    # remove_degenerate_triangles()
    # remove_duplicated_triangles()
    #
    # here because these operations can change the
    # triangle ordering/count and make filtered_labels
    # no longer correspond to the output mesh.
    #
    # Bước 1 already performs mesh cleaning.
    #

    filtered_mesh.remove_unreferenced_vertices()

    filtered_mesh.compute_vertex_normals()

    # --------------------------------------------------------
    # 7. SAVE FILTERED COMPONENT LABELS
    # --------------------------------------------------------

    print(
        "\n[5] Saving component labels..."
    )

    # These labels correspond to the triangles
    # selected by keep_mask.
    #
    # Triangle ordering is preserved.

    filtered_labels = (
        triangle_clusters[keep_mask]
    )

    np.save(
        LABEL_PATH,
        filtered_labels
    )

    print(
        "Component labels saved:"
    )

    print(
        LABEL_PATH
    )

    # --------------------------------------------------------
    # 8. SAVE FILTERED MESH
    # --------------------------------------------------------

    print(
        "\n[6] Saving component-filtered mesh..."
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    success = o3d.io.write_triangle_mesh(
        OUTPUT_PATH,
        filtered_mesh
    )

    if not success:
        raise RuntimeError(
            "Failed to save filtered mesh."
        )

    print(
        "Saved:",
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # 9. REPORT
    # --------------------------------------------------------

    triangles_before = len(
        mesh.triangles
    )

    triangles_after = len(
        filtered_mesh.triangles
    )

    triangle_reduction = (
        (
            triangles_before
            - triangles_after
        )
        / triangles_before
        * 100
        if triangles_before > 0
        else 0
    )

    # Count components satisfying area threshold
    small_area_components = [
        int(component_id)
        for component_id in range(
            num_components
        )
        if cluster_area[component_id]
        < SMALL_AREA
    ]

    report = {

        "input":
            INPUT_PATH,

        "output":
            OUTPUT_PATH,

        "number_of_components":
            int(num_components),

        "largest_component":
            int(largest_component),

        "components_kept":
            [
                int(x)
                for x in keep_components
            ],

        "components_removed":
            [
                int(x)
                for x in remove_components
            ],

        "small_triangle_threshold":
            SMALL_TRIANGLES,

        "small_area_threshold":
            SMALL_AREA,

        "area_threshold_used_for_filtering":
            False,

        "vertices_before":
            int(len(mesh.vertices)),

        "triangles_before":
            int(triangles_before),

        "vertices_after":
            int(len(filtered_mesh.vertices)),

        "triangles_after":
            int(triangles_after),

        "triangles_removed":
            int(
                triangles_before
                - triangles_after
            ),

        "triangle_reduction_percent":
            float(triangle_reduction),

        "components_below_triangle_threshold":
            int(
                len(remove_components)
            ),

        "components_below_area_threshold":
            int(
                len(small_area_components)
            )
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

    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )

    # --------------------------------------------------------
    # 10. FINAL RESULT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "CONNECTED COMPONENT RESULT"
    )

    print(
        "=" * 60
    )

    print(
        "Components before:",
        num_components
    )

    print(
        "Components kept:",
        len(keep_components)
    )

    print(
        "Components removed:",
        len(remove_components)
    )

    print(
        "Triangles before:",
        triangles_before
    )

    print(
        "Triangles after:",
        triangles_after
    )

    print(
        f"Triangle reduction: "
        f"{triangle_reduction:.2f}%"
    )

    # --------------------------------------------------------
    # 11. VISUALIZATION
    # --------------------------------------------------------

    print(
        "\nOpening filtered mesh..."
    )

    filtered_mesh.paint_uniform_color(
        [0.7, 0.7, 0.7]
    )

    o3d.visualization.draw_geometries(
        [filtered_mesh],
        window_name="Component Filtered Mesh",
        width=1400,
        height=900
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "STEP 2 COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()