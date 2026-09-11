import open3d as o3d
import numpy as np
import os
import json
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    r"src\modult5_postprocessing\Output"
    r"\ComponentFiltered_Object_ROI1.ply"
)

OUTPUT_DIR = (
    r"src\modult5_postprocessing\Output"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "TopologyRepaired_Object_ROI1.ply"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "TopologyRepairReport_ROI1.json"
)


# ============================================================
# TOPOLOGY ANALYSIS
# ============================================================

def analyze_topology(mesh):

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # --------------------------------------------------------
    # Build edge -> triangles map
    # --------------------------------------------------------

    edge_to_triangles = defaultdict(list)

    for triangle_id, tri in enumerate(triangles):

        v0, v1, v2 = tri

        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]

        for edge in edges:
            edge_to_triangles[edge].append(
                triangle_id
            )

    # --------------------------------------------------------
    # Classify edges
    # --------------------------------------------------------

    boundary_edges = []
    manifold_edges = []
    non_manifold_edges = []

    for edge, triangle_ids in edge_to_triangles.items():

        count = len(triangle_ids)

        if count == 1:
            boundary_edges.append(edge)

        elif count == 2:
            manifold_edges.append(edge)

        elif count > 2:
            non_manifold_edges.append(
                (edge, triangle_ids)
            )

    # --------------------------------------------------------
    # Boundary loops
    # --------------------------------------------------------

    boundary_graph = defaultdict(set)

    for v0, v1 in boundary_edges:

        boundary_graph[v0].add(v1)
        boundary_graph[v1].add(v0)

    visited_edges = set()
    boundary_loops = []

    for start_vertex in boundary_graph:

        for next_vertex in boundary_graph[start_vertex]:

            edge_key = tuple(
                sorted(
                    (start_vertex, next_vertex)
                )
            )

            if edge_key in visited_edges:
                continue

            loop = []

            current = start_vertex
            previous = None

            while True:

                loop.append(current)

                neighbors = boundary_graph[
                    current
                ]

                candidates = [
                    n for n in neighbors
                    if tuple(
                        sorted((current, n))
                    ) not in visited_edges
                ]

                if not candidates:
                    break

                # Prefer not going back
                if previous is not None:

                    non_back = [
                        n for n in candidates
                        if n != previous
                    ]

                    if non_back:
                        next_v = non_back[0]

                    else:
                        next_v = candidates[0]

                else:
                    next_v = candidates[0]

                visited_edges.add(
                    tuple(
                        sorted(
                            (current, next_v)
                        )
                    )
                )

                previous = current
                current = next_v

                if current == start_vertex:
                    break

            if len(loop) >= 3:
                boundary_loops.append(loop)

    # --------------------------------------------------------
    # Vertex manifoldness
    # --------------------------------------------------------

    vertex_triangle_map = defaultdict(list)

    for triangle_id, tri in enumerate(triangles):

        for vertex_id in tri:

            vertex_triangle_map[
                vertex_id
            ].append(triangle_id)

    non_manifold_vertices = []

    for vertex_id, triangle_ids in vertex_triangle_map.items():

        # Gather triangles incident to vertex
        incident_triangles = set(
            triangle_ids
        )

        # Build local adjacency through edges
        local_graph = defaultdict(set)

        for triangle_id in incident_triangles:

            tri = triangles[triangle_id]

            local_vertices = list(tri)

            for i in range(3):

                a = local_vertices[i]
                b = local_vertices[
                    (i + 1) % 3
                ]

                if (
                    a == vertex_id
                    or b == vertex_id
                ):

                    other = (
                        b
                        if a == vertex_id
                        else a
                    )

                    local_graph[
                        vertex_id
                    ].add(other)

        # Non-manifold edge incidents are already
        # handled separately.
        #
        # Here we only record vertices connected
        # to multiple local branches.

        if len(
            local_graph[vertex_id]
        ) > len(incident_triangles):

            non_manifold_vertices.append(
                int(vertex_id)
            )

    return {
        "vertices": len(vertices),
        "triangles": len(triangles),
        "edges": len(edge_to_triangles),
        "boundary_edges": boundary_edges,
        "manifold_edges": manifold_edges,
        "non_manifold_edges": non_manifold_edges,
        "boundary_loops": boundary_loops,
        "non_manifold_vertices":
            non_manifold_vertices
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3 - TOPOLOGY DETECTION AND REPAIR")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD
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

    # --------------------------------------------------------
    # 2. INITIAL TOPOLOGY ANALYSIS
    # --------------------------------------------------------

    print(
        "\n[2] Initial topology analysis..."
    )

    initial = analyze_topology(mesh)

    print("\n" + "=" * 70)
    print("INITIAL TOPOLOGY")
    print("=" * 70)

    print(
        "Vertices:",
        initial["vertices"]
    )

    print(
        "Triangles:",
        initial["triangles"]
    )

    print(
        "Edges:",
        initial["edges"]
    )

    print(
        "Manifold edges:",
        len(initial["manifold_edges"])
    )

    print(
        "Boundary edges:",
        len(initial["boundary_edges"])
    )

    print(
        "Non-manifold edges:",
        len(initial["non_manifold_edges"])
    )

    print(
        "Boundary loops:",
        len(initial["boundary_loops"])
    )

    print(
        "Non-manifold vertices:",
        len(initial["non_manifold_vertices"])
    )

    # --------------------------------------------------------
    # 3. SHOW NON-MANIFOLD EDGES
    # --------------------------------------------------------

    if initial["non_manifold_edges"]:

        print(
            "\n[3] Non-manifold edges detected:"
        )

        for edge, triangle_ids in (
            initial["non_manifold_edges"]
        ):

            print(
                f"Edge {edge} -> "
                f"triangles {triangle_ids}"
            )

    else:

        print(
            "\n[3] No non-manifold edges detected."
        )

    # --------------------------------------------------------
    # 4. REPAIR NON-MANIFOLD EDGES
    # --------------------------------------------------------

    print(
        "\n[4] Repairing non-manifold edges..."
    )

    triangles_before = len(
        mesh.triangles
    )

    # Open3D removes triangles associated
    # with non-manifold edges.
    mesh.remove_non_manifold_edges()

    triangles_after = len(
        mesh.triangles
    )

    removed_non_manifold_triangles = (
        triangles_before
        - triangles_after
    )

    print(
        "Triangles removed:",
        removed_non_manifold_triangles
    )

    # --------------------------------------------------------
    # 5. GENERAL TOPOLOGY CLEANUP
    # --------------------------------------------------------

    print(
        "\n[5] Cleaning topology..."
    )

    before = len(mesh.triangles)

    mesh.remove_degenerate_triangles()

    degenerate_removed = (
        before
        - len(mesh.triangles)
    )

    before = len(mesh.triangles)

    mesh.remove_duplicated_triangles()

    duplicate_removed = (
        before
        - len(mesh.triangles)
    )

    mesh.remove_unreferenced_vertices()

    print(
        "Degenerate triangles removed:",
        degenerate_removed
    )

    print(
        "Duplicated triangles removed:",
        duplicate_removed
    )

    # --------------------------------------------------------
    # 6. RECOMPUTE NORMALS
    # --------------------------------------------------------

    print(
        "\n[6] Recomputing normals..."
    )

    mesh.compute_vertex_normals()

    # --------------------------------------------------------
    # 7. FINAL TOPOLOGY ANALYSIS
    # --------------------------------------------------------

    print(
        "\n[7] Final topology analysis..."
    )

    final = analyze_topology(mesh)

    print("\n" + "=" * 70)
    print("FINAL TOPOLOGY")
    print("=" * 70)

    print(
        "Vertices:",
        final["vertices"]
    )

    print(
        "Triangles:",
        final["triangles"]
    )

    print(
        "Edges:",
        final["edges"]
    )

    print(
        "Manifold edges:",
        len(final["manifold_edges"])
    )

    print(
        "Boundary edges:",
        len(final["boundary_edges"])
    )

    print(
        "Non-manifold edges:",
        len(final["non_manifold_edges"])
    )

    print(
        "Boundary loops:",
        len(final["boundary_loops"])
    )

    print(
        "Non-manifold vertices:",
        len(final["non_manifold_vertices"])
    )

    # --------------------------------------------------------
    # 8. WATERTIGHT CHECK
    # --------------------------------------------------------

    boundary_edges_final = len(
        final["boundary_edges"]
    )

    non_manifold_edges_final = len(
        final["non_manifold_edges"]
    )

    watertight = (
        boundary_edges_final == 0
        and
        non_manifold_edges_final == 0
    )

    print(
        "\nWatertight:",
        watertight
    )

    # --------------------------------------------------------
    # 9. SAVE
    # --------------------------------------------------------

    print(
        "\n[8] Saving repaired mesh..."
    )

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
            "Failed to save repaired mesh."
        )

    print(
        "Saved:",
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # 10. SAVE REPORT
    # --------------------------------------------------------

    report = {

        "input": INPUT_PATH,

        "output": OUTPUT_PATH,

        "initial": {
            "vertices":
                int(initial["vertices"]),

            "triangles":
                int(initial["triangles"]),

            "edges":
                int(initial["edges"]),

            "boundary_edges":
                len(initial["boundary_edges"]),

            "non_manifold_edges":
                len(initial["non_manifold_edges"]),

            "boundary_loops":
                len(initial["boundary_loops"]),

            "non_manifold_vertices":
                len(
                    initial[
                        "non_manifold_vertices"
                    ]
                )
        },

        "repair": {
            "non_manifold_triangles_removed":
                int(
                    removed_non_manifold_triangles
                ),

            "degenerate_triangles_removed":
                int(
                    degenerate_removed
                ),

            "duplicated_triangles_removed":
                int(
                    duplicate_removed
                )
        },

        "final": {
            "vertices":
                int(final["vertices"]),

            "triangles":
                int(final["triangles"]),

            "edges":
                int(final["edges"]),

            "boundary_edges":
                len(final["boundary_edges"]),

            "non_manifold_edges":
                len(final["non_manifold_edges"]),

            "boundary_loops":
                len(final["boundary_loops"]),

            "non_manifold_vertices":
                len(
                    final[
                        "non_manifold_vertices"
                    ]
                ),

            "watertight":
                bool(watertight)
        }
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
        "Report saved:",
        REPORT_PATH
    )

    # --------------------------------------------------------
    # 11. VISUALIZATION
    # --------------------------------------------------------

    print(
        "\nOpening repaired mesh..."
    )

    mesh.paint_uniform_color(
        [0.7, 0.7, 0.7]
    )

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Topology Repaired Mesh",
        width=1400,
        height=900
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 3 COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()