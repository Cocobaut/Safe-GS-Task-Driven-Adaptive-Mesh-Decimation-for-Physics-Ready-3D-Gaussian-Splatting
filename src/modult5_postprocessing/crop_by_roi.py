import open3d as o3d
import numpy as np
import os


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = r"src\modult5_postprocessing\Input\office0_mesh.ply"

OUTPUT_DIR = r"src\modult5_postprocessing\Output"

# ROI bạn vừa chọn
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
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MESH
# ============================================================

print("=" * 60)
print("Loading mesh...")
print("=" * 60)

mesh = o3d.io.read_triangle_mesh(INPUT_PATH)

if mesh.is_empty():
    raise RuntimeError("Cannot load mesh.")

mesh.compute_vertex_normals()

print("Vertices :", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))


# ============================================================
# CREATE ROI BOUNDING BOX
# ============================================================

roi_box = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=ROI_MIN,
    max_bound=ROI_MAX
)

print("\nROI:")
print("Min:", ROI_MIN)
print("Max:", ROI_MAX)


# ============================================================
# CROP OBJECT MESH
# ============================================================

print("\nCropping object inside ROI...")

object_mesh = mesh.crop(roi_box)

print("\nObject Mesh:")
print("Vertices :", len(object_mesh.vertices))
print("Triangles:", len(object_mesh.triangles))


# ============================================================
# CREATE SCENE MESH
# ============================================================

print("\nCreating Scene Mesh...")

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)


# Một triangle được giữ lại nếu centroid của nó
# nằm ngoài ROI.
triangle_centers = (
    vertices[triangles[:, 0]]
    + vertices[triangles[:, 1]]
    + vertices[triangles[:, 2]]
) / 3.0


inside = np.all(
    (triangle_centers >= ROI_MIN) &
    (triangle_centers <= ROI_MAX),
    axis=1
)

scene_triangles = triangles[~inside]


# ============================================================
# REBUILD SCENE MESH
# ============================================================

used_vertices = np.unique(scene_triangles)

old_to_new = -np.ones(
    len(vertices),
    dtype=np.int64
)

old_to_new[used_vertices] = np.arange(
    len(used_vertices)
)

scene_vertices = vertices[used_vertices]

scene_triangles_new = old_to_new[scene_triangles]


scene_mesh = o3d.geometry.TriangleMesh()

scene_mesh.vertices = o3d.utility.Vector3dVector(
    scene_vertices
)

scene_mesh.triangles = o3d.utility.Vector3iVector(
    scene_triangles_new
)

scene_mesh.compute_vertex_normals()


print("\nScene Mesh:")
print("Vertices :", len(scene_mesh.vertices))
print("Triangles:", len(scene_mesh.triangles))


# ============================================================
# SAVE
# ============================================================

object_path = os.path.join(
    OUTPUT_DIR,
    "Object_ROI1.ply"
)

scene_path = os.path.join(
    OUTPUT_DIR,
    "Scene_ROI1.ply"
)


print("\nSaving...")

o3d.io.write_triangle_mesh(
    object_path,
    object_mesh
)

o3d.io.write_triangle_mesh(
    scene_path,
    scene_mesh
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("Object Mesh:")
print(object_path)

print("\nScene Mesh:")
print(scene_path)


# ============================================================
# VISUALIZATION
# ============================================================

print("\nOpening visualization...")

# Object màu mặc định
object_mesh.paint_uniform_color(
    [0.8, 0.2, 0.2]
)

# Scene màu xám
scene_mesh.paint_uniform_color(
    [0.6, 0.6, 0.6]
)

o3d.visualization.draw_geometries(
    [
        scene_mesh,
        object_mesh
    ],
    window_name="Scene + Object ROI",
    width=1400,
    height=900
)