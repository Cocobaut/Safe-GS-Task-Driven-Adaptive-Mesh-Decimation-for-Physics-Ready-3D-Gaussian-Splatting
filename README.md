## 📂 Cấu trúc dự án (Project Structure)

```text
adaptive-macro-micro-gs/
│
├── configs/                          # Quản lý cấu hình huấn luyện và mô phỏng
│   ├── base_scene.yaml               # Config huấn luyện Scene-GS (20k iters, camera settings)
│   ├── object_roi.yaml               # Config huấn luyện Object-GS (AABB, densification threshold)
│   ├── gof_meshing.yaml              # Config GOF (Tetrahedral grid, level set, binary search steps)
│   └── simulation.yaml               # Config Physics (MuJoCo/PhysX/PyBullet parameters, timestep)
│
├── data/                             # Dữ liệu & kết quả trung gian
│   ├── raw/                          # Dữ liệu ảnh thô (RGB images, camera captures)
│   ├── sfm/                          # Kết quả Colmap/SfM (cameras.bin, images.bin, points3D.bin)
│   ├── segmentation/                 # 2D Masks trích xuất từ SAM / GroundingDINO
│   └── workspace/                    # Cache trung gian qua từng module
│       ├── roi_boxes/                # Tọa độ 3D AABB / Voxel grid metadata (JSON/YAML)
│       ├── checkpoints/              # Checkpoints cho Scene-GS và Object-GS (.ply, .pt)
│       └── meshes/                   # Mesh xuất ra (.obj, .ply, watertight/processed)
│
├── submodules/                       # Custom CUDA Kernels hoặc C++ extensions
│   ├── diff_gaussian_rasterization/  # Rasterizer gốc hoặc tùy biến
│   └── gof_cuda/                     # Custom Ray-Gaussian intersection & Opacity evaluation CUDA
│
├── src/                              # Mã nguồn chính phân tầng theo Modules
│   ├── __init__.py
│   │
│   ├── module1_perception/           # Module 1: SfM & 2D Segmentation
│   │   ├── sfm_pipeline.py           # Tự động hóa COLMAP / trích xuất camera poses, intrinsics
│   │   └── segmenter.py              # SAM2 / LangSAM trích xuất 2D Mask theo text prompt/vật thể
│   │
│   ├── module2_scene_gs/             # Module 2: Global Coarse Scene Training
│   │   ├── scene_trainer.py          # Huấn luyện Scene-GS (20K iters theo chiến lược ROI-GS)
│   │   └── camera_scheduler.py       # Phân loại góc nhìn toàn cảnh vs góc nhìn cục bộ
│   │
│   ├── module3_roi_3d/               # Module 3: 3D ROI & Space Partitioning
│   │   ├── mask_to_3d.py             # Chiếu ngược Mask 2D + Depth/SfM keypoints lên không gian 3D
│   │   ├── aabb_generator.py         # Tính toán 3D Bounding Box (AABB/OBB) và Voxel Grid
│   │   └── view_selection.py         # Thuật toán chọn Camera tối ưu cho ROI (ActiveInitSplat / GP)
│   │
│   ├── module4_object_gof/           # Module 4: Object-GS Training & GOF Meshing
│   │   ├── object_trainer.py         # Huấn luyện Object-GS khởi tạo từ Scene-GS, densify trong ROI
│   │   ├── gs_composer.py            # Hợp nhất hạt Scene-GS và Object-GS theo tọa độ SfM
│   │   ├── opacity_field.py          # Định nghĩa trường độ đục GOF (Ray-Gaussian intersection, Min Opacity)
│   │   ├── regularization.py         # Depth Distortion Loss & Normal Consistency Loss
│   │   └── marching_tetrahedra.py    # Marching Tetrahedra + Binary Search tìm Level-set 0.5
│   │
│   ├── module5_postprocessing/       # Module 5: Mesh Decimation & Watertight Sewing
│   │   ├── mesh_cleaner.py           # Lọc floaters, tam giác dị hình, isolated vertices
│   │   ├── adaptive_decimator.py     # Gộp lưới Macro (thô) - Micro (mịn) theo Task ROI
│   │   └── watertight_sewer.py       # Ghép nối đường biên tiếp giáp (Boundary Stitching/Watertight)
│   │
│   └── module6_physics/              # Module 6: Physics Engine Integration & Topology Guard
│       ├── env_mujoco.py             # Môi trường mô phỏng MuJoCo / PyBullet
│       ├── collision_verifier.py     # Kiểm tra va chạm rác (False Positives) & bảo toàn khe rỗng
│       ├── topology_guard.py         # State Machine 5 trạng thái (Async rebuilder, Bounding volume check)
│       └── benchmarks.py             # Thu thập số liệu: Simulation FPS, Penetration Error, NaN Forces
│
├── scripts/                          # Entry points chạy pipeline thực thi
│   ├── run_stage1_perception.py      # Chạy Module 1 + 3 (SfM -> Segment -> 3D ROI)
│   ├── run_stage2_training.py        # Chạy Module 2 + 4 (Scene-GS -> Object-GS -> Composition)
│   ├── run_stage3_meshing.py         # Chạy Module 4 (GOF Extraction) + Module 5 (Hậu xử lý)
│   └── run_simulation.py             # Chạy Module 6 (Nạp Mesh vào Simulator kiểm thử)
│
├── tests/                            # Unit tests từng thành phần độc lập
│   ├── test_opacity_eval.py
│   ├── test_watertight.py
│   └── test_physics_swap.py
│
├── requirements.txt                  # Python dependencies (PyTorch, PyBullet, MuJoCo, Open3D, Trimesh, CGAL)
└── README.md                         # Hướng dẫn cài đặt, cấu hình pipeline và reproduce kết quả
```