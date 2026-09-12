# Hướng Dẫn Tải Mã Nguồn Các Mô Hình Baseline (3DGS Surface Reconstruction)

Tài liệu tổng hợp liên kết và hướng dẫn dòng lệnh tải toàn bộ mã nguồn của 8 mô hình tái cấu trúc bề mặt từ 3D Gaussian Splatting phục vụ cho việc cài đặt môi trường, thử nghiệm và đánh giá thực nghiệm.

**Lưu ý:** Hầu hết các kho mã nguồn 3DGS đều chứa các module phụ thuộc con (CUDA rasterizer, `diff-gaussian-rasterization`, `simple-knn`), vui lòng luôn sử dụng cờ `--recursive` khi thực hiện `git clone`.

**Lưu ý:** Trong folder Based_Model có chứa các folder con về model, khi **clone về hãy chọn đúng folder mà clone**

### 1. SuGaR: Surface-Aligned Gaussian Splatting
* **Tên mô hình:** SuGaR (Surface-Aligned Gaussian Splatting)
* **Link tải / Trang chủ:** https://github.com/Anttwo/SuGaR
* **Lệnh tải:**
```bash
git clone --recursive [https://github.com/Anttwo/SuGaR.git](https://github.com/Anttwo/SuGaR.git)

```

### 2. GOF: Gaussian Opacity Fields

* **Tên mô hình:** GOF (Gaussian Opacity Fields)
* **Link tải / Trang chủ:** https://github.com/autonomousvision/gaussian-opacity-fields
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/autonomousvision/gaussian-opacity-fields.git](https://github.com/autonomousvision/gaussian-opacity-fields.git)

```

### 3. GSurf: 3D Reconstruction via Signed Distance Fields

* **Tên mô hình:** GSurf (3D Reconstruction via Signed Distance Fields with Direct Gaussian Supervision)
* **Link tải / Trang chủ:** https://github.com/xubaixinxbx/Gsurf
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/xubaixinxbx/Gsurf.git](https://github.com/xubaixinxbx/Gsurf.git)

```

### 4. GSDF: 3DGS Meets SDF

* **Tên mô hình:** GSDF (3DGS Meets SDF for Improved Neural Rendering and Reconstruction)
* **Link tải / Trang chủ:** https://github.com/city-super/GSDF
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/city-super/GSDF.git](https://github.com/city-super/GSDF.git)

```

### 5. 2D Gaussian Splatting (2DGS)

* **Tên mô hình:** 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
* **Link tải / Trang chủ:** https://surfsplatting.github.io/ (GitHub: https://github.com/hbb1/2d-gaussian-splatting)
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/hbb1/2d-gaussian-splatting.git](https://github.com/hbb1/2d-gaussian-splatting.git)

```

### 6. GauStudio (3DGSR)

* **Tên mô hình:** GauStudio / 3DGSR (Implicit Surface Reconstruction with 3DGS)
* **Link tải / Trang chủ:** https://github.com/GAP-LAB-CUHK-SZ/gaustudio
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/GAP-LAB-CUHK-SZ/gaustudio.git](https://github.com/GAP-LAB-CUHK-SZ/gaustudio.git)

```

### 7. RaDe-GS: Rasterizing Depth in Gaussian Splatting

* **Tên mô hình:** RaDe-GS (Rasterizing Depth in Gaussian Splatting)
* **Link tải / Trang chủ:** https://baowenz.github.io/radegs/ (GitHub: https://github.com/BaowenZ/radegs)
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/BaowenZ/radegs.git](https://github.com/BaowenZ/radegs.git)

```

### 8. MILo: Mesh-In-the-Loop Gaussian Splatting

* **Tên mô hình:** MILo (Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction)
* **Link tải / Trang chủ:** https://anttwo.github.io/milo/ (GitHub: https://github.com/Anttwo/MILo)
* **Lệnh tải:**

```bash
git clone --recursive [https://github.com/Anttwo/MILo.git](https://github.com/Anttwo/MILo.git)

```