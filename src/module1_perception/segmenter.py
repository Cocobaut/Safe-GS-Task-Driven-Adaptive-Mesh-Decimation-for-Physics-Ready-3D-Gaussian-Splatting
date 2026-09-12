import os
import cv2
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class RMBG2Segmenter:
    """
    Module phân vùng 2D sử dụng RMBG-2.0 (BiRefNet architecture)
    Tối ưu hóa pipeline với FP16 và GPU-accelerated postprocessing.
    """
    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "briaai/RMBG-2.0",
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.use_fp16 = (self.device == "cuda")
        print(f"[Loading] Đang tải RMBG-2.0 từ: {model_name_or_path} lên thiết bị [{self.device}] (FP16={self.use_fp16})...")
        
        # 1. Load model trực tiếp bằng float16 nếu dùng CUDA
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(model_name_or_path),
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

        self.input_size = (1024, 1024)
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict_single_image(self, orig_img: Image.Image) -> Dict:
        w, h = orig_img.size

        # Tiền xử lý tensor đưa vào GPU
        input_tensor = self.transform(orig_img).unsqueeze(0).to(self.device)
        if self.use_fp16:
            input_tensor = input_tensor.half()

        # Suy luận FP16
        with torch.amp.autocast(device_type="cuda", enabled=self.use_fp16):
            preds = self.model(input_tensor)[-1].sigmoid()

        # 2. Resize trực tiếp trên GPU bằng Bilinear nội suy của PyTorch
        # Tránh chuyển về CPU -> PIL Image -> Resize
        preds_orig = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=False)
        
        # 3. Nhị phân hóa ngay trên GPU trước khi kéo sang CPU
        binary_mask_tensor = (preds_orig[0, 0] > self.threshold).to(torch.uint8) * 255
        binary_mask = binary_mask_tensor.cpu().numpy()

        # Tính hộp bao 2D AABB
        coords = np.argwhere(binary_mask > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox_xyxy = [float(x_min), float(y_min), float(x_max), float(y_max)]
        else:
            bbox_xyxy = [0.0, 0.0, 0.0, 0.0]

        return {
            "width": w,
            "height": h,
            "mask": binary_mask,
            "bbox_xyxy": bbox_xyxy
        }

    def process_directory(
        self,
        images_dir: Union[str, Path],
        output_mask_dir: Union[str, Path],
        save_visualizations: bool = False,
        vis_output_dir: Optional[Union[str, Path]] = None
    ):
        images_dir = Path(images_dir)
        output_mask_dir = Path(output_mask_dir)
        output_mask_dir.mkdir(parents=True, exist_ok=True)

        if save_visualizations and vis_output_dir is not None:
            vis_output_dir = Path(vis_output_dir)
            vis_output_dir.mkdir(parents=True, exist_ok=True)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_extensions])

        if not image_files:
            print(f"[Error] Không có ảnh hợp lệ tại: {images_dir}")
            return

        all_metadata = {}
        # Tối ưu cờ nén PNG để giảm nghẽn CPU khi ghi ổ cứng
        fast_png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        print(f"[Loading] Bắt đầu RMBG-2.0 Segmentation trên {len(image_files)} ảnh...")
        for img_path in tqdm(image_files, desc="RMBG-2.0 Segmenting"):
            # Mở ảnh 1 lần duy nhất
            orig_pil = Image.open(img_path).convert("RGB")
            
            result = self.predict_single_image(orig_pil)

            mask_filename = f"{img_path.stem}.png"
            mask_save_path = output_mask_dir / mask_filename
            
            # Ghi mask ra đĩa với compression level thấp (nhanh hơn rất nhiều)
            cv2.imwrite(str(mask_save_path), result["mask"], fast_png_params)

            all_metadata[img_path.name] = {
                "mask_file": mask_filename,
                "width": result["width"],
                "height": result["height"],
                "bbox_xyxy": result["bbox_xyxy"],
                "detections": [{
                    "class_name": "foreground_object",
                    "bbox_xyxy": result["bbox_xyxy"]
                }]
            }

            # Tận dụng luôn ảnh gốc đã nạp, không đọc lại từ đĩa
            if save_visualizations and vis_output_dir is not None:
                img_np = np.array(orig_pil)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                color_mask = np.zeros_like(img_bgr)
                color_mask[:, :, 1] = result["mask"]  # Kênh Green
                overlay = cv2.addWeighted(img_bgr, 0.7, color_mask, 0.3, 0)
                cv2.imwrite(str(vis_output_dir / f"vis_{img_path.name}"), overlay)

        meta_path = output_mask_dir / "segmentation_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)

        print(f"[Done] Đã lưu {len(image_files)} masks tại: {output_mask_dir}")
        print(f"[Done] Metadata JSON lưu tại: {meta_path}")


if __name__ == "__main__":
    # Doc duong dan tu configs/base_scene.toml thay vi hard-code (thay doi
    # duong dan khi chuyen may chi can sua file config, khong sua code).
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open("configs/base_scene.toml", "rb") as f:
        _cfg = tomllib.load(f)
    _workspace_dir = Path(_cfg.get("workspace_dir", "data/workspace"))

    from src.common.config_loader import resolve_raw_image_dir

    raw_image_directionary = resolve_raw_image_dir(_cfg)
    mask_output_directionary = str(_workspace_dir / "segmentation")
    vis_output_directionary = str(_workspace_dir / "segmentation" / "vis")

    segmenter = RMBG2Segmenter(
        model_name_or_path="briaai/RMBG-2.0",
        threshold=0.5
    )

    if Path(raw_image_directionary).exists() and any(Path(raw_image_directionary).iterdir()):
        # Tắt save_visualizations=False nếu không thực sự cần thiết để tiết kiệm thêm một nửa thời gian I/O
        segmenter.process_directory(
            images_dir=raw_image_directionary,
            output_mask_dir=mask_output_directionary,
            save_visualizations=False,
            vis_output_dir=vis_output_directionary
        )
    else:
        print(f"Vui lòng đặt ảnh vào {raw_image_directionary} trước khi kiểm thử.")