import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.module2_scene_gs.scene_branch.scene_trainer import SceneGSTrainer


def main():
    print("=" * 60)
    print(">>> BẮT ĐẦU CHẠY MODULE 2: NHÁNH SCENE-GS TRAINING <<<")
    print("=" * 60)

    config_path = "configs/base_scene.toml"
    trainer = SceneGSTrainer(config_path=config_path)
    trainer.train()

    print("=" * 60)
    print(">>> HOÀN THÀNH HUẤN LUYỆN SCENE-GS THÀNH CÔNG <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()