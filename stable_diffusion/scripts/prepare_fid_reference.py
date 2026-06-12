from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ============================================================
# 1. 路径配置
# ============================================================

# 原始平铺参考图目录
SRC_ROOT = Path(os.environ.get("RGTC_PATH", ""))

# 输出固定尺寸参考图目录
DST_ROOT = Path(os.environ.get("RGTC_PATH", ""))

# 统一尺寸，和 SD 生成图保持一致
TARGET_SIZE = 512

# 支持的图片格式
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 2. 图像处理函数
# ============================================================

def resize_center_crop(img: Image.Image, size: int = 512) -> Image.Image:
    """
    先按短边 resize 到 size，再中心裁剪成 size × size。
    这样不会直接强行拉伸变形。
    """
    img = img.convert("RGB")
    w, h = img.size

    # 计算缩放比例：短边对齐到 size
    scale = size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # resize
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size

    img = img.crop((left, top, right, bottom))
    return img


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in SRC_ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ]

    image_files = sorted(image_files)

    print(f"[INFO] Source images: {len(image_files)}")
    print(f"[INFO] Save to: {DST_ROOT}")

    for src_path in tqdm(image_files, desc="Resize reference images"):
        dst_path = DST_ROOT / src_path.with_suffix(".png").name

        if dst_path.exists():
            continue

        try:
            img = Image.open(src_path)
            img = resize_center_crop(img, TARGET_SIZE)
            img.save(dst_path)
        except Exception as e:
            print(f"[WARN] Failed: {src_path} | {e}")

    final_files = [
        p for p in DST_ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ]

    print(f"[DONE] Final resized reference images: {len(final_files)}")
    print(f"[DONE] Output folder: {DST_ROOT}")


if __name__ == "__main__":
    main()