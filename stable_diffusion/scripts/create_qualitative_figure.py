import csv
import re
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1. 路径配置
# ============================================================

RESULT_ROOT = Path(
    os.environ.get("RGTC_PATH", "")
)

PAIRWISE_CSV = RESULT_ROOT / "pairwise_win_analysis.csv"

FULL_DIR = RESULT_ROOT / "full"
TOME_DIR = RESULT_ROOT / "tome"
OURS_DIR = RESULT_ROOT / "reliability_guided"

SAVE_PATH = RESULT_ROOT / "visual_compare_r05_horizontal.png"


# ============================================================
# 2. 可视化样本配置
# ============================================================

# 横向展示的样本数量
NUM_SAMPLES = 12

# 每张小图尺寸
# 如果论文中觉得太宽，可以改成 128 或把 NUM_SAMPLES 改成 10
IMAGE_SIZE = 138

# 是否启用手动优先样本
USE_MANUAL_SELECTION = True

# 不适合论文展示的类别，自动排除
EXCLUDE_KEYWORDS = [
    "hatchet",          # 斧头
    "scabbard",         # 剑鞘
    "mortarboard",      # 学士帽
    "toilet_seat",      # 马桶座圈
    "toilet seat",
"crate",
"mailbox",
]

# 优先选择视觉效果更好的类别
# 只要文件名中包含这些关键词，就会优先被选中
PREFERRED_KEYWORDS = [
    "Brittany_spaniel",
    "golden_retriever",
    "Labrador_retriever",
    "beagle",
    "tabby",
    "tiger_cat",
    "Persian_cat",
    "speedboat",
    "fireboat",
    "sports_car",
    "racer",
    "airliner",

    "geyser",
    "hot_pot",
    "pizza",
    "espresso",
    "coral_reef",
    "volcano",
    "lakeside",
    "valley",
    "castle",
    "church",

]

# 手动优先展示样本
# 这里已经去掉了 hatchet / scabbard / mortarboard / toilet_seat
MANUAL_IMAGE_NAMES = [
    "class_0215_seed0_a_photo_of_a_Brittany_spaniel.png",
    "class_0814_seed0_a_photo_of_a_speedboat.png",
    "class_0926_seed0_a_photo_of_a_hot_pot.png",
    "class_0637_seed0_a_photo_of_a_mailbox.png",
    "class_0974_seed0_a_photo_of_a_geyser.png",
    "class_0519_seed1_a_photo_of_a_crate.png",
]


# ============================================================
# 3. 画布样式配置
# ============================================================

MARGIN = 24
LABEL_W = 115
HEADER_H = 65
CELL_GAP = 6

ROW_NAMES = ["Full", "ToMe-SD", "Reliability-Guided-SD"]

BG_COLOR = "white"
TEXT_COLOR = (0, 0, 0)
BORDER_COLOR = (40, 40, 40)


# ============================================================
# 4. 工具函数
# ============================================================

def load_font(size=20, bold=False):
    """
    加载字体。
    Windows 下优先使用 Arial，找不到则使用默认字体。
    """
    if bold:
        candidates = [
            os.environ.get("RGTC_PATH", ""),
            os.environ.get("RGTC_PATH", ""),
        ]
    else:
        candidates = [
            os.environ.get("RGTC_PATH", ""),
            os.environ.get("RGTC_PATH", ""),
        ]

    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)

    return ImageFont.load_default()


def read_pairwise_csv(csv_path):
    """
    读取 pairwise_win_analysis.csv。
    默认按 LPIPS 改善值和 MS-SSIM 改善值从大到小排序。
    """
    rows = []

    if not csv_path.exists():
        print(f"[WARN] Pairwise CSV not found: {csv_path}")
        return rows

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                row["lpips_diff_tome_minus_reliability_guided"] = float(
                    row["lpips_diff_tome_minus_reliability_guided"]
                )
                row["msssim_diff_reliability_guided_minus_tome"] = float(
                    row["msssim_diff_reliability_guided_minus_tome"]
                )
                rows.append(row)
            except Exception:
                continue

    rows = sorted(
        rows,
        key=lambda x: (
            x["lpips_diff_tome_minus_reliability_guided"],
            x["msssim_diff_reliability_guided_minus_tome"],
        ),
        reverse=True,
    )

    return rows


def image_exists_in_all_methods(image_name):
    """
    判断同名图片是否同时存在于 Full、ToMe、Reliability-Guided 三个目录。
    """
    return (
        (FULL_DIR / image_name).exists()
        and (TOME_DIR / image_name).exists()
        and (OURS_DIR / image_name).exists()
    )


def is_excluded_image(image_name):
    """
    判断图片是否属于不适合展示的类别。
    """
    name_lower = image_name.lower()

    for key in EXCLUDE_KEYWORDS:
        if key.lower() in name_lower:
            return True

    return False


def contains_preferred_keyword(image_name):
    """
    判断图片是否属于优先展示类别。
    """
    name_lower = image_name.lower()

    for key in PREFERRED_KEYWORDS:
        if key.lower() in name_lower:
            return True

    return False


def list_all_valid_images():
    """
    从 full 目录读取所有图片，并保证 ToMe / Reliability-Guided 中也存在同名图片。
    """
    all_images = sorted([p.name for p in FULL_DIR.glob("*.png")])

    valid = []

    for name in all_images:
        if is_excluded_image(name):
            continue

        if image_exists_in_all_methods(name):
            valid.append(name)

    return valid


def select_images():
    """
    选择需要可视化的图片。

    选择逻辑：
    1. 先使用手动指定样本；
    2. 再从 pairwise_win_analysis.csv 中选择优先类别，并且尽量选择 Reliability-Guided 改善明显的样本；
    3. 再选择 Reliability-Guided 在 LPIPS 和 MS-SSIM 上都优于 ToMe 的样本；
    4. 再按 LPIPS 改善补齐；
    5. 最后从全部有效图片中兜底补齐。
    """
    selected = []

    # ------------------------------------------------------------
    # 1. 手动样本优先
    # ------------------------------------------------------------
    if USE_MANUAL_SELECTION:
        for name in MANUAL_IMAGE_NAMES:
            if is_excluded_image(name):
                continue

            if image_exists_in_all_methods(name):
                selected.append(name)

            if len(selected) >= NUM_SAMPLES:
                return selected[:NUM_SAMPLES]

    rows = read_pairwise_csv(PAIRWISE_CSV)

    # ------------------------------------------------------------
    # 2. 优先类别 + Reliability-Guided 相对 ToMe 有改善
    # ------------------------------------------------------------
    for row in rows:
        name = row["image_name"]

        if name in selected:
            continue

        if is_excluded_image(name):
            continue

        if not contains_preferred_keyword(name):
            continue

        if row["lpips_diff_tome_minus_reliability_guided"] <= 0:
            continue

        if image_exists_in_all_methods(name):
            selected.append(name)

        if len(selected) >= NUM_SAMPLES:
            return selected[:NUM_SAMPLES]

    # ------------------------------------------------------------
    # 3. 优先类别，不强制要求指标改善
    # ------------------------------------------------------------
    all_valid_images = list_all_valid_images()

    for name in all_valid_images:
        if name in selected:
            continue

        if is_excluded_image(name):
            continue

        if contains_preferred_keyword(name):
            selected.append(name)

        if len(selected) >= NUM_SAMPLES:
            return selected[:NUM_SAMPLES]

    # ------------------------------------------------------------
    # 4. Reliability-Guided 在 LPIPS 和 MS-SSIM 上都优于 ToMe
    # ------------------------------------------------------------
    for row in rows:
        name = row["image_name"]

        if name in selected:
            continue

        if is_excluded_image(name):
            continue

        if row["lpips_diff_tome_minus_reliability_guided"] <= 0:
            continue

        if row["msssim_diff_reliability_guided_minus_tome"] <= 0:
            continue

        if image_exists_in_all_methods(name):
            selected.append(name)

        if len(selected) >= NUM_SAMPLES:
            return selected[:NUM_SAMPLES]

    # ------------------------------------------------------------
    # 5. 只按 LPIPS 改善补齐
    # ------------------------------------------------------------
    for row in rows:
        name = row["image_name"]

        if name in selected:
            continue

        if is_excluded_image(name):
            continue

        if row["lpips_diff_tome_minus_reliability_guided"] <= 0:
            continue

        if image_exists_in_all_methods(name):
            selected.append(name)

        if len(selected) >= NUM_SAMPLES:
            return selected[:NUM_SAMPLES]

    # ------------------------------------------------------------
    # 6. 最后兜底：从全部有效图片中补齐
    # ------------------------------------------------------------
    for name in all_valid_images:
        if name in selected:
            continue

        if is_excluded_image(name):
            continue

        selected.append(name)

        if len(selected) >= NUM_SAMPLES:
            return selected[:NUM_SAMPLES]

    return selected[:NUM_SAMPLES]


def extract_short_label(image_name):
    """
    从文件名中提取简洁类别名。
    例如：
    class_0215_seed0_a_photo_of_a_Brittany_spaniel.png
    -> Brittany spaniel
    """
    stem = Path(image_name).stem

    match = re.search(r"seed\d+_(.*)", stem)
    if match:
        prompt = match.group(1).replace("_", " ")
    else:
        prompt = stem.replace("_", " ")

    prompt = prompt.strip()

    remove_prefixes = [
        "a photo of an ",
        "a photo of a ",
        "a photo of the ",
        "a photo of ",
    ]

    label = prompt

    for prefix in remove_prefixes:
        if label.lower().startswith(prefix):
            label = label[len(prefix):]
            break

    return label


def draw_centered_text(draw, box, text, font, fill=TEXT_COLOR):
    """
    在指定区域中居中绘制单行文本。
    """
    x1, y1, x2, y2 = box

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = x1 + (x2 - x1 - tw) / 2
    y = y1 + (y2 - y1 - th) / 2

    draw.text((x, y), text, font=font, fill=fill)


def draw_wrapped_center_text(draw, box, text, font, max_chars=16, fill=TEXT_COLOR):
    """
    在指定区域内居中绘制多行文本。
    """
    x1, y1, x2, y2 = box

    lines = textwrap.wrap(text, width=max_chars)

    if len(lines) > 2:
        lines = lines[:2]
        lines[-1] = lines[-1] + "..."

    line_heights = []
    line_widths = []

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_h = sum(line_heights) + max(0, len(lines) - 1) * 3
    cur_y = y1 + (y2 - y1 - total_h) / 2

    for line, tw, th in zip(lines, line_widths, line_heights):
        x = x1 + (x2 - x1 - tw) / 2
        draw.text((x, cur_y), line, font=font, fill=fill)
        cur_y += th + 3


def open_and_resize(path, size):
    """
    打开图片并缩放到固定大小。
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img


# ============================================================
# 5. 主函数
# ============================================================

def main():
    for d in [FULL_DIR, TOME_DIR, OURS_DIR]:
        if not d.exists():
            raise FileNotFoundError(f"Folder not found: {d}")

    selected_names = select_images()

    if len(selected_names) == 0:
        raise RuntimeError("No valid images selected.")

    print("[INFO] Selected images:")
    for name in selected_names:
        print("  ", name)

    title_font = load_font(size=22, bold=True)
    row_font = load_font(size=22, bold=True)
    label_font = load_font(size=20, bold=False)

    num_cols = len(selected_names)
    num_rows = len(ROW_NAMES)

    canvas_w = (
        MARGIN * 2
        + LABEL_W
        + num_cols * IMAGE_SIZE
        + (num_cols - 1) * CELL_GAP
    )

    canvas_h = (
        MARGIN * 2
        + HEADER_H
        + num_rows * IMAGE_SIZE
        + (num_rows - 1) * CELL_GAP
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # ------------------------------------------------------------
    # 顶部类别标题
    # ------------------------------------------------------------
    start_x = MARGIN + LABEL_W
    start_y = MARGIN + HEADER_H

    for col_idx, image_name in enumerate(selected_names):
        x = start_x + col_idx * (IMAGE_SIZE + CELL_GAP)
        y = MARGIN

        label = extract_short_label(image_name)

        draw_wrapped_center_text(
            draw=draw,
            box=(x, y, x + IMAGE_SIZE, y + HEADER_H),
            text=label,
            font=label_font,
            max_chars=20,
        )

    # ------------------------------------------------------------
    # 左侧方法名称
    # ------------------------------------------------------------
    for row_idx, row_name in enumerate(ROW_NAMES):
        y = start_y + row_idx * (IMAGE_SIZE + CELL_GAP)

        draw_centered_text(
            draw=draw,
            box=(MARGIN, y, MARGIN + LABEL_W - CELL_GAP, y + IMAGE_SIZE),
            text=row_name,
            font=row_font,
        )

    # ------------------------------------------------------------
    # 绘制图片矩阵
    # ------------------------------------------------------------
    method_dirs = [FULL_DIR, TOME_DIR, OURS_DIR]

    for col_idx, image_name in enumerate(selected_names):
        for row_idx, method_dir in enumerate(method_dirs):
            img_path = method_dir / image_name

            img = open_and_resize(img_path, IMAGE_SIZE)

            x = start_x + col_idx * (IMAGE_SIZE + CELL_GAP)
            y = start_y + row_idx * (IMAGE_SIZE + CELL_GAP)

            canvas.paste(img, (x, y))

            draw.rectangle(
                [x, y, x + IMAGE_SIZE, y + IMAGE_SIZE],
                outline=BORDER_COLOR,
                width=2,
            )

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(SAVE_PATH, dpi=(300, 300))

    print(f"[DONE] Horizontal visualization saved to: {SAVE_PATH}")
    print(f"[INFO] Canvas size: {canvas_w} x {canvas_h}")


if __name__ == "__main__":
    main()