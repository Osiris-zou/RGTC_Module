import os
import sys
import csv
import time
import importlib.util
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer


# ============================================================
# 1. 需要你直接修改的参数
# ============================================================

# Stable Diffusion 权重路径
CKPT_PATH = os.environ.get("RGTC_PATH", "")

# tokenizer 文件夹，里面应包含 vocab.json 和 merges.txt
TOKENIZER_DIR = os.environ.get("RGTC_PATH", "")

# ImageNet prompt 文件，每行一个 prompt，例如：a photo of a goldfish
PROMPT_FILE = os.environ.get("RGTC_PATH", "")

# 输出目录
OUT_DIR = os.environ.get("RGTC_PATH", "")

# 使用设备
DEVICE = "cuda"

# 生成参数
N_INFERENCE_STEPS = 50
CFG_SCALE = 7.5
SAMPLER_NAME = "ddpm"   # 如果你的 pipeline 支持 plms/ddim，可以改；不确定就保持 ddpm

# 每个类别生成几张图。ToMeSD 论文是每类 2 张。
SEEDS = [0, 1]

# 调试时可以只跑前 N 个 prompt。正式实验改成 None。
MAX_PROMPTS = None
# MAX_PROMPTS = 10

# token merging 参数
MERGE_RATIO = 0.5
OURS_BETA = 0.005

MERGE_MAX_DOWNSAMPLE = 1
MERGE_SX = 2
MERGE_SY = 2
MERGE_USE_RAND = True

# 是否跳过已经生成过的图片
SKIP_EXISTING = True


# ============================================================
# 2. 路径与本地模块导入
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SD_DIR = PROJECT_ROOT / "sd"
TOMESD_DIR = PROJECT_ROOT / "tomesd"

if not SD_DIR.exists():
    raise FileNotFoundError(f"Cannot find sd folder: {SD_DIR}")

if not TOMESD_DIR.exists():
    raise FileNotFoundError(f"Cannot find tomesd folder: {TOMESD_DIR}")

for p in [PROJECT_ROOT, SD_DIR, TOMESD_DIR]:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_py_module(module_name: str, file_path: Path):
    """
    按文件路径加载本地 Python 模块，避免 PyCharm 标红或相对导入问题。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find python file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load_py_module(
    module_name="sd_pipeline",
    file_path=SD_DIR / "pipeline.py",
)

model_loader = load_py_module(
    module_name="sd_model_loader",
    file_path=SD_DIR / "model_loader.py",
)

preload_models_from_standard_weights = model_loader.preload_models_from_standard_weights


# ============================================================
# 3. 实验方法配置
# ============================================================

METHODS = [
    {
        "name": "full",
        "merge_method": "full",
        "ratio": 0.0,
        "beta": 0.0,
    },
    {
        "name": "tome",
        "merge_method": "tome",
        "ratio": MERGE_RATIO,
        "beta": 0.0,
    },
    {
        "name": "reliability_guided",
        "merge_method": "reliability_guided",
        "ratio": MERGE_RATIO,
        "beta": OURS_BETA,
    },
]


# ============================================================
# 4. 工具函数
# ============================================================

def read_prompts(prompt_file: str, max_prompts=None):
    """
    读取 ImageNet 类别 prompt 文件。
    每一行应是一个 prompt。
    """
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompts = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                prompts.append(text)

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if len(prompts) == 0:
        raise ValueError("No prompts found. Please check imagenet_prompts.txt.")

    return prompts


def safe_name(text: str, max_len: int = 80):
    """
    把 prompt 转成安全文件名片段。
    """
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            keep.append(ch)
        elif ch == " ":
            keep.append("_")
    name = "".join(keep)
    return name[:max_len]


def save_image(np_img, save_path: Path):
    """
    保存 pipeline.generate 返回的 numpy 图像。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np_img)
    img.save(save_path)


def init_csv(csv_path: Path):
    """
    初始化 generation_log.csv。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and SKIP_EXISTING:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "class_id",
            "prompt",
            "seed",
            "ratio",
            "beta",
            "image_path",
            "time_sec",
            "peak_mem_gb",
        ])


def append_csv(csv_path: Path, row: dict):
    """
    追加一条生成记录。
    """
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["method"],
            row["class_id"],
            row["prompt"],
            row["seed"],
            row["ratio"],
            row["beta"],
            row["image_path"],
            row["time_sec"],
            row["peak_mem_gb"],
        ])


def generate_one_image(
    models,
    tokenizer,
    device,
    prompt: str,
    seed: int,
    method_cfg: dict,
):
    """
    生成单张图片，并统计时间和显存。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()

    image = pipeline.generate(
        prompt=prompt,
        uncond_prompt="",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=CFG_SCALE,
        sampler_name=SAMPLER_NAME,
        n_inference_steps=N_INFERENCE_STEPS,
        models=models,
        seed=seed,
        device=device,
        idle_device=None,
        tokenizer=tokenizer,

        merge_method=method_cfg["merge_method"],
        merge_ratio=method_cfg["ratio"],
        merge_beta=method_cfg["beta"],
        merge_max_downsample=MERGE_MAX_DOWNSAMPLE,
        merge_sx=MERGE_SX,
        merge_sy=MERGE_SY,
        merge_use_rand=MERGE_USE_RAND,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()
    time_sec = end - start

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    else:
        peak_mem_gb = 0.0

    return image, time_sec, peak_mem_gb


# ============================================================
# 5. 主函数
# ============================================================

def main():
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "generation_log.csv"
    init_csv(csv_path)

    print("[INFO] Reading prompts...")
    prompts = read_prompts(PROMPT_FILE, max_prompts=MAX_PROMPTS)
    print(f"[INFO] Number of prompts: {len(prompts)}")
    print(f"[INFO] Seeds per prompt: {SEEDS}")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading tokenizer...")
    tokenizer = CLIPTokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        merges_file=os.path.join(TOKENIZER_DIR, "merges.txt"),
    )

    print("[INFO] Loading Stable Diffusion models...")
    models = preload_models_from_standard_weights(CKPT_PATH, device)

    total_jobs = len(METHODS) * len(prompts) * len(SEEDS)
    print(f"[INFO] Total generation jobs: {total_jobs}")

    for method_cfg in METHODS:
        method_name = method_cfg["name"]

        print("\n" + "=" * 80)
        print(
            f"[METHOD] {method_name} | "
            f"merge_method={method_cfg['merge_method']} | "
            f"ratio={method_cfg['ratio']} | beta={method_cfg['beta']}"
        )
        print("=" * 80)

        method_dir = out_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(
            enumerate(prompts),
            total=len(prompts),
            desc=f"Generating {method_name}",
        )

        for class_id, prompt in progress:
            prompt_tag = safe_name(prompt)

            for seed in SEEDS:
                file_name = f"class_{class_id:04d}_seed{seed}_{prompt_tag}.png"
                save_path = method_dir / file_name

                if SKIP_EXISTING and save_path.exists():
                    continue

                try:
                    image, time_sec, peak_mem_gb = generate_one_image(
                        models=models,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        seed=seed,
                        method_cfg=method_cfg,
                    )

                    save_image(image, save_path)

                    append_csv(csv_path, {
                        "method": method_name,
                        "class_id": class_id,
                        "prompt": prompt,
                        "seed": seed,
                        "ratio": method_cfg["ratio"],
                        "beta": method_cfg["beta"],
                        "image_path": str(save_path),
                        "time_sec": f"{time_sec:.4f}",
                        "peak_mem_gb": f"{peak_mem_gb:.4f}",
                    })

                    progress.set_postfix({
                        "class": class_id,
                        "seed": seed,
                        "time": f"{time_sec:.2f}s",
                        "mem": f"{peak_mem_gb:.2f}GB",
                    })

                except Exception as e:
                    print(
                        f"\n[ERROR] method={method_name}, "
                        f"class_id={class_id}, seed={seed}, prompt={prompt}"
                    )
                    print(f"[ERROR] {repr(e)}")

                    append_csv(csv_path, {
                        "method": method_name,
                        "class_id": class_id,
                        "prompt": prompt,
                        "seed": seed,
                        "ratio": method_cfg["ratio"],
                        "beta": method_cfg["beta"],
                        "image_path": "ERROR",
                        "time_sec": "ERROR",
                        "peak_mem_gb": "ERROR",
                    })

    print("\n[DONE] Image generation finished.")
    print(f"[DONE] Results saved to: {out_root}")
    print(f"[DONE] Log saved to: {csv_path}")


if __name__ == "__main__":
    main()