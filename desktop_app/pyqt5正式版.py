# -*- coding: utf-8 -*-
import os

# ---- Determinism-related env flags (set before importing torch) ----
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
_yolo_config_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ultralytics_config")
try:
    os.makedirs(_yolo_config_dir, exist_ok=True)
except Exception:
    import tempfile
    _yolo_config_dir = os.path.join(tempfile.gettempdir(), "temple_agent_ultralytics_config")
    os.makedirs(_yolo_config_dir, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _yolo_config_dir)

import sys
import random
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QFrame, QTextEdit, QComboBox, QMessageBox, QGridLayout,
    QProgressBar, QLineEdit, QGroupBox, QSplitter
)

import timm
from torchvision import transforms, models
from torchvision.models import (
    Swin_V2_T_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
)

# 尝试导入 YOLO，如未安装则给出提示
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[警告] 未安装 ultralytics，YOLO检测功能将不可用。请运行: pip install ultralytics")

# =========================
# 路径配置 - 各任务独立配置
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 原有分类任务路径配置
MODEL_PATHS = {
    "塌寿三分类": os.path.join(BASE_DIR, "models", "tashou_best.pth"),
    "屋顶四分类": os.path.join(BASE_DIR, "models", "roof_resnet34_best.pth"),
    "开间分类": os.path.join(BASE_DIR, "models", "kaijian_best_swinv2.pth"),
    "瓦片分类": os.path.join(BASE_DIR, "models", "tile_best.pt"),
    "YOLO检测+分类": "",  # 留空，由下方独立配置区管理
    "屋脊装饰识别": os.path.join(BASE_DIR, "models", "roof_ridge_ornament_best.pt"),
    "建筑主体区域识别": os.path.join(BASE_DIR, "models", "body_yolo_best.pt"),
    "建筑屋顶区域识别": os.path.join(BASE_DIR, "models", "roof_yolo_best.pt"),
}

TASK_LABELS = {
    "塌寿三分类": ["guta", "pingta", "shuangta"],
    "屋顶四分类": ["duanyanshengjiankou", "jiasichui", "putong", "sanchuanji"],
    "开间分类": ["1kaijian", "3kaijian", "5kaijian", "7kaijian"],
    "瓦片分类": ["banwa", "tongwa"],
    "YOLO检测+分类": ["target"],  # 检测目标类别，由下方配置决定
    "屋脊装饰识别": ["龙", "珠", "塔", "瓶", "人物"],
    "建筑主体区域识别": ["建筑主体区域"],
    "建筑屋顶区域识别": ["建筑屋顶区域"],
}

REGION_YOLO_CONFIGS = {
    "建筑主体区域识别": {
        "yolo_model_path": MODEL_PATHS["建筑主体区域识别"],
        "label": "建筑主体区域",
        "conf_threshold": 0.25,
        "iou_threshold": 0.70,
        "max_det": 20,
        "box_color": (14, 165, 233),
        "text_color": (255, 255, 255),
    },
    "建筑屋顶区域识别": {
        "yolo_model_path": MODEL_PATHS["建筑屋顶区域识别"],
        "label": "建筑屋顶区域",
        "conf_threshold": 0.25,
        "iou_threshold": 0.70,
        "max_det": 20,
        "box_color": (34, 197, 94),
        "text_color": (255, 255, 255),
    },
}

# ============================================================================
# 屋顶四分类任务 - YOLO目标检测前序操作配置（新增独立配置区）
# ============================================================================
ROOF_YOLO_CONFIG = {
    # 主YOLO检测模型路径（用于屋顶检测）
    "yolo_model_path": os.path.join(BASE_DIR, "models", "roof_yolo_best.pt"),
    # 必填：例如 r"models\roof_yolo_best.pt"

    # 备用YOLO检测模型路径（用于建筑主体检测）
    # 当主YOLO未检测出屋顶目标时，先用此模型检测建筑主体，再把主体裁剪图交还给主YOLO二次检测
    "backup_yolo_model_path": os.path.join(BASE_DIR, "models", "body_yolo_best.pt"),
    # 例如 r"models\body_yolo_best.pt"

    # 主YOLO检测参数（尽量贴近你原始脚本）
    "conf_threshold": 0.25,   # 主模型检测置信度阈值
    "iou_threshold": 0.70,    # 主模型NMS IOU阈值（与原始脚本一致）
    "max_det": 10,            # 主模型最大检测数量（屋顶通常一个足够）
    "target_class": 0,        # 仅在 use_target_class_filter=True 时生效
    "use_target_class_filter": False,   # 默认关闭类别过滤，贴近原始脚本“只要检测到框就参与排序”
    "box_select_mode": "largest_area",  # "largest_area"（原始脚本）或 "highest_conf"

    # 备用YOLO检测参数（建筑主体检测）
    "backup_conf_threshold": 0.25,
    "backup_iou_threshold": 0.70,
    "backup_max_det": 10,
    "backup_target_class": 0,
    "backup_use_target_class_filter": False,
    "backup_box_select_mode": "largest_area",

    # 检测框处理参数
    "min_size": 0,  # 0 表示不额外限制检测框尺寸；更贴近原始脚本

    # 可视化参数（最终只显示主YOLO检测框）
    "draw_boxes": True,
    "box_color": (0, 255, 0),
    "text_color": (255, 255, 255),
}

# ============================================================================
# 中英文类别名称映射配置区（用户根据实际类别填写）
# ============================================================================
CLASS_NAME_MAPPING = {
    # 塌寿三分类
    "guta": "孤塌",
    "pingta": "透塌",
    "shuangta": "双塌",

    # 屋顶四分类
    "duanyanshengjiankou": "断檐升箭口",
    "jiasichui": "假四垂",
    "putong": "普通",
    "sanchuanji": "三川脊",

    # 开间分类
    "1kaijian": "一开间",
    "3kaijian": "三开间",
    "5kaijian": "五开间",
    "7kaijian": "七开间",

    # 瓦片分类
    "banwa": "板瓦",
    "tongwa": "筒瓦",

    # YOLO检测+分类任务
    "class_0": "类别0",
    "class_1": "类别1",
    "class_2": "类别2",
    "class_3": "类别3",

    # 屋脊装饰实例分割任务
    "龙": "龙",
    "珠": "珠",
    "塔": "塔",
    "瓶": "瓶",
    "人物": "人物",
}

# 默认中文名称（当映射中找不到时显示）
DEFAULT_CHINESE_NAME = "未知类别"

# ============================================================================
# YOLO检测+分类任务 - 独立配置区（用户必须填写以下配置）
# ============================================================================
YOLO_DETECT_CLS_CONFIG = {
    # ========== 第一部分：YOLO检测模型路径 ==========
    "yolo_model_path": os.path.join(BASE_DIR, "models", "roof_yolo_best.pt"),

    # ========== 第二部分：分类模型独立配置 ==========
    "cls_model_path": os.path.join(BASE_DIR, "models", "roof_resnet34_best.pth"),
    "cls_model_arch": "resnet34",
    "cls_num_classes": 4,
    "cls_labels": [
        "duanyanshengjiankou",
        "jiasichui",
        "putong",
        "sanchuanji",
    ],
    "cls_input_size": 224,
    "cls_normalize": True,
    "cls_mean": [0.485, 0.456, 0.406],
    "cls_std": [0.229, 0.224, 0.225],
    "cls_pad_bg_color": (248, 250, 252),
    "cls_use_tta": False,
    "cls_preprocess_mode": "roof_resize",
    "cls_tta_mode": "none",

    # ========== 第三部分：YOLO检测参数配置 ==========
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "max_det": 50,
    "draw_boxes": True,
    "box_color": (255, 0, 0),
    "text_color": (255, 255, 255),
}

# ============================================================================
# 屋脊装饰识别任务 - YOLO实例分割配置区
# ============================================================================
ROOF_RIDGE_ORNAMENT_CONFIG = {
    "yolo_model_path": os.path.join(BASE_DIR, "models", "roof_ridge_ornament_best.pt"),
    "labels": ["龙", "珠", "塔", "瓶", "人物"],
    "imgsz": 960,
    "conf_threshold": 0.25,
    "iou_threshold": 0.70,
    "max_det": 100,
    "use_roof_pre_detection": True,
    "roof_yolo_model_path": os.path.join(BASE_DIR, "models", "roof_yolo_best.pt"),
    "roof_conf_threshold": 0.25,
    "roof_iou_threshold": 0.70,
    "roof_max_det": 20,
    "roof_box_select_mode": "largest_area",
    "roof_padding_ratio": 0.06,
    "roof_padding_px": 12,
    "draw_roof_box": True,
    "roof_box_color": (34, 197, 94),
    "retina_masks": True,
    "draw_masks": True,
    "draw_boxes": True,
    "mask_alpha": 90,
    "box_width": 3,
    "text_color": (255, 255, 255),
    "class_colors": {
        "龙": (37, 99, 235),
        "珠": (20, 184, 166),
        "塔": (245, 158, 11),
        "瓶": (16, 185, 129),
        "人物": (124, 58, 237),
    },
}

# ============================================================================
# 各单任务推理配置区（按你后面四个独立 py 程序的处理方式整合）
# ============================================================================
TASHOU_INFER_CONFIG = {
    "seed": 3407,
    "img_size": 256,
    "resize_scale": 1.14,
    "print_low_conf": True,
    "low_conf_th": 0.55,
}

ROOF_CLASSIFY_CONFIG = {
    "input_size": 224,
    "preprocess_mode": "roof_resize",  # 对应独立屋顶分类脚本：Resize((224,224))
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

KAIJIAN_INFER_CONFIG = {
    "img_size": 224,
    "class_names": ["1kaijian", "3kaijian", "5kaijian", "7kaijian"],
    "resize_mode": "resize",      # "resize" or "resize320_cc"
    "ac": False,
    "g": 1.0,
    "sh": 1.0,
    "tta": "flip2",               # "none" / "flip2" / "flip4"
}

TILE_INFER_CONFIG = {
    "image_size": 224,
    "batch_size": 32,
    "masks_dir": r"",
    "use_mask_crop": False,
    "mask_pad": 12,
}


# =========================
# 工具函数
# =========================
def set_global_determinism(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def center_crop_with_padding(img: Image.Image, size: int = 224, bg_color=(248, 250, 252)) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), bg_color)
    paste_x = (size - nw) // 2
    paste_y = (size - nh) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas


class Gamma:
    def __init__(self, gamma=1.0):
        self.gamma = float(gamma)

    def __call__(self, img):
        if abs(self.gamma - 1.0) < 1e-6:
            return img
        return TF.adjust_gamma(img, gamma=self.gamma, gain=1.0)


class Sharpness:
    def __init__(self, factor=1.0):
        self.factor = float(factor)

    def __call__(self, img):
        if abs(self.factor - 1.0) < 1e-6:
            return img
        return TF.adjust_sharpness(img, sharpness_factor=self.factor)


class AutoContrast:
    def __call__(self, img):
        return TF.autocontrast(img)


def build_kaijian_transform(mean, std, cfg):
    img_size = int(cfg.get("img_size", 224))
    tfm = []
    if cfg.get("resize_mode", "resize") == "resize":
        tfm += [transforms.Resize((img_size, img_size))]
    elif cfg.get("resize_mode") == "resize320_cc":
        tfm += [transforms.Resize(320), transforms.CenterCrop(img_size)]
    else:
        raise ValueError(f"未知开间 resize_mode: {cfg.get('resize_mode')}")

    if cfg.get("ac", False):
        tfm += [AutoContrast()]
    if abs(float(cfg.get("g", 1.0)) - 1.0) > 1e-6:
        tfm += [Gamma(cfg.get("g", 1.0))]
    if abs(float(cfg.get("sh", 1.0)) - 1.0) > 1e-6:
        tfm += [Sharpness(cfg.get("sh", 1.0))]

    tfm += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tfm)


def build_tashou_eval_transform(mean, std, cfg):
    img_size = int(cfg.get("img_size", 256))
    resize_scale = float(cfg.get("resize_scale", 1.14))
    return transforms.Compose([
        transforms.Resize(int(img_size * resize_scale)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGB":
        img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def pretty_exception() -> str:
    return traceback.format_exc()


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model_state", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        tensor_like = any(isinstance(v, torch.Tensor) for v in ckpt.values())
        if tensor_like:
            return ckpt
    raise ValueError("未能从 checkpoint 中解析出 state_dict。")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_sd[nk] = v
    return new_sd


def tensor_probs_to_text(labels: List[str], probs: torch.Tensor) -> str:
    lines = []
    order = torch.argsort(probs, descending=True)
    for idx in order.tolist():
        name = labels[idx] if idx < len(labels) else f"class_{idx}"
        lines.append(f"{name}: {probs[idx].item() * 100:.2f}%")
    return "\n".join(lines)


def parse_inv_map_to_labels(inv_map: Any) -> List[str]:
    if isinstance(inv_map, list):
        return [str(x) for x in inv_map]
    if isinstance(inv_map, dict):
        try:
            keys_int = sorted(int(k) for k in inv_map.keys())
            return [str(inv_map[k] if k in inv_map else inv_map[str(k)]) for k in keys_int]
        except Exception:
            try:
                return [str(inv_map[k]) for k in sorted(inv_map.keys())]
            except Exception:
                return [str(v) for v in inv_map.values()]
    raise ValueError("无法从 inv_map 解析类别名。")


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def ensure_rgb(img_bgr: np.ndarray):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def apply_mask_and_crop(img_rgb: np.ndarray, mask: np.ndarray, pad: int = 10):
    if mask is None:
        return img_rgb

    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return img_rgb

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    h, w = mask.shape[:2]
    y1 = max(0, y1 - pad)
    y2 = min(h - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w - 1, x2 + pad)
    return img_rgb[y1:y2 + 1, x1:x2 + 1].copy()


def resize_pad_to_square(img_rgb: np.ndarray, size: int = 224):
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh = max(1, nh)
    nw = max(1, nw)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# =========================
# 新增：中英文类别名称转换工具函数
# =========================
def get_chinese_class_name(english_name: str) -> str:
    """
    将英文类别名转换为中文显示名称
    """
    return CLASS_NAME_MAPPING.get(english_name, english_name)


def get_english_class_name(chinese_name: str) -> str:
    """
    将中文类别名转换回英文（反向查找）
    """
    for eng, chn in CLASS_NAME_MAPPING.items():
        if chn == chinese_name:
            return eng
    return chinese_name


def convert_labels_to_chinese(labels: List[str]) -> List[str]:
    """
    批量转换英文类别名列表为中文
    """
    return [get_chinese_class_name(label) for label in labels]


# =========================
# 训练同构：瓦片分类 TileNet
# =========================
class TileNet(nn.Module):
    def __init__(self, backbone_name: str, n_tile: int):
        super().__init__()
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.backbone_name = "torchvision_resnet34"
        self.head_tile = nn.Linear(self.feat_dim, n_tile)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_tile(feat)


# =========================
# 模型包装 - 新增YOLO支持
# =========================
@dataclass
class LoadedModel:
    task_name: str
    model: nn.Module
    labels: List[str]
    input_size: int
    device: str
    extra_info: str = ""
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    use_kaijian_tta: bool = False
    tta_mode: str = "none"
    preprocess_mode: str = "pad_square"
    preprocess_cfg: Optional[Dict[str, Any]] = None
    pad_bg_color: Tuple[int, int, int] = (248, 250, 252)
    normalize: bool = True
    masks_dir: Optional[str] = None
    use_mask_crop: bool = False
    yolo_model: Optional[Any] = None
    backup_yolo_model: Optional[Any] = None
    yolo_config: Optional[Dict] = None
    cls_model: Optional['LoadedModel'] = None


class InferenceEngine:
    def __init__(self, device: Optional[str] = None):
        set_global_determinism(TASHOU_INFER_CONFIG.get("seed", 3407))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: Dict[str, LoadedModel] = {}

    def get_transform(self, loaded: LoadedModel):
        mode = getattr(loaded, "preprocess_mode", "pad_square")
        cfg = loaded.preprocess_cfg or {}
        mean = loaded.mean if loaded.mean is not None else [0.485, 0.456, 0.406]
        std = loaded.std if loaded.std is not None else [0.229, 0.224, 0.225]

        if mode == "roof_resize":
            ops = [transforms.Resize((loaded.input_size, loaded.input_size)), transforms.ToTensor()]
            if loaded.normalize:
                ops.append(transforms.Normalize(mean=mean, std=std))
            return transforms.Compose(ops)

        if mode == "tashou_eval":
            return build_tashou_eval_transform(mean, std, cfg)

        if mode == "kaijian_cfg":
            kcfg = dict(cfg)
            kcfg.setdefault("img_size", loaded.input_size)
            return build_kaijian_transform(mean, std, kcfg)

        ops = [
            transforms.Lambda(lambda im: center_crop_with_padding(im, loaded.input_size, bg_color=loaded.pad_bg_color)),
            transforms.ToTensor(),
        ]
        if loaded.normalize:
            ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)

    @torch.no_grad()
    def _forward_with_tta(self, model, x, tta_mode: str = "none"):
        tta_mode = (tta_mode or "none").lower()
        if tta_mode == "none":
            return model(x)
        if tta_mode == "flip2":
            return (model(x) + model(torch.flip(x, dims=[3]))) / 2.0
        if tta_mode == "flip4":
            logits = model(x)
            logits += model(torch.flip(x, dims=[3]))
            logits += model(torch.flip(x, dims=[2]))
            logits += model(torch.flip(x, dims=[2, 3]))
            return logits / 4.0
        raise ValueError(f"未知 TTA 模式: {tta_mode}")

    def _predict_single_classification(self, loaded: LoadedModel, img: Image.Image):
        transform = self.get_transform(loaded)
        x = transform(img).unsqueeze(0).to(loaded.device)
        loaded.model.eval()
        with torch.no_grad():
            tta_mode = loaded.tta_mode if loaded.tta_mode else ("flip2" if loaded.use_kaijian_tta else "none")
            logits = self._forward_with_tta(loaded.model, x, tta_mode)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)[0].detach().cpu()
        pred_idx = int(torch.argmax(probs).item())
        pred_name = loaded.labels[pred_idx] if pred_idx < len(loaded.labels) else f"class_{pred_idx}"
        conf = float(probs[pred_idx].item())
        topk = [
            (loaded.labels[i] if i < len(loaded.labels) else f"class_{i}", float(probs[i].item()))
            for i in torch.argsort(probs, descending=True).tolist()
        ]
        chinese_pred = get_chinese_class_name(pred_name)
        detail = self._format_detail_text(pred_name, chinese_pred, conf, topk)
        return pred_name, conf, topk, detail, None, chinese_pred

    def _prepare_tile_tensor(self, image_path: str, loaded: LoadedModel) -> torch.Tensor:
        cfg = loaded.preprocess_cfg or {}
        image_size = int(cfg.get("image_size", loaded.input_size))
        mask_pad = int(cfg.get("mask_pad", 12))
        masks_dir = cfg.get("masks_dir", "") or loaded.masks_dir
        use_mask_crop = bool(cfg.get("use_mask_crop", False) or loaded.use_mask_crop)
        img_path = Path(image_path)
        img_bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            img_rgb = ensure_rgb(img_bgr)
        if masks_dir and use_mask_crop:
            mask_path = Path(masks_dir) / f"{img_path.stem}.png"
            if mask_path.exists():
                mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    img_rgb = apply_mask_and_crop(img_rgb, mask, pad=mask_pad)
        img_rgb = resize_pad_to_square(img_rgb, size=image_size)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return x.unsqueeze(0).to(loaded.device)

    def _predict_tile(self, loaded: LoadedModel, image_path: str):
        x = self._prepare_tile_tensor(image_path, loaded)
        loaded.model.eval()
        with torch.no_grad():
            logits = loaded.model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)[0].detach().cpu()
        pred_idx = int(torch.argmax(probs).item())
        pred_name = loaded.labels[pred_idx] if pred_idx < len(loaded.labels) else f"class_{pred_idx}"
        conf = float(probs[pred_idx].item())
        topk = [
            (loaded.labels[i] if i < len(loaded.labels) else f"class_{i}", float(probs[i].item()))
            for i in torch.argsort(probs, descending=True).tolist()
        ]
        chinese_pred = get_chinese_class_name(pred_name)
        detail = self._format_detail_text(pred_name, chinese_pred, conf, topk)
        return pred_name, conf, topk, detail, None, chinese_pred

    def predict(self, task_name: str, image_path: str) -> Tuple[
        str, float, List[Tuple[str, float]], str, Optional[Image.Image], str]:
        """
        返回: (预测类别(英文), 置信度, TopK列表(英文), 详情文本(中文), 可视化图片, 中文预测名)
        """
        loaded = self.load_model(task_name)
        if task_name == "YOLO检测+分类":
            return self._predict_yolo_detect_cls(loaded, image_path)
        if task_name == "屋脊装饰识别":
            return self._predict_roof_ridge_ornaments(loaded, image_path)
        if task_name in REGION_YOLO_CONFIGS:
            return self._predict_region_yolo(loaded, image_path)
        if task_name == "屋顶四分类":
            return self._predict_roof_with_yolo(loaded, image_path)
        if task_name == "瓦片分类":
            return self._predict_tile(loaded, image_path)
        img = Image.open(image_path).convert("RGB")
        return self._predict_single_classification(loaded, img)

    def _format_detail_text(self, eng_pred: str, chn_pred: str, conf: float,
                            topk: List[Tuple[str, float]]) -> str:
        """
        格式化详情文本为中文显示格式
        """
        detail_lines = [
            f"预测结果（中文）：{chn_pred}",
            f"预测结果（英文）：{eng_pred}",
            f"置信度：{conf * 100:.2f}%\n",
            "各类别概率（中文 | 英文）："
        ]

        for name, prob in topk:
            chn_name = get_chinese_class_name(name)
            detail_lines.append(f"  {chn_name} | {name}: {prob * 100:.2f}%")

        return "\n".join(detail_lines)

    def _get_draw_font(self, size: int = 20):
        try:
            return ImageFont.truetype("msyh.ttc", size)
        except Exception:
            try:
                return ImageFont.truetype("arial.ttf", size)
            except Exception:
                return ImageFont.load_default()

    def _measure_text(self, draw: ImageDraw.ImageDraw, text: str, font):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                return draw.textsize(text, font=font)
            except Exception:
                return (max(20, len(text) * 10), 24)

    def _extract_best_yolo_box(
        self,
        results,
        target_class: int = 0,
        use_target_class_filter: bool = False,
        box_select_mode: str = "largest_area",
    ):
        """
        尽量贴近原始脚本：
        1) 默认不做类别过滤；
        2) 默认按面积最大的框作为最终检测框；
        3) 仅在明确配置时才启用 target_class 过滤或最高置信度选框。
        """
        if len(results) == 0 or getattr(results[0], "boxes", None) is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.ones(len(boxes), dtype=float)

        if use_target_class_filter and results[0].boxes.cls is not None:
            classes = results[0].boxes.cls.cpu().numpy()
            valid_indices = [i for i, c in enumerate(classes) if int(c) == int(target_class)]
            if len(valid_indices) == 0:
                return None
            boxes = boxes[valid_indices]
            confs = confs[valid_indices]

        if len(boxes) == 0:
            return None

        if str(box_select_mode).lower() == "highest_conf":
            best_idx = int(np.argmax(confs))
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))

        best_box = tuple(int(v) for v in boxes[best_idx])
        best_conf = float(confs[best_idx])
        return best_box, best_conf

    def _run_yolo_detection(
        self,
        yolo_model,
        source,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
        target_class: int,
        use_target_class_filter: bool = False,
        box_select_mode: str = "largest_area",
    ):
        results = yolo_model(
            source,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False,
        )
        det = self._extract_best_yolo_box(
            results,
            target_class=target_class,
            use_target_class_filter=use_target_class_filter,
            box_select_mode=box_select_mode,
        )
        return det, results

    def _predict_roof_with_yolo(self, loaded: LoadedModel, image_path: str) -> Tuple[
        str, float, List[Tuple[str, float]], str, Optional[Image.Image], str]:
        """
        屋顶四分类任务：
        1) 先用主YOLO检测屋顶；
        2) 若主YOLO未检出，则用备用YOLO检测建筑主体；
        3) 若备用YOLO检出主体，则按其检测框裁剪原图，再交给主YOLO二次检测；
        4) 最终只显示主YOLO的检测框；
        5) 若某个模型未检出，则在界面中返回明确报错信息，而不是抛出 traceback。
        """
        config = loaded.yolo_config or ROOF_YOLO_CONFIG
        main_yolo_path = str(config.get("yolo_model_path", "") or "").strip()
        backup_yolo_path = str(config.get("backup_yolo_model_path", "") or "").strip()

        def fail(message: str, show_img: Optional[Image.Image] = None):
            return "detection_failed", 0.0, [], message, show_img, "检测失败"

        if not YOLO_AVAILABLE:
            return fail("未安装 ultralytics，无法执行屋顶YOLO检测流程。请先安装：pip install ultralytics")
        if not main_yolo_path:
            return fail("屋顶四分类任务的主YOLO模型路径未配置：ROOF_YOLO_CONFIG['yolo_model_path'] 不能为空。")
        if loaded.yolo_model is None:
            return fail("主YOLO模型未正确加载，无法进行屋顶检测。")

        try:
            orig_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return fail(f"原图读取失败：{e}")

        orig_w, orig_h = orig_img.size

        try:
            main_det, _ = self._run_yolo_detection(
                loaded.yolo_model,
                image_path,
                conf_threshold=float(config.get("conf_threshold", 0.25)),
                iou_threshold=float(config.get("iou_threshold", 0.70)),
                max_det=int(config.get("max_det", 10)),
                target_class=int(config.get("target_class", 0)),
                use_target_class_filter=bool(config.get("use_target_class_filter", False)),
                box_select_mode=str(config.get("box_select_mode", "largest_area")),
            )
        except Exception as e:
            return fail(f"主YOLO检测执行失败：{e}", orig_img)

        detection_stage = "主YOLO直接检测"
        body_box_abs = None

        if main_det is not None:
            (x1, y1, x2, y2), best_conf = main_det
        else:
            if not backup_yolo_path:
                return fail(
                    "主YOLO未检测到屋顶目标，且未配置备用建筑主体YOLO模型："
                    "请填写 ROOF_YOLO_CONFIG['backup_yolo_model_path']。",
                    orig_img,
                )
            if loaded.backup_yolo_model is None:
                return fail("主YOLO未检测到屋顶目标，且备用建筑主体YOLO模型未正确加载。", orig_img)

            try:
                backup_det, _ = self._run_yolo_detection(
                    loaded.backup_yolo_model,
                    image_path,
                    conf_threshold=float(config.get("backup_conf_threshold", 0.25)),
                    iou_threshold=float(config.get("backup_iou_threshold", 0.70)),
                    max_det=int(config.get("backup_max_det", 10)),
                    target_class=int(config.get("backup_target_class", 0)),
                    use_target_class_filter=bool(config.get("backup_use_target_class_filter", False)),
                    box_select_mode=str(config.get("backup_box_select_mode", "largest_area")),
                )
            except Exception as e:
                return fail(f"备用建筑主体YOLO检测执行失败：{e}", orig_img)

            if backup_det is None:
                return fail(
                    "主YOLO未检测到屋顶目标；备用建筑主体YOLO也未检测到目标。\n"
                    f"主模型路径: {main_yolo_path}\n备用模型路径: {backup_yolo_path}",
                    orig_img,
                )

            (bx1, by1, bx2, by2), backup_conf = backup_det
            bx1 = max(0, min(orig_w, int(bx1)))
            by1 = max(0, min(orig_h, int(by1)))
            bx2 = max(0, min(orig_w, int(bx2)))
            by2 = max(0, min(orig_h, int(by2)))
            if bx2 <= bx1 or by2 <= by1:
                return fail("备用建筑主体YOLO检测到了无效检测框，无法裁剪原图。", orig_img)

            body_box_abs = (bx1, by1, bx2, by2)
            body_roi = orig_img.crop(body_box_abs)
            body_roi_np = np.array(body_roi)

            try:
                main_det_roi, _ = self._run_yolo_detection(
                    loaded.yolo_model,
                    body_roi_np,
                    conf_threshold=float(config.get("conf_threshold", 0.25)),
                    iou_threshold=float(config.get("iou_threshold", 0.70)),
                    max_det=int(config.get("max_det", 10)),
                    target_class=int(config.get("target_class", 0)),
                    use_target_class_filter=bool(config.get("use_target_class_filter", False)),
                    box_select_mode=str(config.get("box_select_mode", "largest_area")),
                )
            except Exception as e:
                return fail(f"主YOLO在备用主体裁剪图上的二次检测执行失败：{e}", orig_img)

            if main_det_roi is None:
                return fail(
                    "主YOLO在原图上未检测到屋顶目标；备用建筑主体YOLO检测成功，"
                    "但主YOLO在主体裁剪图上仍未检测到屋顶目标。\n"
                    f"备用主体框: ({bx1}, {by1}) - ({bx2}, {by2})\n"
                    f"备用主体检测置信度: {backup_conf:.2%}",
                    orig_img,
                )

            (rx1, ry1, rx2, ry2), best_conf = main_det_roi
            x1, y1, x2, y2 = bx1 + int(rx1), by1 + int(ry1), bx1 + int(rx2), by1 + int(ry2)
            detection_stage = "主YOLO失败 -> 备用主体YOLO裁剪 -> 主YOLO二次检测"

        x1 = max(0, min(orig_w, int(x1)))
        y1 = max(0, min(orig_h, int(y1)))
        x2 = max(0, min(orig_w, int(x2)))
        y2 = max(0, min(orig_h, int(y2)))

        crop_w = x2 - x1
        crop_h = y2 - y1
        min_size = int(config.get("min_size", 0))
        if crop_w <= 0 or crop_h <= 0:
            return fail("主YOLO最终生成了无效检测框，无法进行屋顶裁剪分类。", orig_img)
        if min_size > 0 and (crop_w < min_size or crop_h < min_size):
            return fail(
                f"主YOLO最终检测框尺寸过小：{crop_w}x{crop_h}，小于 min_size={min_size}，无法可靠分类。",
                orig_img,
            )

        roof_roi = orig_img.crop((x1, y1, x2, y2))
        try:
            pred_name, pred_conf, topk, _, _, chinese_pred = self._predict_single_classification(loaded, roof_roi)
        except Exception as e:
            return fail(f"屋顶ROI分类执行失败：{e}", orig_img)

        detail_lines = [
            "屋顶四分类任务（贴近原始YOLO脚本的检测逻辑）",
            f"检测流程：{detection_stage}",
            f"主YOLO模型：{main_yolo_path}",
            f"主YOLO选框方式：{str(config.get('box_select_mode', 'largest_area'))}",
            f"主YOLO类别过滤：{'开启' if bool(config.get('use_target_class_filter', False)) else '关闭'}",
        ]
        if backup_yolo_path:
            detail_lines.append(f"备用建筑主体YOLO模型：{backup_yolo_path}")
        detail_lines += [
            f"主YOLO最终检测置信度：{best_conf:.2%}",
            f"主YOLO最终检测框：({x1}, {y1}) - ({x2}, {y2})，尺寸：{crop_w}x{crop_h}",
        ]
        if body_box_abs is not None:
            detail_lines.append(
                f"备用YOLO主体框：({body_box_abs[0]}, {body_box_abs[1]}) - ({body_box_abs[2]}, {body_box_abs[3]})"
            )
        detail_lines += [
            "",
            f"预测结果（中文）：{chinese_pred}",
            f"预测结果（英文）：{pred_name}",
            f"分类置信度：{pred_conf * 100:.2f}%",
            "",
            "各类别概率（中文 | 英文）：",
        ]
        for name, prob in topk:
            chn_name = get_chinese_class_name(name)
            detail_lines.append(f"  {chn_name} | {name}: {prob * 100:.2f}%")
        detail_text = "\n".join(detail_lines)

        vis_img = orig_img.copy()
        draw = ImageDraw.Draw(vis_img)
        font = self._get_draw_font(20)
        box_color = config.get("box_color", (0, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        label_text = f"屋顶区域 {best_conf:.2f}"
        text_w, text_h = self._measure_text(draw, label_text, font)
        text_y1 = max(0, y1 - text_h - 4)
        text_y2 = max(0, y1)
        draw.rectangle([x1, text_y1, x1 + text_w, text_y2], fill=box_color)
        draw.text((x1, max(0, y1 - text_h - 2)), label_text, fill=config.get("text_color", (255, 255, 255)), font=font)

        return pred_name, pred_conf, topk, detail_text, vis_img, chinese_pred

    def _predict_without_detection(self, loaded: LoadedModel, img: Image.Image, reason: str) -> Tuple[
        str, float, List[Tuple[str, float]], str, Optional[Image.Image], str]:
        """
        未检测到目标时，直接使用原图进行分类
        """
        pred_name, pred_conf, topk, _, _, chinese_pred = self._predict_single_classification(loaded, img)
        detail_lines = [
            f"屋顶四分类任务（{reason}，使用原图）",
            f"",
            f"预测结果（中文）：{chinese_pred}",
            f"预测结果（英文）：{pred_name}",
            f"置信度：{pred_conf * 100:.2f}%\n",
            "各类别概率（中文 | 英文）："
        ]
        for name, prob in topk:
            chn_name = get_chinese_class_name(name)
            detail_lines.append(f"  {chn_name} | {name}: {prob * 100:.2f}%")
        return pred_name, pred_conf, topk, "\n".join(detail_lines), None, chinese_pred

    def _predict_yolo_detect_cls(self, loaded: LoadedModel, image_path: str) -> Tuple[
        str, float, List[Tuple[str, float]], str, Optional[Image.Image], str]:
        """
        YOLO检测 + 分类 流程（独立任务）
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("未安装 ultralytics，无法使用YOLO功能。请运行: pip install ultralytics")

        if loaded.yolo_model is None or loaded.cls_model is None:
            raise RuntimeError("YOLO模型或分类模型未正确加载")

        config = loaded.yolo_config or YOLO_DETECT_CLS_CONFIG

        # 读取原图
        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size

        # YOLO检测
        results = loaded.yolo_model(
            image_path,
            conf=config.get("conf_threshold", 0.25),
            iou=config.get("iou_threshold", 0.45),
            max_det=config.get("max_det", 50),
            verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return "未检测到目标", 0.0, [], "未检测到任何目标", orig_img, "未检测到目标"

        # 准备分类
        cls_loaded = loaded.cls_model
        transform = self.get_transform(cls_loaded)

        detections = []
        draw_img = orig_img.copy()
        draw = ImageDraw.Draw(draw_img)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("msyh.ttc", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else [1.0] * len(boxes)

        for i, (box, det_conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 裁剪ROI
            roi = orig_img.crop((x1, y1, x2, y2))

            # 分类ROI
            x = transform(roi).unsqueeze(0).to(cls_loaded.device)

            tta_mode = cls_loaded.tta_mode if cls_loaded.tta_mode else ("flip2" if (config.get("cls_use_tta", False) or cls_loaded.use_kaijian_tta) else "none")

            cls_loaded.model.eval()
            with torch.no_grad():
                logits = self._forward_with_tta(cls_loaded.model, x, tta_mode)

                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = F.softmax(logits, dim=1)[0].cpu()

            pred_idx = int(torch.argmax(probs).item())
            pred_name = cls_loaded.labels[pred_idx] if pred_idx < len(cls_loaded.labels) else f"class_{pred_idx}"
            pred_conf = float(probs[pred_idx].item())

            detections.append({
                'box': (x1, y1, x2, y2),
                'det_conf': float(det_conf),
                'cls_name': pred_name,
                'cls_conf': pred_conf,
                'cls_probs': probs,
                'all_probs': [(cls_loaded.labels[j] if j < len(cls_loaded.labels) else f"class_{j}",
                               float(probs[j].item())) for j in range(len(probs))]
            })

            # 绘制检测框（使用中文标签）
            if config.get("draw_boxes", True):
                box_color = config.get("box_color", (255, 0, 0))
                if isinstance(box_color, tuple) and len(box_color) == 3:
                    draw_color = box_color
                else:
                    draw_color = (255, 0, 0)

                draw.rectangle([x1, y1, x2, y2], outline=draw_color, width=3)

                chn_label = get_chinese_class_name(pred_name)
                label_text = f"{chn_label} {pred_conf:.2f}"
                text_w, text_h = self._measure_text(draw, label_text, font)

                draw.rectangle([x1, y1 - text_h - 4, x1 + text_w, y1], fill=draw_color)
                draw.text((x1, y1 - text_h - 2), label_text, fill=config.get("text_color", (255, 255, 255)), font=font)

        # 生成结果文本
        if len(detections) == 0:
            return "未检测到有效目标", 0.0, [], "检测框裁剪后无效", draw_img, "未检测到有效目标"

        # 以置信度最高的检测结果作为主要输出
        best_det = max(detections, key=lambda x: x['cls_conf'] * x['det_conf'])
        main_pred = best_det['cls_name']
        main_conf = best_det['cls_conf']
        main_pred_chn = get_chinese_class_name(main_pred)

        # 构建TopK
        topk = sorted(best_det['all_probs'], key=lambda x: x[1], reverse=True)

        detail_lines = [
            f"YOLO检测+分类任务",
            f"检测到 {len(detections)} 个目标",
            f"主要预测（中文）：{main_pred_chn}",
            f"主要预测（英文）：{main_pred}",
            f"综合置信度: {main_conf * best_det['det_conf']:.2%}",
            "",
            "详细检测结果:",
        ]

        for i, det in enumerate(detections):
            det_chn = get_chinese_class_name(det['cls_name'])
            detail_lines.append(f"\n目标 {i + 1}:")
            detail_lines.append(f"  位置: ({det['box'][0]}, {det['box'][1]}) - ({det['box'][2]}, {det['box'][3]})")
            detail_lines.append(f"  检测置信度: {det['det_conf']:.2%}")
            detail_lines.append(f"  分类结果（中文）：{det_chn}")
            detail_lines.append(f"  分类结果（英文）：{det['cls_name']} ({det['cls_conf']:.2%})")
            detail_lines.append(f"  类别概率Top3（中文 | 英文）：")
            for cls_name, cls_prob in sorted(det['all_probs'], key=lambda x: x[1], reverse=True)[:3]:
                chn_cls_name = get_chinese_class_name(cls_name)
                detail_lines.append(f"    - {chn_cls_name} | {cls_name}: {cls_prob:.2%}")

        detail_text = "\n".join(detail_lines)

        return main_pred, main_conf, topk, detail_text, draw_img, main_pred_chn

    def _predict_roof_ridge_ornaments(self, loaded: LoadedModel, image_path: str) -> Tuple[
        str, float, List[Tuple[str, float]], str, Optional[Image.Image], str]:
        """屋脊装饰识别：先定位屋顶区域，再在屋顶 ROI 内执行 YOLO 实例分割，并把结果映射回完整原图。"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("未安装 ultralytics，无法使用屋脊装饰识别功能。请运行: pip install ultralytics")
        if loaded.yolo_model is None:
            raise RuntimeError("屋脊装饰 YOLO 模型未正确加载")

        config = loaded.yolo_config or ROOF_RIDGE_ORNAMENT_CONFIG
        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        roof_box_abs = (0, 0, orig_w, orig_h)
        roof_conf = 0.0
        roof_stage = "未启用屋顶前置检测，直接使用完整原图"

        use_roof_pre_detection = bool(config.get("use_roof_pre_detection", True))
        if use_roof_pre_detection:
            if loaded.backup_yolo_model is None:
                detail = (
                    "屋脊装饰识别任务\n"
                    "屋顶前置检测已启用，但屋顶区域 YOLO 模型未正确加载，无法继续执行脊饰识别。\n"
                    f"屋顶模型路径：{config.get('roof_yolo_model_path', '')}"
                )
                return "屋顶检测模型未加载", 0.0, [], detail, orig_img, "屋顶检测模型未加载"

            try:
                roof_det, _ = self._run_yolo_detection(
                    loaded.backup_yolo_model,
                    image_path,
                    conf_threshold=float(config.get("roof_conf_threshold", 0.25)),
                    iou_threshold=float(config.get("roof_iou_threshold", 0.70)),
                    max_det=int(config.get("roof_max_det", 20)),
                    target_class=int(config.get("roof_target_class", 0)),
                    use_target_class_filter=bool(config.get("roof_use_target_class_filter", False)),
                    box_select_mode=str(config.get("roof_box_select_mode", "largest_area")),
                )
            except Exception as e:
                detail = f"屋脊装饰识别任务\n屋顶前置检测执行失败：{e}"
                return "屋顶检测失败", 0.0, [], detail, orig_img, "屋顶检测失败"

            if roof_det is None:
                detail = (
                    "屋脊装饰识别任务\n"
                    "未检测到建筑屋顶区域，因此未继续执行屋脊装饰分割。\n"
                    f"屋顶模型路径：{config.get('roof_yolo_model_path', '')}\n"
                    f"输入尺寸：{orig_w}x{orig_h}"
                )
                return "未检测到屋顶区域", 0.0, [], detail, orig_img, "未检测到屋顶区域"

            (rx1, ry1, rx2, ry2), roof_conf = roof_det
            pad_ratio = float(config.get("roof_padding_ratio", 0.0))
            pad_px = int(config.get("roof_padding_px", 0))
            box_w = max(1, int(rx2) - int(rx1))
            box_h = max(1, int(ry2) - int(ry1))
            pad_x = int(round(box_w * pad_ratio)) + pad_px
            pad_y = int(round(box_h * pad_ratio)) + pad_px
            x1 = max(0, int(rx1) - pad_x)
            y1 = max(0, int(ry1) - pad_y)
            x2 = min(orig_w, int(rx2) + pad_x)
            y2 = min(orig_h, int(ry2) + pad_y)
            if x2 <= x1 or y2 <= y1:
                detail = "屋脊装饰识别任务\n屋顶前置检测得到无效检测框，无法继续执行脊饰识别。"
                return "屋顶检测框无效", 0.0, [], detail, orig_img, "屋顶检测框无效"
            roof_box_abs = (x1, y1, x2, y2)
            roof_stage = "屋顶前置检测 -> 屋顶 ROI 脊饰分割 -> 坐标映射回完整原图"

        crop_x1, crop_y1, crop_x2, crop_y2 = roof_box_abs
        roof_roi = orig_img.crop(roof_box_abs)
        roof_roi_np = np.array(roof_roi)
        crop_w, crop_h = roof_roi.size

        results = loaded.yolo_model(
            roof_roi_np,
            imgsz=int(config.get("imgsz", 960)),
            conf=float(config.get("conf_threshold", 0.25)),
            iou=float(config.get("iou_threshold", 0.70)),
            max_det=int(config.get("max_det", 100)),
            retina_masks=bool(config.get("retina_masks", True)),
            verbose=False,
        )

        def draw_roof_box(image: Image.Image) -> Image.Image:
            if not use_roof_pre_detection or not bool(config.get("draw_roof_box", True)):
                return image
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)
            font = self._get_draw_font(20)
            color = tuple(config.get("roof_box_color", (34, 197, 94)))
            text_color = tuple(config.get("text_color", (255, 255, 255)))
            draw.rectangle([crop_x1, crop_y1, crop_x2, crop_y2], outline=color, width=max(2, int(config.get("box_width", 3))))
            label_text = f"屋顶区域 {roof_conf:.2f}"
            text_w, text_h = self._measure_text(draw, label_text, font)
            label_y1 = max(0, crop_y1 - text_h - 6)
            draw.rectangle([crop_x1, label_y1, crop_x1 + text_w + 6, label_y1 + text_h + 6], fill=color)
            draw.text((crop_x1 + 3, label_y1 + 2), label_text, fill=text_color, font=font)
            return draw_img

        if len(results) == 0 or getattr(results[0], "boxes", None) is None or len(results[0].boxes) == 0:
            vis_img = draw_roof_box(orig_img)
            detail = (
                "屋脊装饰识别任务（屋顶前置检测 + YOLO实例分割）\n"
                f"处理流程：{roof_stage}\n"
                f"屋顶检测框：({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})，尺寸：{crop_w}x{crop_h}\n"
                f"屋顶检测置信度：{roof_conf:.2%}\n"
                "未在屋顶区域内检测到龙、珠、塔、瓶、人物等屋脊装饰目标。"
            )
            return "未检测到目标", 0.0, [], detail, vis_img, "未检测到目标"

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes), dtype=float)
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
        model_names = getattr(result, "names", None) or getattr(loaded.yolo_model, "names", {}) or {}
        labels = list(config.get("labels", loaded.labels or []))
        class_colors = config.get("class_colors", {}) or {}
        vis_img = orig_img.copy()
        overlay = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        font = self._get_draw_font(20)
        detections = []
        mask_polygons = []
        if getattr(result, "masks", None) is not None and getattr(result.masks, "xy", None) is not None:
            mask_polygons = result.masks.xy

        for i, (box, det_conf, cls_id) in enumerate(zip(boxes, confs, classes)):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(crop_w, x2), min(crop_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            abs_box = (crop_x1 + x1, crop_y1 + y1, crop_x1 + x2, crop_y1 + y2)
            cls_name = labels[int(cls_id)] if labels and 0 <= int(cls_id) < len(labels) else str(model_names.get(int(cls_id), f"class_{int(cls_id)}"))
            chn_name = get_chinese_class_name(cls_name)
            color = class_colors.get(cls_name, (37, 99, 235))
            if bool(config.get("draw_masks", True)) and i < len(mask_polygons):
                polygon = [(float(x) + crop_x1, float(y) + crop_y1) for x, y in mask_polygons[i]]
                if len(polygon) >= 3:
                    overlay_draw.polygon(polygon, fill=tuple(color) + (int(config.get("mask_alpha", 90)),))
            detections.append({"class_id": int(cls_id), "class_name": cls_name, "chinese_name": chn_name, "conf": float(det_conf), "box": abs_box})

        if bool(config.get("draw_masks", True)):
            vis_img = Image.alpha_composite(vis_img.convert("RGBA"), overlay).convert("RGB")
        vis_img = draw_roof_box(vis_img)
        draw = ImageDraw.Draw(vis_img)
        if bool(config.get("draw_boxes", True)):
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                color = class_colors.get(det["class_name"], (37, 99, 235))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=int(config.get("box_width", 3)))
                label_text = f"{det['chinese_name']} {det['conf']:.2f}"
                text_w, text_h = self._measure_text(draw, label_text, font)
                label_y1 = max(0, y1 - text_h - 4)
                draw.rectangle([x1, label_y1, x1 + text_w + 4, label_y1 + text_h + 4], fill=color)
                draw.text((x1 + 2, label_y1 + 1), label_text, fill=config.get("text_color", (255, 255, 255)), font=font)

        if not detections:
            detail = (
                "屋脊装饰识别任务（屋顶前置检测 + YOLO实例分割）\n"
                "检测结果存在，但有效框为空。\n"
                f"屋顶检测框：({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})"
            )
            return "未检测到有效目标", 0.0, [], detail, vis_img, "未检测到有效目标"

        counts: Dict[str, int] = {}
        conf_sum: Dict[str, float] = {}
        for det in detections:
            counts[det["chinese_name"]] = counts.get(det["chinese_name"], 0) + 1
            conf_sum[det["chinese_name"]] = conf_sum.get(det["chinese_name"], 0.0) + det["conf"]
        best_det = max(detections, key=lambda item: item["conf"])
        avg_conf = float(sum(det["conf"] for det in detections) / len(detections))
        topk = sorted([(name, conf_sum[name] / max(counts[name], 1)) for name in counts], key=lambda item: item[1], reverse=True)
        summary = "，".join(f"{name}{count}个" for name, count in sorted(counts.items()))
        main_pred = f"检测到{len(detections)}个目标"
        chinese_pred = summary if summary else main_pred
        detail_lines = [
            "屋脊装饰识别任务（屋顶前置检测 + YOLO实例分割）",
            f"处理流程：{roof_stage}",
            f"屋顶模型路径：{config.get('roof_yolo_model_path', '')}",
            f"脊饰模型路径：{config.get('yolo_model_path', '')}",
            f"原图尺寸：{orig_w}x{orig_h}",
            f"屋顶检测框：({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})，尺寸：{crop_w}x{crop_h}",
            f"屋顶检测置信度：{roof_conf:.2%}",
            f"脊饰检测目标数：{len(detections)}",
            f"平均置信度：{avg_conf:.2%}",
            f"最高置信度目标：{best_det['chinese_name']} ({best_det['conf']:.2%})",
            "",
            "类别统计：",
        ]
        for name, count in sorted(counts.items()):
            detail_lines.append(f"  {name}: {count} 个，平均置信度 {conf_sum[name] / count:.2%}")
        detail_lines.append("\n详细检测结果（坐标为完整原图坐标）：")
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["box"]
            detail_lines.append(f"  目标{i}: {det['chinese_name']} | 置信度 {det['conf']:.2%} | 位置 ({x1}, {y1}) - ({x2}, {y2})")
        return main_pred, avg_conf, topk, "\n".join(detail_lines), vis_img, chinese_pred
    def load_model(self, task_name: str) -> LoadedModel:
        if task_name in self.cache:
            return self.cache[task_name]

        if task_name == "塌寿三分类":
            loaded = self._load_tashou_model()
        elif task_name == "屋顶四分类":
            loaded = self._load_roof_model()
        elif task_name == "开间分类":
            loaded = self._load_kaijian_model()
        elif task_name == "瓦片分类":
            loaded = self._load_tile_model()
        elif task_name == "YOLO检测+分类":
            loaded = self._load_yolo_detect_cls_model()
        elif task_name == "屋脊装饰识别":
            loaded = self._load_roof_ridge_ornament_model()
        elif task_name in REGION_YOLO_CONFIGS:
            loaded = self._load_region_yolo_model(task_name)
        else:
            raise ValueError(f"未知任务：{task_name}")

        self.cache[task_name] = loaded
        return loaded

    def clear_task_cache(self, task_name: str):
        if task_name in self.cache:
            del self.cache[task_name]

    def _safe_load_ckpt(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"权重文件不存在：{path}")
        return torch.load(path, map_location="cpu")

    def _load_region_yolo_model(self, task_name: str) -> LoadedModel:
        """加载建筑主体/屋顶区域 YOLO 检测模型。"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("未安装 ultralytics。请运行: pip install ultralytics")
        config = REGION_YOLO_CONFIGS[task_name]
        yolo_path = str(config.get("yolo_model_path", "") or "").strip()
        if not yolo_path:
            raise RuntimeError(f"{task_name}模型路径未配置")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"{task_name}模型文件不存在：{yolo_path}")
        yolo_model = YOLO(yolo_path)
        label = str(config.get("label", task_name))
        info = (
            f"YOLO目标检测模型: {os.path.basename(yolo_path)}\n"
            f"检测类别: {label}\n"
            f"conf: {config.get('conf_threshold', 0.25)} | iou: {config.get('iou_threshold', 0.70)}"
        )
        return LoadedModel(
            task_name=task_name,
            model=nn.Identity(),
            labels=[label],
            input_size=640,
            device=self.device,
            extra_info=info,
            normalize=False,
            yolo_model=yolo_model,
            yolo_config=config,
        )

    def _predict_region_yolo(self, loaded: LoadedModel, image_path: str) -> Tuple[
        str, float, List[Tuple[str, float]], str, Optional[Image.Image], str]:
        """执行建筑主体/屋顶区域检测，并返回带框结果图。"""
        config = loaded.yolo_config or REGION_YOLO_CONFIGS.get(loaded.task_name, {})
        if loaded.yolo_model is None:
            return "not_loaded", 0.0, [], f"{loaded.task_name}模型未正确加载。", None, "模型未加载"

        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        results = loaded.yolo_model(
            image_path,
            conf=float(config.get("conf_threshold", 0.25)),
            iou=float(config.get("iou_threshold", 0.70)),
            max_det=int(config.get("max_det", 20)),
            verbose=False,
        )

        detections = []
        if results and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.ones(len(boxes), dtype=float)
            for box, score in zip(boxes, confs):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({"box": (x1, y1, x2, y2), "conf": float(score)})

        vis_img = orig_img.copy()
        draw = ImageDraw.Draw(vis_img)
        font = self._get_draw_font(22)
        label = str(config.get("label", loaded.task_name))
        color = tuple(config.get("box_color", (34, 197, 94)))
        text_color = tuple(config.get("text_color", (255, 255, 255)))
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            text_w, text_h = self._measure_text(draw, label, font)
            label_y = max(0, y1 - text_h - 8)
            draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 8], fill=color)
            draw.text((x1 + 4, label_y + 2), label, fill=text_color, font=font)

        count = len(detections)
        avg_conf = float(sum(det["conf"] for det in detections) / count) if count else 0.0
        chinese_pred = f"{label}{count}个" if count else f"未检测到{label}"
        pred_name = "detected" if count else "not_detected"
        topk = [(label, avg_conf)] if count else []
        detail_lines = [
            f"{loaded.task_name}",
            f"模型路径：{config.get('yolo_model_path', '')}",
            f"输入尺寸：{orig_w}x{orig_h}",
            f"检测区域数：{count}",
        ]
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["box"]
            detail_lines.append(f"区域{i}: 置信度 {det['conf']:.2%} | 位置 ({x1}, {y1}) - ({x2}, {y2})")
        return pred_name, avg_conf, topk, "\n".join(detail_lines), vis_img, chinese_pred

    def _load_roof_ridge_ornament_model(self) -> LoadedModel:
        """加载屋脊装饰 YOLO 实例分割模型，并按配置加载屋顶前置检测模型。"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("未安装 ultralytics。请运行: pip install ultralytics")
        config = ROOF_RIDGE_ORNAMENT_CONFIG
        yolo_path = str(config.get("yolo_model_path", "") or "").strip()
        if not yolo_path:
            raise RuntimeError("屋脊装饰识别模型路径未配置")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"屋脊装饰识别模型文件不存在：{yolo_path}")
        yolo_model = YOLO(yolo_path)

        roof_yolo_model = None
        roof_yolo_info = "屋顶前置检测：未启用"
        if bool(config.get("use_roof_pre_detection", True)):
            roof_yolo_path = str(config.get("roof_yolo_model_path", "") or "").strip()
            if not roof_yolo_path:
                raise RuntimeError("屋脊装饰识别已启用屋顶前置检测，但 roof_yolo_model_path 未配置")
            if not os.path.exists(roof_yolo_path):
                raise FileNotFoundError(f"屋顶前置检测模型文件不存在：{roof_yolo_path}")
            roof_yolo_model = YOLO(roof_yolo_path)
            roof_yolo_info = (
                f"屋顶前置检测模型: {os.path.basename(roof_yolo_path)}\n"
                f"屋顶conf: {config.get('roof_conf_threshold', 0.25)} | 屋顶iou: {config.get('roof_iou_threshold', 0.70)}"
            )

        labels = list(config.get("labels", TASK_LABELS["屋脊装饰识别"]))
        info = (
            f"YOLO实例分割模型: {os.path.basename(yolo_path)}\n"
            f"类别: {', '.join(labels)}\n"
            f"imgsz: {config.get('imgsz', 960)}\n"
            f"conf: {config.get('conf_threshold', 0.25)} | iou: {config.get('iou_threshold', 0.70)}\n"
            f"{roof_yolo_info}"
        )
        return LoadedModel(
            task_name="屋脊装饰识别",
            model=nn.Identity(),
            labels=labels,
            input_size=int(config.get("imgsz", 960)),
            device=self.device,
            extra_info=info,
            normalize=False,
            yolo_model=yolo_model,
            backup_yolo_model=roof_yolo_model,
            yolo_config=config,
        )
    def _load_roof_model(self) -> LoadedModel:
        """
        加载屋顶四分类模型：
        - 分类模型按独立屋顶分类脚本：torchvision.resnet34 + Resize(224,224)
        - 检测模型支持主YOLO（屋顶）+ 备用YOLO（建筑主体）
        """
        path = MODEL_PATHS["屋顶四分类"]
        labels = list(TASK_LABELS["屋顶四分类"])
        num_classes = len(labels)
        ckpt = self._safe_load_ckpt(path)
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            state_dict = extract_state_dict(ckpt)
        state_dict = strip_module_prefix(state_dict)

        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        load_note = "strict=True"
        try:
            model.load_state_dict(state_dict, strict=True)
            missing, unexpected = [], []
        except Exception:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            load_note = f"strict=False | missing={len(missing)} unexpected={len(unexpected)}"
        model.to(self.device)
        model.eval()

        yolo_config = ROOF_YOLO_CONFIG
        main_yolo_path = str(yolo_config.get("yolo_model_path", "") or "").strip()
        backup_yolo_path = str(yolo_config.get("backup_yolo_model_path", "") or "").strip()
        yolo_model = None
        backup_yolo_model = None

        main_yolo_info = ""
        if not main_yolo_path:
            main_yolo_info = "主YOLO路径未配置"
        elif not YOLO_AVAILABLE:
            main_yolo_info = "未安装ultralytics，主YOLO无法加载"
        elif not os.path.exists(main_yolo_path):
            main_yolo_info = f"主YOLO路径不存在: {main_yolo_path}"
        else:
            try:
                yolo_model = YOLO(main_yolo_path)
                main_yolo_info = f"主YOLO已加载: {os.path.basename(main_yolo_path)}"
            except Exception as e:
                main_yolo_info = f"主YOLO加载失败: {e}"
                yolo_model = None

        backup_yolo_info = ""
        if not backup_yolo_path:
            backup_yolo_info = "备用主体YOLO路径未配置"
        elif not YOLO_AVAILABLE:
            backup_yolo_info = "未安装ultralytics，备用主体YOLO无法加载"
        elif not os.path.exists(backup_yolo_path):
            backup_yolo_info = f"备用主体YOLO路径不存在: {backup_yolo_path}"
        else:
            try:
                backup_yolo_model = YOLO(backup_yolo_path)
                backup_yolo_info = f"备用主体YOLO已加载: {os.path.basename(backup_yolo_path)}"
            except Exception as e:
                backup_yolo_info = f"备用主体YOLO加载失败: {e}"
                backup_yolo_model = None

        info = (
            f"骨干网络：torchvision.models.resnet34\n"
            f"预处理：Resize(({ROOF_CLASSIFY_CONFIG['input_size']}, {ROOF_CLASSIFY_CONFIG['input_size']})) + Normalize\n"
            f"权重加载：{load_note}\n"
            f"{main_yolo_info}\n"
            f"{backup_yolo_info}"
        )
        return LoadedModel(
            task_name="屋顶四分类",
            model=model,
            labels=labels,
            input_size=int(ROOF_CLASSIFY_CONFIG.get("input_size", 224)),
            device=self.device,
            extra_info=info,
            mean=list(ROOF_CLASSIFY_CONFIG.get("mean", [0.485, 0.456, 0.406])),
            std=list(ROOF_CLASSIFY_CONFIG.get("std", [0.229, 0.224, 0.225])),
            normalize=bool(ROOF_CLASSIFY_CONFIG.get("normalize", True)),
            preprocess_mode=str(ROOF_CLASSIFY_CONFIG.get("preprocess_mode", "roof_resize")),
            preprocess_cfg=dict(ROOF_CLASSIFY_CONFIG),
            yolo_model=yolo_model,
            yolo_config=yolo_config,
            backup_yolo_model=backup_yolo_model,
        )

    def _load_yolo_detect_cls_model(self) -> LoadedModel:
        """
        加载YOLO检测+分类双模型
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("未安装 ultralytics。请运行: pip install ultralytics")
        config = YOLO_DETECT_CLS_CONFIG
        yolo_path = config.get("yolo_model_path", "")
        if not yolo_path:
            raise RuntimeError("YOLO模型路径未配置")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO模型文件不存在：{yolo_path}")
        yolo_model = YOLO(yolo_path)
        cls_path = config.get("cls_model_path", "")
        if not cls_path:
            raise RuntimeError("分类模型路径未配置")
        if not os.path.exists(cls_path):
            raise FileNotFoundError(f"分类模型文件不存在：{cls_path}")
        cls_arch = config.get("cls_model_arch", "resnet34")
        cls_num_classes = int(config.get("cls_num_classes", 4))
        cls_labels = list(config.get("cls_labels", [f"class_{i}" for i in range(cls_num_classes)]))
        cls_input_size = int(config.get("cls_input_size", 224))
        cls_normalize = bool(config.get("cls_normalize", True))
        cls_mean = list(config.get("cls_mean", [0.485, 0.456, 0.406]))
        cls_std = list(config.get("cls_std", [0.229, 0.224, 0.225]))
        cls_pad_bg_color = tuple(config.get("cls_pad_bg_color", (248, 250, 252)))
        cls_tta_mode = str(config.get("cls_tta_mode", "flip2" if config.get("cls_use_tta", False) else "none"))
        cls_preprocess_mode = str(config.get("cls_preprocess_mode", "roof_resize"))
        ckpt = torch.load(cls_path, map_location="cpu")
        state_dict = strip_module_prefix(extract_state_dict(ckpt))
        if cls_arch == "resnet34":
            cls_model = models.resnet34(weights=None)
            cls_model.fc = nn.Linear(cls_model.fc.in_features, cls_num_classes)
            try:
                cls_model.load_state_dict(state_dict, strict=True)
                missing, unexpected = [], []
                load_note = "strict=True"
            except Exception:
                missing, unexpected = cls_model.load_state_dict(state_dict, strict=False)
                load_note = f"strict=False | missing={len(missing)} unexpected={len(unexpected)}"
        elif cls_arch == "swin_v2_t":
            weights = Swin_V2_T_Weights.DEFAULT
            cls_mean = list(weights.transforms().mean)
            cls_std = list(weights.transforms().std)
            cls_model = models.swin_v2_t(weights=weights)
            cls_model.head = nn.Linear(cls_model.head.in_features, cls_num_classes)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else state_dict
            cls_model.load_state_dict(strip_module_prefix(state), strict=True)
            load_note = "strict=True"
        else:
            cls_model = timm.create_model(cls_arch, pretrained=False, num_classes=cls_num_classes)
            missing, unexpected = cls_model.load_state_dict(state_dict, strict=False)
            load_note = f"strict=False | missing={len(missing)} unexpected={len(unexpected)}"
        cls_model.to(self.device)
        cls_model.eval()
        cls_info = (
            f"独立分类模型: {cls_arch}\n"
            f"预处理模式: {cls_preprocess_mode}\n"
            f"TTA: {cls_tta_mode}\n"
            f"权重加载：{load_note}"
        )
        cls_loaded = LoadedModel(
            task_name="YOLO分类模型",
            model=cls_model,
            labels=cls_labels,
            input_size=cls_input_size,
            device=self.device,
            extra_info=cls_info,
            mean=cls_mean,
            std=cls_std,
            normalize=cls_normalize,
            pad_bg_color=cls_pad_bg_color,
            tta_mode=cls_tta_mode,
            preprocess_mode=cls_preprocess_mode,
            preprocess_cfg=dict(config),
        )
        info = (
            f"YOLO检测模型: {os.path.basename(yolo_path)}\n"
            f"分类模型: {cls_arch} ({os.path.basename(cls_path)})\n"
            f"分类类别（英文）: {', '.join(cls_labels)}\n"
            f"分类类别（中文）: {', '.join([get_chinese_class_name(l) for l in cls_labels])}"
        )
        return LoadedModel(
            task_name="YOLO检测+分类",
            model=cls_model,
            labels=cls_labels,
            input_size=cls_input_size,
            device=self.device,
            extra_info=info,
            mean=cls_mean,
            std=cls_std,
            normalize=cls_normalize,
            pad_bg_color=cls_pad_bg_color,
            tta_mode=cls_tta_mode,
            preprocess_mode=cls_preprocess_mode,
            preprocess_cfg=dict(config),
            yolo_model=yolo_model,
            cls_model=cls_loaded,
            yolo_config=config
        )

    def _load_tashou_model(self) -> LoadedModel:
        path = MODEL_PATHS["塌寿三分类"]
        ckpt = self._safe_load_ckpt(path)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "class_to_idx" not in ckpt:
            raise RuntimeError("塌寿权重格式不符合新脚本要求：必须包含 model_state_dict 和 class_to_idx")
        class_to_idx = {str(k).lower(): int(v) for k, v in ckpt["class_to_idx"].items()}
        idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
        labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        num_classes = len(labels)
        weights = EfficientNet_V2_S_Weights.DEFAULT
        mean = list(weights.transforms().mean)
        std = list(weights.transforms().std)
        model = efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(self.device)
        model.eval()
        info = (
            f"骨干网络：efficientnet_v2_s\n"
            f"预处理：Resize(int({TASHOU_INFER_CONFIG['img_size']}*{TASHOU_INFER_CONFIG['resize_scale']})) + CenterCrop({TASHOU_INFER_CONFIG['img_size']})\n"
            f"类别来源：checkpoint.class_to_idx\n"
            f"类别名：{', '.join(labels)}"
        )
        if "acc" in ckpt:
            try:
                info += f"\ncheckpoint记录精度：{float(ckpt['acc']):.2f}%"
            except Exception:
                pass
        if missing:
            info += f"\nmissing keys: {len(missing)}"
        if unexpected:
            info += f"\nunexpected keys: {len(unexpected)}"
        return LoadedModel(
            task_name="塌寿三分类",
            model=model,
            labels=labels,
            input_size=int(TASHOU_INFER_CONFIG.get("img_size", 256)),
            device=self.device,
            extra_info=info,
            mean=mean,
            std=std,
            normalize=True,
            preprocess_mode="tashou_eval",
            preprocess_cfg=dict(TASHOU_INFER_CONFIG),
        )

    def _load_kaijian_model(self) -> LoadedModel:
        path = MODEL_PATHS["开间分类"]
        ckpt = self._safe_load_ckpt(path)
        labels = list(KAIJIAN_INFER_CONFIG.get("class_names", TASK_LABELS["开间分类"]))
        if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], list) and len(ckpt["classes"]) > 0:
            labels = ckpt["classes"]
        weights = Swin_V2_T_Weights.DEFAULT
        mean = list(weights.transforms().mean)
        std = list(weights.transforms().std)
        model = models.swin_v2_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, len(labels))
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(strip_module_prefix(state), strict=True)
        model.to(self.device)
        model.eval()
        cfg = dict(KAIJIAN_INFER_CONFIG)
        cfg["img_size"] = int(cfg.get("img_size", 224))
        return LoadedModel(
            task_name="开间分类",
            model=model,
            labels=labels,
            input_size=int(cfg.get("img_size", 224)),
            device=self.device,
            extra_info=(
                "骨干网络：torchvision.models.swin_v2_t\n"
                f"预处理：{cfg.get('resize_mode', 'resize')} | ac={cfg.get('ac', False)} | g={cfg.get('g', 1.0)} | sh={cfg.get('sh', 1.0)}\n"
                f"TTA：{cfg.get('tta', 'flip2')}"
            ),
            mean=mean,
            std=std,
            normalize=True,
            use_kaijian_tta=(str(cfg.get("tta", "none")).lower() != "none"),
            tta_mode=str(cfg.get("tta", "flip2")),
            preprocess_mode="kaijian_cfg",
            preprocess_cfg=cfg,
        )

    def _load_tile_model(self) -> LoadedModel:
        path = MODEL_PATHS["瓦片分类"]
        ckpt = self._safe_load_ckpt(path)
        if not isinstance(ckpt, dict):
            raise RuntimeError("瓦片权重不是 dict 格式")
        if "model" not in ckpt:
            raise RuntimeError("瓦片权重中未找到 'model'")
        if "label_maps" not in ckpt or "tile_type" not in ckpt["label_maps"] or "inv_tile_type" not in ckpt["label_maps"]:
            raise RuntimeError("瓦片权重中未找到 label_maps['tile_type'] / label_maps['inv_tile_type']")
        label_maps = ckpt["label_maps"]
        labels = parse_inv_map_to_labels(label_maps["inv_tile_type"])
        num_classes = len(labels)
        backbone_name = ckpt.get("backbone", "torchvision_resnet34")
        model = TileNet(backbone_name=backbone_name, n_tile=num_classes)
        state_dict = strip_module_prefix(ckpt["model"])
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        cfg = dict(TILE_INFER_CONFIG)
        return LoadedModel(
            task_name="瓦片分类",
            model=model,
            labels=labels,
            input_size=int(cfg.get("image_size", 224)),
            device=self.device,
            extra_info=(
                "骨干网络：TileNet(torchvision resnet34)\n"
                f"预处理：resize_pad_to_square(size={cfg.get('image_size', 224)}) + /255\n"
                f"可选 mask 裁剪：{cfg.get('use_mask_crop', False)}\n"
                f"类别名：{', '.join(labels)}"
            ),
            normalize=False,
            pad_bg_color=(0, 0, 0),
            preprocess_mode="tile_square_pad",
            preprocess_cfg=cfg,
            masks_dir=cfg.get("masks_dir", ""),
            use_mask_crop=bool(cfg.get("use_mask_crop", False)),
        )


# =========================
# 自定义控件
# =========================
class ImagePreview(QLabel):
    def __init__(self, title: str = "图片预览区域"):
        super().__init__()
        self.setMinimumSize(620, 500)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8fafc,
                    stop:1 #eef2f7
                );
                border: 2px dashed #cbd5e1;
                border-radius: 22px;
                color: #64748b;
                font-size: 18px;
                font-weight: 600;
            }
        """)
        self.setText(f"{title}\n\n点击“选择图片”后在这里展示\n或直接拖入图片")

    def set_pil_image(self, img: Image.Image):
        pix = pil_to_qpixmap(img)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap() is not None:
            self.setPixmap(self.pixmap().scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                local_path = urls[0].toLocalFile().lower()
                if local_path.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                local_path = urls[0].toLocalFile().lower()
                if local_path.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                local_path = urls[0].toLocalFile()
                if local_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    parent = self.parentWidget()
                    while parent is not None and not hasattr(parent, "load_image_from_path"):
                        parent = parent.parentWidget()
                    if parent is not None:
                        parent.load_image_from_path(local_path)
                    event.acceptProposedAction()
                    return
        event.ignore()


class CardFrame(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet("""
            QFrame#card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 20px;
            }
        """)
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(18, 18, 18, 18)
        self.layout_main.setSpacing(12)

        if title:
            lbl = QLabel(title)
            lbl.setStyleSheet("""
                QLabel {
                    color: #0f172a;
                    font-size: 18px;
                    font-weight: 700;
                }
            """)
            self.layout_main.addWidget(lbl)


# =========================
# 主界面
# =========================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.engine = InferenceEngine()
        self.current_image_path: Optional[str] = None
        self.current_result_image: Optional[Image.Image] = None

        self.setWindowTitle("宫庙建筑要素识别系统")
        self.resize(1450, 860)
        self.setMinimumSize(1280, 760)
        self.setAcceptDrops(True)
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        header = CardFrame()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(14)

        title_block = QVBoxLayout()
        title = QLabel("宫庙建筑要素识别系统")
        title.setStyleSheet("""
            QLabel {
                color: #0f172a;
                font-size: 30px;
                font-weight: 800;
            }
        """)
        subtitle = QLabel("支持塌寿、屋顶、开间、瓦片、屋脊装饰等任务的单图推理展示\n新增：YOLO检测+分类、屋脊装饰实例分割识别")
        subtitle.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.device_label = QLabel(f"运行设备：{'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.device_label.setStyleSheet("""
            QLabel {
                background: #eff6ff;
                color: #1d4ed8;
                border: 1px solid #bfdbfe;
                border-radius: 16px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 700;
            }
        """)
        self.device_label.setFixedHeight(42)

        hbox.addLayout(title_block, 1)
        hbox.addWidget(self.device_label, 0, Qt.AlignRight | Qt.AlignVCenter)
        header.layout_main.addLayout(hbox)
        root.addWidget(header, 0)

        mid = QHBoxLayout()
        mid.setSpacing(16)

        # 左侧：图片展示
        left_card = CardFrame("输入/检测结果展示")

        self.view_switch_btn = QPushButton("查看检测结果")
        self.view_switch_btn.setEnabled(False)
        self.view_switch_btn.setStyleSheet("""
            QPushButton {
                background: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e2e8f0;
            }
            QPushButton:disabled {
                background: #f1f5f9;
                color: #94a3b8;
            }
        """)
        self.view_switch_btn.clicked.connect(self.toggle_result_view)

        self.preview = ImagePreview("图片预览区域")
        self.is_showing_result = False

        self.path_label = QLabel("当前图片：未选择")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("""
            QLabel {
                color: #475569;
                font-size: 13px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 10px 12px;
            }
        """)

        left_card.layout_main.addWidget(self.view_switch_btn, 0)
        left_card.layout_main.addWidget(self.preview, 1)
        left_card.layout_main.addWidget(self.path_label, 0)
        mid.addWidget(left_card, 3)

        right_wrap = QVBoxLayout()
        right_wrap.setSpacing(16)

        task_card = CardFrame("模型与任务")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        lbl_task = QLabel("识别任务")
        lbl_task.setStyleSheet("font-size:14px;font-weight:700;color:#334155;")
        self.task_combo = QComboBox()
        self.task_combo.addItems(["塌寿三分类", "屋顶四分类", "开间分类", "瓦片分类", "YOLO检测+分类", "屋脊装饰识别", "建筑主体区域识别", "建筑屋顶区域识别"])
        self.task_combo.setCurrentIndex(0)

        lbl_model = QLabel("权重路径")
        lbl_model.setStyleSheet("font-size:14px;font-weight:700;color:#334155;")
        self.model_path_show = QLabel(MODEL_PATHS[self.task_combo.currentText()])
        self.model_path_show.setWordWrap(True)
        self.model_path_show.setStyleSheet("""
            QLabel {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 10px 12px;
                color: #475569;
                font-size: 12px;
            }
        """)

        self.load_btn = QPushButton("加载当前模型")
        self.open_btn = QPushButton("选择图片")
        self.run_btn = QPushButton("开始识别")

        grid.addWidget(lbl_task, 0, 0)
        grid.addWidget(self.task_combo, 0, 1)
        grid.addWidget(lbl_model, 1, 0)
        grid.addWidget(self.model_path_show, 1, 1)
        grid.addWidget(self.load_btn, 2, 0)
        grid.addWidget(self.open_btn, 2, 1)
        grid.addWidget(self.run_btn, 3, 0, 1, 2)

        task_card.layout_main.addLayout(grid)

        self.status_label = QLabel("模型状态：未加载")
        self.status_label.setStyleSheet("""
            QLabel {
                background: #fff7ed;
                color: #c2410c;
                border: 1px solid #fed7aa;
                border-radius: 12px;
                padding: 10px 12px;
                font-size: 13px;
                font-weight: 700;
            }
        """)
        task_card.layout_main.addWidget(self.status_label)
        right_wrap.addWidget(task_card)

        result_card = CardFrame("预测结果")
        self.pred_label = QLabel("--")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setMinimumHeight(90)
        self.pred_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eff6ff,
                    stop:1 #eef2ff
                );
                color: #1e3a8a;
                border: 1px solid #c7d2fe;
                border-radius: 18px;
                font-size: 26px;
                font-weight: 800;
            }
        """)

        self.conf_label = QLabel("置信度：--")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet("""
            QLabel {
                color: #475569;
                font-size: 16px;
                font-weight: 700;
            }
        """)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)

        result_card.layout_main.addWidget(self.pred_label)
        result_card.layout_main.addWidget(self.conf_label)
        result_card.layout_main.addWidget(self.progress)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMinimumHeight(220)
        self.detail_text.setPlaceholderText("这里展示各类别概率和模型信息。")
        result_card.layout_main.addWidget(self.detail_text)

        right_wrap.addWidget(result_card, 1)

        mid.addLayout(right_wrap, 2)
        root.addLayout(mid, 1)

        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
            }
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 12, 18, 12)

        self.footer_left = QLabel("提示：请先加载模型，再选择单张图片进行识别。")
        self.footer_left.setStyleSheet("""
            QLabel {
                color: #475569;
                font-size: 13px;
                font-weight: 600;
            }
        """)

        self.footer_right = QLabel(" ")
        self.footer_right.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 12px;
                font-weight: 600;
            }
        """)

        footer_layout.addWidget(self.footer_left, 1)
        footer_layout.addWidget(self.footer_right, 0, Qt.AlignRight)
        root.addWidget(footer, 0)

        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        self.load_btn.clicked.connect(self.load_current_model)
        self.open_btn.clicked.connect(self.select_image)
        self.run_btn.clicked.connect(self.run_predict)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background: #f3f6fb;
                font-family: "Microsoft YaHei", "Segoe UI";
            }

            QComboBox {
                background: white;
                border: 1px solid #dbe2ea;
                border-radius: 12px;
                padding: 10px 12px;
                min-height: 20px;
                font-size: 14px;
                color: #0f172a;
                font-weight: 600;
            }

            QComboBox::drop-down {
                border: none;
                width: 24px;
            }

            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2563eb,
                    stop:1 #4f46e5
                );
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 700;
            }

            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1d4ed8,
                    stop:1 #4338ca
                );
            }

            QPushButton:pressed {
                background: #1e40af;
            }

            QTextEdit {
                background: #fbfdff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 8px;
                color: #0f172a;
                font-size: 13px;
            }

            QProgressBar {
                background: #e2e8f0;
                border: none;
                border-radius: 5px;
            }

            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6,
                    stop:1 #8b5cf6
                );
                border-radius: 5px;
            }
        """)

    def toggle_result_view(self):
        """切换显示原图和检测结果"""
        if self.current_result_image is None:
            return

        self.is_showing_result = not self.is_showing_result

        if self.is_showing_result:
            self.preview.set_pil_image(self.current_result_image)
            self.view_switch_btn.setText("查看原图")
        else:
            if self.current_image_path:
                img = Image.open(self.current_image_path).convert("RGB")
                self.preview.set_pil_image(img)
            self.view_switch_btn.setText("查看检测结果")

    def on_task_changed(self, task_name: str):
        self.model_path_show.setText(MODEL_PATHS.get(task_name, ""))
        self.engine.clear_task_cache(task_name)

        self.status_label.setText("模型状态：未加载")
        self.status_label.setStyleSheet("""
            QLabel {
                background: #fff7ed;
                color: #c2410c;
                border: 1px solid #fed7aa;
                border-radius: 12px;
                padding: 10px 12px;
                font-size: 13px;
                font-weight: 700;
            }
        """)
        self.detail_text.clear()
        self.pred_label.setText("--")
        self.conf_label.setText("置信度：--")
        self.progress.setValue(0)
        # 重置视图切换状态
        self.view_switch_btn.setEnabled(False)
        self.view_switch_btn.setText("查看检测结果")
        self.is_showing_result = False
        self.current_result_image = None
        self.footer_left.setText(f"当前任务已切换为：{task_name}")

    def load_current_model(self):
        task_name = self.task_combo.currentText()
        self.footer_left.setText(f"正在加载：{task_name}")
        QApplication.processEvents()

        try:
            self.engine.clear_task_cache(task_name)
            loaded = self.engine.load_model(task_name)

            self.status_label.setText(f"模型状态：已加载成功 | {loaded.extra_info}")
            self.status_label.setStyleSheet("""
                QLabel {
                    background: #ecfdf5;
                    color: #047857;
                    border: 1px solid #a7f3d0;
                    border-radius: 12px;
                    padding: 10px 12px;
                    font-size: 13px;
                    font-weight: 700;
                }
            """)

            normalize_desc = "是" if loaded.normalize else "否"
            pad_desc = str(loaded.pad_bg_color)

            info_text = (
                f"任务名称：{loaded.task_name}\n"
                f"输入尺寸：{loaded.input_size}\n"
                f"运行设备：{loaded.device}\n"
                f"类别列表（英文）：{', '.join(loaded.labels)}\n"
                f"类别列表（中文）：{', '.join([get_chinese_class_name(l) for l in loaded.labels])}\n"
                f"Normalize：{normalize_desc}\n"
                f"Pad背景色：{pad_desc}\n\n"
                f"{loaded.extra_info}"
            )

            # YOLO任务额外显示配置
            if task_name == "YOLO检测+分类" and loaded.yolo_config:
                config = loaded.yolo_config
                info_text += (
                    f"\n\nYOLO配置："
                    f"\n检测模型：{config.get('yolo_model_path', '未配置')}"
                    f"\n分类模型：{config.get('cls_model_arch', '未配置')} ({config.get('cls_model_path', '未配置')})"
                    f"\n分类类别（英文）：{', '.join(config.get('cls_labels', []))}"
                    f"\n分类类别（中文）：{', '.join([get_chinese_class_name(l) for l in config.get('cls_labels', [])])}"
                    f"\n置信度阈值：{config.get('conf_threshold', 0.25)}"
                )

            # 屋脊装饰识别任务显示YOLO配置
            if task_name == "屋脊装饰识别" and loaded.yolo_config:
                config = loaded.yolo_config
                info_text += (f"\n\n屋脊装饰识别配置：" f"\nYOLO实例分割模型：{config.get('yolo_model_path', '未配置')}" f"\n类别：{', '.join(config.get('labels', []))}" f"\n置信度阈值：{config.get('conf_threshold', 0.25)}" f"\nIOU阈值：{config.get('iou_threshold', 0.70)}" f"\n输入尺寸：{config.get('imgsz', 960)}")

            # 屋顶四分类任务显示YOLO配置
            if task_name == "屋顶四分类" and loaded.yolo_config:
                config = loaded.yolo_config
                yolo_path = config.get("yolo_model_path", "")
                if yolo_path and yolo_path.strip() != "":
                    backup_yolo_path = config.get("backup_yolo_model_path", "")
                    info_text += (
                        f"\n\n屋顶YOLO检测配置："
                        f"\n主检测模型（屋顶）：{yolo_path}"
                        f"\n主检测阈值：{config.get('conf_threshold', 0.25)}"
                        f"\n主IOU阈值：{config.get('iou_threshold', 0.45)}"
                        f"\n主目标类别：{config.get('target_class', 0)}"
                        f"\n备用检测模型（建筑主体）：{backup_yolo_path if backup_yolo_path else '未配置'}"
                        f"\n备用检测阈值：{config.get('backup_conf_threshold', 0.25)}"
                        f"\n备用IOU阈值：{config.get('backup_iou_threshold', 0.45)}"
                        f"\n备用目标类别：{config.get('backup_target_class', 0)}"
                        f"\n联动逻辑：主YOLO未检出 -> 备用主体YOLO裁剪 -> 主YOLO二次检测"
                        f"\n最终显示：仅显示主YOLO检测框"
                    )
                else:
                    info_text += "\n\n[警告] 主YOLO路径未配置，屋顶四分类无法执行检测链路"

            self.detail_text.setText(info_text)
            self.footer_left.setText(f"{task_name} 模型已加载完成")
        except Exception:
            self.status_label.setText("模型状态：加载失败")
            self.status_label.setStyleSheet("""
                QLabel {
                    background: #fef2f2;
                    color: #b91c1c;
                    border: 1px solid #fecaca;
                    border-radius: 12px;
                    padding: 10px 12px;
                    font-size: 13px;
                    font-weight: 700;
                }
            """)
            self.detail_text.setText(
                f"加载失败：\n\n详细报错：\n{pretty_exception()}"
            )
            self.footer_left.setText("模型加载失败，请检查权重路径或骨干结构")

    def load_image_from_path(self, file_path: str):
        if not file_path:
            return

        self.current_image_path = file_path
        self.path_label.setText(f"当前图片：{file_path}")
        self.footer_left.setText("图片已选择，可以开始识别")
        # 重置所有与结果相关的状态
        self.is_showing_result = False
        self.view_switch_btn.setEnabled(False)
        self.view_switch_btn.setText("查看检测结果")
        self.current_result_image = None

        try:
            img = Image.open(file_path).convert("RGB")
            self.preview.set_pil_image(img)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"图片读取失败：\n{e}")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not file_path:
            return
        self.load_image_from_path(file_path)

    def run_predict(self):
        task_name = self.task_combo.currentText()

        if self.current_image_path is None:
            QMessageBox.warning(self, "提示", "请先选择一张图片。")
            return

        self.progress.setValue(15)
        QApplication.processEvents()

        try:
            if task_name not in self.engine.cache:
                self.load_current_model()

            self.progress.setValue(45)
            QApplication.processEvents()

            # 调用predict接口
            result = self.engine.predict(task_name, self.current_image_path)
            pred_name, conf, topk, detail, result_img, chinese_pred = result

            self.progress.setValue(100)
            # 主预测结果显示中文
            self.pred_label.setText(chinese_pred)
            self.conf_label.setText(f"置信度：{conf * 100:.2f}%")

            # 如果有结果图，保存并启用切换按钮
            if result_img is not None:
                self.current_result_image = result_img
                self.view_switch_btn.setEnabled(True)
                self.is_showing_result = True
                self.preview.set_pil_image(result_img)
                self.view_switch_btn.setText("查看原图")
            else:
                self.view_switch_btn.setEnabled(False)
                self.is_showing_result = False
                # 如果没有结果图，显示原图
                if self.current_image_path:
                    img = Image.open(self.current_image_path).convert("RGB")
                    self.preview.set_pil_image(img)

            # 详情文本已在predict方法中构建好，直接显示
            self.detail_text.setText(detail)
            self.footer_left.setText(f"识别完成：{chinese_pred}")
        except Exception:
            self.progress.setValue(0)
            self.pred_label.setText("识别失败")
            self.conf_label.setText("置信度：--")
            self.detail_text.setText(
                f"推理失败：\n\n详细报错：\n{pretty_exception()}"
            )
            self.footer_left.setText("识别失败，请查看右侧详细报错信息")
            self.view_switch_btn.setEnabled(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                local_path = urls[0].toLocalFile().lower()
                if local_path.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                local_path = urls[0].toLocalFile()
                if local_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    self.load_image_from_path(local_path)
                    event.acceptProposedAction()
                    return
        event.ignore()


# =========================
# 主程序入口
# =========================
def check_paths():
    lines = []
    ok = True
    for task, p in MODEL_PATHS.items():
        if task == "YOLO检测+分类":
            continue
        exists = os.path.exists(p)
        lines.append(f"{task}: {'存在' if exists else '不存在'} -> {p}")
        if not exists:
            ok = False
    return ok, "\n".join(lines)


def main():
    set_global_determinism(TASHOU_INFER_CONFIG.get("seed", 3407))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    ok, msg = check_paths()
    if not ok:
        print("以下权重路径不存在，请先调整目录：")
        print(msg)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

