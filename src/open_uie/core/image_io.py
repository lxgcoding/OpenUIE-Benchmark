
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(image_path, mode="RGB"):
    """
    读取图像，并统一转换为 NumPy 数组。

    Args:
        image_path: 图像文件路径。
        mode: RGB 或 L（灰度图）。

    Returns:
        RGB: (H, W, 3)
        L:   (H, W)
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if mode not in ("RGB", "L"):
        raise ValueError("mode must be 'RGB' or 'L'.")

    with Image.open(image_path) as image:
        image = image.convert(mode)
        image_array = np.asarray(image, dtype=np.uint8)

    return image_array


def to_uint8(image):
    """
    将输入图像转换为 uint8 格式。

    支持：
        - [0, 1] 浮点图像
        - [0, 255] 数值图像
        - NumPy 数组
    """
    image = np.asarray(image)

    if image.ndim not in (2, 3):
        raise ValueError("Image must be a 2D or 3D array.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf values.")

    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and image.min() >= 0 and image.max() <= 1:
            image = image * 255.0

    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def save_image(image, save_path):
    """
    保存图像。

    输入图像默认采用 RGB 通道顺序。
    支持 RGB、RGBA 和灰度图。
    """
    image = to_uint8(image)

    if image.ndim == 2:
        output = Image.fromarray(image)

    elif image.shape[2] == 3:
        output = Image.fromarray(image)

    elif image.shape[2] == 4:
        output = Image.fromarray(image)

    else:
        raise ValueError(
            "Image must have 1, 3, or 4 channels."
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output.save(save_path)

    return str(save_path)
