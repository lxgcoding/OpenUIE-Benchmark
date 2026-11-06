
import numpy as np
from PIL import Image, ImageFilter

from open_uie.models.base import BaseEnhancementModel


class UDCP(BaseEnhancementModel):
    """
    Underwater Dark Channel Prior (UDCP).

    输入：RGB uint8 图像，形状为 (H, W, 3)
    输出：RGB uint8 增强图像，形状为 (H, W, 3)

    使用绿色、蓝色通道计算水下暗通道，
    估计背景光和透射率，再恢复图像。
    """

    def __init__(
        self,
        omega=0.95,
        patch_size=15,
        t_min=0.1,
        top_percent=0.001
    ):
        super().__init__(model_name="UDCP")

        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd number.")

        if not 0 < omega <= 1:
            raise ValueError("omega must be in (0, 1].")

        if not 0 < t_min <= 1:
            raise ValueError("t_min must be in (0, 1].")

        if not 0 < top_percent <= 1:
            raise ValueError("top_percent must be in (0, 1].")

        self.omega = omega
        self.patch_size = patch_size
        self.t_min = t_min
        self.top_percent = top_percent

    def _min_filter(self, image):
        """计算局部最小值滤波。"""

        pil_image = Image.fromarray(
            np.clip(image * 255, 0, 255).astype(np.uint8)
        )

        filtered = pil_image.filter(
            ImageFilter.MinFilter(self.patch_size)
        )

        return np.asarray(filtered).astype(np.float32) / 255.0

    def _get_dark_channel(self, image):
        """使用绿色和蓝色通道计算水下暗通道。"""

        green = image[:, :, 1]
        blue = image[:, :, 2]

        min_channel = np.minimum(green, blue)

        return self._min_filter(min_channel)

    def _estimate_background_light(self, image, dark_channel):
        """估计背景光。"""

        h, w = dark_channel.shape

        pixel_count = max(
            1,
            int(h * w * self.top_percent)
        )

        flat_dark = dark_channel.reshape(-1)

        indices = np.argpartition(
            flat_dark,
            -pixel_count
        )[-pixel_count:]

        # 在暗通道值最大的候选像素中，
        # 选择绿色和蓝色通道亮度最大的像素。
        candidates = image.reshape(-1, 3)[indices]

        brightness = np.max(candidates[:, 1:3], axis=1)

        background = candidates[np.argmax(brightness)]

        return np.maximum(background, 1e-3)

    def predict(self, image):
        """执行 UDCP 水下图像增强。"""

        image = image.astype(np.float32) / 255.0

        # 1. 计算水下暗通道
        dark_channel = self._get_dark_channel(image)

        # 2. 估计背景光
        background = self._estimate_background_light(
            image,
            dark_channel
        )

        # 3. 计算归一化暗通道
        normalized_image = image / background.reshape(1, 1, 3)

        normalized_dark = self._get_dark_channel(
            normalized_image
        )

        # 4. 估计透射率
        transmission = 1.0 - self.omega * normalized_dark

        transmission = np.maximum(
            transmission,
            self.t_min
        )

        # 5. 恢复增强图像
        restored = (
            image - background.reshape(1, 1, 3)
        ) / transmission[:, :, None] + background.reshape(1, 1, 3)

        # 6. 限制数值范围并转换为 uint8
        restored = np.clip(restored, 0.0, 1.0)

        return (restored * 255.0).round().astype(np.uint8)
