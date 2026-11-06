
from abc import ABC, abstractmethod

import numpy as np


class BaseEnhancementModel(ABC):
    """
    所有水下图像增强模型的统一基类。

    统一约定：
        输入：RGB 格式的 uint8 NumPy 数组，形状为 (H, W, 3)
        输出：RGB 格式的 uint8 NumPy 数组，形状为 (H, W, 3)
    """

    def __init__(self, model_name="BaseModel"):
        self.model_name = model_name

    @abstractmethod
    def predict(self, image):
        """
        执行图像增强。

        子类必须实现这个方法。
        """
        raise NotImplementedError

    def validate_input(self, image):
        """检查输入图像是否符合统一规范。"""

        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input image must have shape (H, W, 3)."
            )

        if image.dtype != np.uint8:
            raise TypeError(
                "Input image must have dtype uint8."
            )

        return True

    def validate_output(self, image):
        """检查模型输出是否符合统一规范。"""

        if not isinstance(image, np.ndarray):
            raise TypeError("Output must be a NumPy array.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Output image must have shape (H, W, 3)."
            )

        if image.dtype != np.uint8:
            raise TypeError(
                "Output image must have dtype uint8."
            )

        return True

    def __call__(self, image):
        """允许直接通过 model(image) 调用模型。"""

        self.validate_input(image)

        result = self.predict(image)

        self.validate_output(result)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.model_name}')"
