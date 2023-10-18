import numpy as np
import cv2 as cv
import skimage


class ImageBGR:

    def __init__(self, file: str = None, image: np.ndarray = None):
        if file is not None:
            self.__image = skimage.io.imread(file)
        elif image is not None:
            self.__image = image
        else:
            raise AttributeError("Incorrect value")

    #There's a little mistake. Skimage.io loads the image in RGB, so to make it consistent you have to convert it.

    def gray(self) -> np.ndarray:
        return cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)

    def lab(self) -> np.ndarray:
        return cv.cvtColor(self.__image, cv.COLOR_BGR2LAB)

    def rgb(self) -> np.ndarray:
        return cv.cvtColor(self.__image, cv.COLOR_BGR2RGB)

    def bgr(self) -> np.ndarray:
        return cv.cvtColor(self.__image, cv.COLOR_RGB2BGR)

    def resize(self, width: int, height: int) -> 'ImageBGR':
        resized_image = cv.resize(self.__image, (width, height))
        new_instance = ImageBGR(image=resized_image)
        return new_instance

    def rotate(self, angle: int, keep_ratio: bool) -> 'ImageBGR':
        if keep_ratio:
            (h, w) = self.__image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)  # center of operation, rotation, scale
            return ImageBGR(image=cv.warpAffine(self.__image, M, (w, h)))

        (h, w) = self.__image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv.getRotationMatrix2D((cX, cY), -45, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return ImageBGR(image=cv.warpAffine(self.__image, M, (nW, nH)))

    def histogram(self) -> np.ndarray:
        return cv.calcHist([self.__image], [0], None, [256], [0, 256])

    @property
    def shape(self) -> tuple:
        return self.__image.shape

    @property
    def size(self) -> int:
        return self.__image.itemsize
