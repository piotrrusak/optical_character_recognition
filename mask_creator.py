import cv2
import numpy as np
from PIL import Image

for char in [chr(i) for i in range(ord("a"), ord("z") + 1)]:
    font_filename = "font/" + char + ".png"
    mask_filename = "font_mask/" + char + ".png"

    image = Image.open(font_filename)

    mask = np.ones((image.size[1], image.size[0]), dtype=np.uint8)

    cv2.imwrite(mask_filename, mask)
