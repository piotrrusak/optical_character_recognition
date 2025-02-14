import cv2
import numpy as np

for char in [chr(i) for i in range(ord("a"), ord("z") + 1)]:
    font_filename = "font/" + char + ".png"
    mask_filename = "font_mask/" + char + ".png"

    image = cv2.imread(font_filename)

    shape = image.shape

    mask = np.ones((shape[0], shape[0]), dtype=np.uint8)

    cv2.imwrite(mask_filename, mask)