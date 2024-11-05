from tabnanny import check

import cv2
import os
from PIL import Image, ImageOps
import numpy as np

import ocr

def to_grayscale(image, inv=False):
    if image.mode != 'L':  # 'L' stands for grayscale
        image = image.convert('L')
    if inv:
        image = ImageOps.invert(image)
    return image

def check_if_line_starts(np_array_image):
    if np_array_image is None:
        return None
    for i in range(np_array_image.shape[0]):
        for j in range(np_array_image.shape[1]):
            if np_array_image[i][j] > 0:
                return i - 1

def chop_off_the_line(np_array_image):
    first_black_line = check_if_line_starts(np_array_image)
    if first_black_line is None:
        return None, None
    return np_array_image[first_black_line:first_black_line + 32], np_array_image[first_black_line + 32:np_array_image.shape[0]]

lines = 0

def chop_all_lines(np_array_image, lines):
    i = 0
    if check_if_line_starts(np_array_image):
        chopped_line, rest_of_image = chop_off_the_line(np_array_image)
        cv2.imwrite("chopped0.png", chopped_line)
        cv2.imwrite("rest.png", rest_of_image)
        i += 1
    else:
        print("This is a blank")
        return
    while check_if_line_starts(rest_of_image) is not None:
        chopped_line, rest_of_image = chop_off_the_line(rest_of_image)

        if chopped_line is not None and chopped_line.size > 0:
            cv2.imwrite(f"chopped{i}.png", chopped_line)

        if rest_of_image is not None and rest_of_image.size > 0:
            cv2.imwrite("rest.png", rest_of_image)

        i += 1

    lines = i

    return lines

image = Image.open(os.path.join("images/otto.png"))

# chopped_line, rest_of_image = chop_off_the_line(np.asarray(to_grayscale(image, inv=True)))
#
# cv2.imwrite("chopped.png", chopped_line)
# cv2.imwrite("rest.png", rest_of_image)

lines = chop_all_lines(np.asarray(to_grayscale(image, inv=True)), lines)

for i in range(lines):
    ocr.ocr("chopped" + str(i) + ".png", inv=False)
