import cv2
import os
import numpy as np

from ocr import ocr

def to_grayscale(image, inv=False):

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if inv:
        gray_image = 255 - gray_image

    return gray_image

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

def chop_all_lines(np_array_image):
    i = 0
    if check_if_line_starts(np_array_image):
        chopped_line, rest_of_image = chop_off_the_line(np_array_image)
        cv2.imwrite("lines/chopped0.png", chopped_line)
        i += 1
    else:
        print("This is a blank")
        return
    while check_if_line_starts(rest_of_image) is not None:
        chopped_line, rest_of_image = chop_off_the_line(rest_of_image)

        if chopped_line is not None and chopped_line.size > 0:
            cv2.imwrite(f"lines/chopped{i}.png", chopped_line)

        i += 1

    return i

image = cv2.imread(os.path.join("images/edison.png"))

lines = chop_all_lines(np.asarray(to_grayscale(image, inv=True)))

for i in range(lines):

    filename = "lines/chopped" + str(i) + ".png"

    print(ocr(filename, inv=False, save_preprocessing_data=True))
