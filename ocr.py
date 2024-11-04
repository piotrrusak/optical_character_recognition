from copy import copy

import cv2
import os
import numpy as np
from numpy import rot90
from numpy.fft import fft2, ifft2
from PIL import Image, ImageOps

def to_grayscale(image, inv=False):
    if image.mode != 'L':  # 'L' stands for grayscale
        image = image.convert('L')
    if inv:
        image = ImageOps.invert(image)
    return image

def correlation(whole_image, searched_image):
    image_width, image_height = whole_image.size
    result = ifft2(fft2(whole_image) * fft2(rot90(rot90(searched_image)), s=(image_height, image_width))).real
    result /= np.abs(np.max(result))
    return result

def highlight_pattern(np_image_array, corr, certainty, highlight_size, pattern_name, locations):

    pattern_mask_pwd =f"font_mask/{pattern_name}.png"
    letter_mask = cv2.cvtColor(cv2.imread(pattern_mask_pwd), cv2.COLOR_BGR2GRAY)
    new_np_image_array = copy(np_image_array)

    found_locations = []
    for y in range(corr.shape[0]):
        for x in range(corr.shape[1]):
            if corr[y, x] > certainty and y != 0:
                found_locations.append((y, x))
                locations.append((pattern_name, x, y, corr[y, x]))


    for y, x in found_locations:
        top_left = (x - abs(highlight_size[0]) // 2, y - abs(highlight_size[1]) // 2)
        bottom_right = (x + abs(highlight_size[0]) // 2, y + abs(highlight_size[1]) // 2)

        cv2.rectangle(np_image_array, top_left, bottom_right, (0, 0, 0), 2)

        max_height = y
        max_width = x

        mask_height, mask_width = letter_mask.shape[:2]
        insert_height = min(mask_height, max_height)
        insert_width = min(mask_width, max_width)

        cropped_letter_mask = letter_mask[mask_height - insert_height:, mask_width - insert_width:]

        new_np_image_array[y - insert_height:y, x - insert_width:x] = cropped_letter_mask

    output_path = f"highlights/highlight_{pattern_name}.png"
    cv2.imwrite(output_path, np_image_array)

    cv2.imwrite("overlayed_image.png", new_np_image_array)

def detect_space(locations):

    xses = []

    for _, x, y, _ in locations:
        xses.append(x)

    for i in range(len(xses)-1):

        type, _, _, _ = locations[i+1]
        filename = "font/" + type + ".png"
        pattern_size = to_grayscale(image=Image.open(os.path.join(filename)), inv=True).size

        if xses[i+1] - xses[i] - pattern_size[0] > 0:
            locations.append((' ', xses[i+1]-pattern_size[0], 0, 1))

locations = []

whole_in_grey = np.array(to_grayscale(image=Image.open(os.path.join("images/otto.png")), inv=True))

alphabet = np.array(to_grayscale(image=Image.open(os.path.join("alphabet_27.png")), inv=True))

combined_image = cv2.vconcat([whole_in_grey, alphabet])

cv2.imwrite("overlayed_image.png", combined_image)

alphabet_best = ['r', 'a', 'b', 'd', 'e', 'f', 'g', 'k', 'p', 'q', 's', 't', 'w', 'x', 'y', 'z', 'c', 'v', 'h', 'l', 'm', 'n', 'u', 'o', 'j', 'i']

for char in alphabet_best:

    whole = Image.open(os.path.join("overlayed_image.png"))

    filename = "font/" + char + ".png"

    pattern = to_grayscale(image=Image.open(os.path.join(filename)), inv=True)

    if char == 'j':
        a=1

    if char == 'f':
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.92,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)
    elif char == 's':
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.87,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)
    elif char == 'i':
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.88,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)
    elif char == 'm':
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.95,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)
    elif char == 'v':
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.89,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)
    elif char == 'y':
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.90,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)
    else:
        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=0.93,
                          highlight_size=(8, 8), pattern_name=char, locations=locations)

locations.sort(key=lambda x: x[1])

# Usuwa litery wykryte z doklejonego alfabetu
for tup in locations[:]:
    _, _, y, _ = tup
    if y > 40:
        locations.remove(tup)

# Wykrywa i dokleja spacje
detect_space(locations)
locations.sort(key=lambda x: x[1])

# Printuje napis końcowy
string = ""
for name, _, _, _ in locations:
    string += name
print(string)
