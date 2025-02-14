from copy import copy

import cv2
import os
import numpy as np

def to_grayscale(image, inv=False):

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if inv:
        gray_image = 255 - gray_image

    return gray_image

def correlation(whole_image, searched_image):
    image_width, image_height = whole_image.shape
    result = np.fft.ifft2(np.fft.fft2(whole_image) * np.fft.fft2(np.rot90(np.rot90(searched_image)), s=(image_width, image_height))).real
    result /= np.abs(np.max(result))
    return result

def highlight_pattern(np_image_array, corr, certainty, highlight_size, pattern_name, locations):

    pattern_mask_pwd =f"font_mask/{pattern_name}.png"
    letter_mask = to_grayscale(cv2.imread(pattern_mask_pwd), False)
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
        pattern_size = to_grayscale(image=cv2.imread(os.path.join(filename)), inv=True).shape

        if xses[i+1] - xses[i] - pattern_size[1] > 0:
            locations.append((' ', xses[i+1]-pattern_size[1], 0, 1))

def ocr(ocr_filename, inv, save_preprocessing_data = False):

    locations = []

    whole_in_grey = np.array(to_grayscale(image=cv2.imread(os.path.join(ocr_filename)), inv=inv))

    alphabet = np.array(to_grayscale(image=cv2.imread(os.path.join("font/alphabet_27.png")), inv=True))

    combined_image = cv2.vconcat([whole_in_grey, alphabet])

    cv2.imwrite("overlayed_image.png", combined_image)

    if save_preprocessing_data:
        os.makedirs("preprocessed_data", exist_ok=True)
        cv2.imwrite("preprocessed_data/overlayed_image.png", combined_image)

    alphabet_best = [
        'r',
        'a',
        'b',
        'd',
        'e',
        'f',
        'g',
        'k',
        'p',
        'q',
        's',
        't',
        'w',
        'x',
        'y',
        'z',
        'c',
        'v',
        'h',
        'l',
        'm',
        'n',
        'u',
        'o',
        'j',
        'i'
    ]

    certainties = {
        'a': 0.93,
        'b': 0.93,
        'c': 0.93,
        'd': 0.93,
        'e': 0.93,
        'f': 0.92,
        'g': 0.93,
        'h': 0.93,
        'i': 0.95,
        'j': 0.93,
        'k': 0.93,
        'l': 0.90,
        'm': 0.95,
        'n': 0.93,
        'o': 0.93,
        'p': 0.93,
        'q': 0.93,
        'r': 0.93,
        's': 0.87,
        't': 0.95,
        'u': 0.93,
        'v': 0.89,
        'w': 0.93,
        'x': 0.93,
        'y': 0.90,
        'z': 0.93,
    }

    for char in alphabet_best:

        whole = to_grayscale(cv2.imread(os.path.join("overlayed_image.png")), inv=inv)

        filename = "font/" + char + ".png"

        pattern = to_grayscale(image=cv2.imread(os.path.join(filename)), inv=True)

        highlight_pattern(np_image_array=np.asarray(whole),
                          corr=correlation(whole_image=to_grayscale(whole), searched_image=pattern), certainty=certainties[char],
                          highlight_size=(8, 8), pattern_name=char, locations=locations)

    os.remove("overlayed_image.png")

    locations.sort(key=lambda x: x[1])

    for tup in locations[:]:
        _, _, y, _ = tup
        if y > 40:
            locations.remove(tup)

    detect_space(locations)
    locations.sort(key=lambda x: x[1])

    string = ""
    for name, _, _, _ in locations:
        string += name

    return string