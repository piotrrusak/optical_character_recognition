import cv2
import numpy as np
import os
from PIL import Image

# Loop through each character from 'a' to 'z'
for char in [chr(i) for i in range(ord("a"), ord("z") + 1)]:
    # Define the font and mask file paths
    font_filename = "font/" + char + ".png"
    mask_filename = "font_mask/" + char + ".png"

    # Open the font image
    image = Image.open(font_filename)

    # Create a white mask of the same size as the image
    mask = np.ones((image.size[1], image.size[0]), dtype=np.uint8)

    # Save the mask as a PNG file
    cv2.imwrite(mask_filename, mask)
