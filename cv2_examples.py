import cv2
import numpy as np

# 1. Odczyt obrazu
# ----------------
# Odczyt obrazu z pliku.
image1 = cv2.imread('we_shall_fight_on_the_beaches.png')
image2 = cv2.imread('we_shall_fight_on_the_beaches.png')

# Sprawdzenie czy obrazy zostały poprawnie załadowane
if image1 is None or image2 is None:
    print("Nie można wczytać jednego z obrazów!")
    exit()

# 2. Wyświetlanie obrazów
cv2.imshow('Obraz 1', image1)
cv2.imshow('Obraz 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Konwersja obrazu do skali szarości
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow('Skala Szarości Obraz 1', gray_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Zapis obrazu
cv2.imwrite('gray_image.jpg', gray_image1)

# 5. Wykrywanie krawędzi za pomocą algorytmu Canny
edges = cv2.Canny(gray_image1, 100, 200)
cv2.imshow('Krawędzie (Canny)', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Rysowanie prostokątów i kół na obrazach
image_with_shapes = image1.copy()
cv2.rectangle(image_with_shapes, (50, 50), (200, 200), (0, 255, 0), 3)
cv2.circle(image_with_shapes, (300, 300), 50, (255, 0, 0), -1)
cv2.imshow('Rysowanie na Obrazie', image_with_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Łączenie obrazów (horyzontalne i wertykalne)
def concatenate_images(img1, img2):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h_concat = np.hstack((img1, img2_resized))  # Horyzontalne
    v_concat = np.vstack((img1, img2_resized))  # Wertykalne
    return h_concat, v_concat

h_concat, v_concat = concatenate_images(image1, image2)
cv2.imshow('Połączone Horyzontalnie', h_concat)
cv2.imshow('Połączone Wertykalnie', v_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. Łączenie kanałów kolorów
b, g, r = cv2.split(image1)
b = cv2.add(b, 50)
merged_image = cv2.merge([b, g, r])
cv2.imshow('Zmodyfikowany Kanał Niebieski', merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 9. Operacje morfologiczne: Erozja, Dylacja, Otwieranie, Zamknięcie
gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)

eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
dilated_image = cv2.dilate(thresh_image, kernel, iterations=1)
opening_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
closing_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Erozja', eroded_image)
cv2.imshow('Dylacja', dilated_image)
cv2.imshow('Otwieranie', opening_image)
cv2.imshow('Zamknięcie', closing_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 10. Maskowanie i operacje bitowe
mask = np.zeros(image1.shape[:2], dtype="uint8")
cv2.circle(mask, (image1.shape[1]//2, image1.shape[0]//2), 100, 255, -1)
masked_image = cv2.bitwise_and(image1, image1, mask=mask)
cv2.imshow('Obraz z Maską', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 11. Transformacja perspektywiczna
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [240, 200]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped_image = cv2.warpPerspective(image1, matrix, (300, 300))
cv2.imshow('Zniekształcony Obraz', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 12. Wykrywanie konturów
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image1.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
cv2.imshow('Obraz z Konturami', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 13. Segmentacja obrazu za pomocą Watershed
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(image1, markers)
image1[markers == -1] = [0, 0, 255]
cv2.imshow('Segmentacja Watershed', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 14. Obrót obrazu
height, width = image1.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image1, rotation_matrix, (width, height))
cv2.imshow('Obrócony Obraz', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
