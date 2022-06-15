import cv2
import numpy as np
import pytesseract
import imutils

img = cv2.imread("image303.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered =cv2.bilateralFilter(gray, 6, 250, 250)
edged = cv2.Canny(filtered, 30, 200 )

contours = cv2.findContours(edged,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

screen = None

for c in cnts:
    epsilon = 0.018 * cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 4:
        screen = approx
        break

mask = np.zeros(gray.shape, np.uint8)

new_img = cv2.drawContours(mask, [screen], 0, (255,255,255), -1)

new_image = cv2.bitwise_and(img, img, mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))

cropped = gray[topx:bottomx+1, topy:bottomy+1]


cv2.imshow("Licence Plate", img)
cv2.imshow("Licence Plate Gray", gray)
cv2.imshow("Licence Plate Filter", filtered)
cv2.imshow("Licence Plate Edged", edged)
cv2.imshow("Licence Plate Mask", mask)
cv2.imshow("Licence Plate Cropped", cropped)


cv2.imwrite("Plate.jpg",img)
cv2.imwrite("LicencePlateGray.jpg",gray )
cv2.imwrite("LicencePlateFilter.jpg",filtered)
cv2.imwrite("LicencePlateEdged.jpg",edged)
cv2.imwrite("LicencePlateMask.jpg",mask)
cv2.imwrite("LicencePlateCropped.jpg",cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

text = pytesseract.image_to_string(cropped, lang=("eng"))
print(f"Detected Text: {text}")