import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
# import easyocr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# the steps:
#        1) read the image and get its gray scaled image
#        2) apply any filter to remove the noise
#        3) use canny edge detection to get the edges of the image
#        4) find the Contours from the image
#        5) sort them in descending order according to their area to get the biggest 10 objects
#        6) get the location of the biggest object has 4 sides (quadrilateral as rectangle)
#           because our licence plate is rectangular shape
#        7) create mask and apply it to make only the plate appear
#        8) crop the plate from the image and use ocr to get the text





# Read the image, then get its grayscale
img = cv2.imread('image4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

#Apply filter to reduce noice
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction

#I tried gaussian filter but it didn't help as bilateralFilter
#filtered_img = cv2.GaussianBlur(bfilter, (5, 5), 2) #Noise reduction



#use canny edge detector to detect edges
edged = cv2.Canny(bfilter, 1, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

# find the contors of the image
keypoints = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
#print("the length of contours: " , len(contours))

#sort them decendingly according to thier area to get the biggest 10 objects
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]


# get the location of the biggest(area) contour(object)
# that is quadrilaterals(has 4 sides as rectangle because our licence plate are rectangular)
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 20, True)
    if len(approx) == 4: #we used 4 because the plat is rectangular
        location = approx
        break


#make mask and use it to get only the plat from the image using
#the location that we got from the previous few lines
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()


#crop and get the plat from the image
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = img[x1:x2+1, y1:y2+1]


#get the plat number as a text
text = pytesseract.image_to_string(cropped_image)
print(text[:-1])

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("the plate number is: "+text[:-1],fontweight="bold")
plt.show()
