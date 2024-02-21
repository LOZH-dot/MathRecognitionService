import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'F:\TesseractOCR\tesseract.exe'


img = cv2.imread("testImage.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))

dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

im2 = img.copy()

file = open("recognized.txt", "w+")
file.write("")
file.close()

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 1

lastCnt = (0, 0 ,0, 0)

i = 1;
for cnt in sorted_ctrs:
    x, y, w, h = cv2.boundingRect(cnt)

    posX = ''
    posY = ''

    if lastCnt[0] > x:
        posX = 'left'
    elif lastCnt[0] == x:
        posX = 'none'
    else:
        posX = 'right'

    if lastCnt[1] > y:
        posY = 'up'
    elif lastCnt[1] == y:
        posY = 'none'
    else:
        posY = 'down'

    if i == 1:
        posX = 'first'
        posY = ''

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #im2 = cv2.putText(im2, '{} {} {}'.format(i, posX, posY), (x, y), fontFace=font, fontScale=fontScale, color=color, thickness=thickness)
    im2 = cv2.putText(im2, '{}'.format(i), (x, y), fontFace=font, fontScale=fontScale, color=color, thickness=thickness)


    # Cropping the text block for giving input to OCR
    cropped = img[y:y + h, x:x + w]
    cv2.imshow('crop', cropped)
    cv2.waitKey(0)

    # Open the file in append mode
    file = open("recognized.txt", "a")

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped, lang='eng+equ', config='--psm 6')
    print('Контур: {} - текст: {}'.format(i, text))

    # Appending the text into file
    file.write(text)
    file.write("\n")

    # Close the file
    file.close()
    i += 1
    lastCnt = (x, y, w, h)

scale_percent = 150 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(im2, dim, interpolation = cv2.INTER_AREA)

cv2.imshow('img', resized)
cv2.waitKey(0)
