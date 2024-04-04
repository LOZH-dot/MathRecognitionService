import cv2
import keras
import numpy as np
import tensorflow as tf


def segmentation(image_path):
    model = keras.models.load_model('model.keras')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 6))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    im2 = img.copy()

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

    lastCnt = (0, 0, 0, 0)

    i = 1
    merged_contour = contours[0]
    for cnt in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # im2 = cv2.putText(im2, '{} {} {}'.format(i, posX, posY), (x, y), fontFace=font, fontScale=fontScale, color=color, thickness=thickness)
        im2 = cv2.putText(im2, '{}'.format(i), (x, y), fontFace=font, fontScale=fontScale, color=color,
                          thickness=thickness)

        # Cropping the text block for giving input to OCR

        cropped = img[y:y + h, x:x + w]

        sharp_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

        cropped_resized = cv2.resize(cropped, (45, 45), interpolation=cv2.INTER_LINEAR)
        cropped_resized = cv2.filter2D(src=cropped_resized, ddepth=-1, kernel=sharp_kernel)
        cv2.imshow('crop', cropped_resized)
        cv2.waitKey(0)
        cv2.imwrite("cropped-img.png", cropped_resized)

        img = tf.keras.utils.load_img(
            "cropped-img.png", target_size=(45, 45)
        )
        class_names = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C',
                       'Delta',
                       'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'ascii_124', 'b', 'beta', 'cos', 'd',
                       'div',
                       'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int',
                       'j', 'k',
                       'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm',
                       'prime',
                       'q', 'rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times', 'u', 'v', 'w', 'y',
                       'z', '{',
                       '}']
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

    scale_percent = 150  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(im2, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('img', resized)
    cv2.waitKey(0)


segmentation("123432.png")
