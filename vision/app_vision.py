"""
Toma una imagen no orientada apropiadamente y la coloca derecha

https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

Si no funciona probar
https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/

pip install imutils
porque hace un resize del ancho manteniendo el ratio de la altura
"""


#%%
import cv2
import imutils
import numpy as np

#%%

def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    """Alinea las imagenes teniendi en cuenta un template

    img = align_images(image=os.path.join('../', "data", r'8_rotated1.jpg'), 
             template=os.path.join('../', "data", r'8.jpg'),
             debug=False)

    Args:
        image ([type]): [description]
        template ([type]): [description]
        maxFeatures (int, optional): [description]. Defaults to 500.
        keepPercent (float, optional): [description]. Defaults to 0.2.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    image = cv2.imread(image)
    template = cv2.imread(template)

    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # return the aligned image
    return aligned

def prepocesar_imagen(imagen, erode, dilate):

    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = cv2.bitwise_not(img)
    img = cv2.erode(img, struct, iterations=erode)
    img = cv2.dilate(img, struct, anchor=(-1, -1), iterations=dilate)

    return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)