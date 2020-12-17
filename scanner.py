import numpy as np
import cv2


# THIS PART OF CODE IS USED FOR TESTING PURPOSES AND HAS NO EFFECT IN THE MODULE USED IN THE APP

# shows all required images by accepting an image array
def show_images(images, names):
    for i in range(len(images)):
        cv2.imshow(names[i], images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# to save the images
def save_images(images, names):
    for i in range(len(images)):
        cv2.imwrite(names[i] + '.jpg', images[i])


# THIS PART OF THE CODE CONTAINS ALL OF THE FUNCTIONS THAT CAN BE USED TO ENHANCE THE APP

# changes the hsv values of an image, with the help of offset values provided
def change_HSV(image, h_offset, s_offset, v_offset):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image[:, :, 0] = np.clip(1.00 * image[:, :, 0] + h_offset, 0, 179)
    image[:, :, 1] = np.clip(1.00 * image[:, :, 1] + s_offset, 0, 255)
    image[:, :, 2] = np.clip(1.00 * image[:, :, 2] + v_offset, 0, 255)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


# changes the hsv values of an image, with the help of offset values provided
def change_RGB(image, r_offset, g_offset, b_offset):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image[:, :, 0] = np.clip(1.00 * image[:, :, 0] + r_offset, 0, 255)
    image[:, :, 1] = np.clip(1.00 * image[:, :, 1] + g_offset, 0, 255)
    image[:, :, 2] = np.clip(1.00 * image[:, :, 2] + b_offset, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


# changes brightness and contrast of current positioned values of slider, relative to the original position
def change_brightness_contrast(original_image, brightness, contrast):
    contrast = float(contrast)
    return np.clip(contrast*original_image + brightness, 0, 255).astype('int16')


# function to change the size of the pictures
def change_size(original_image, size_index):
    # size_index is an int which specifies which size we want to resize it to
    # higher
    init_height, init_width, _ = original_image.shape
    scale = 1

    if size_index == 1:  # this is for very low quality
        scale = 6
    elif size_index == 2:  # this is for medium quality
        scale = 4
    elif size_index == 3:  # this is for decent quality
        scale = 3

    temp = cv2.resize(original_image, (int(init_width//scale), int(init_height//scale)))
    return cv2.resize(temp, (init_width, init_height))


# THIS PART OF CODE IS USED FOR GETTING THE SCANNED IMAGE

# reorders all the points accordingly
def rectify(corner_points):
    corner_points = corner_points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)
    temp1 = corner_points.sum(1)
    new_points[0] = corner_points[np.argmin(temp1)]
    new_points[2] = corner_points[np.argmax(temp1)]
    temp2 = np.diff(corner_points, axis=1)
    new_points[1] = corner_points[np.argmin(temp2)]
    new_points[3] = corner_points[np.argmax(temp2)]
    return new_points


# returns the corner points of the document, from all the possible contours
def get_rectangle(contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        p = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * p, True)

        if len(approx) == 4:
            return approx


# finds the binary image of the document through adaptive threshold
def get_binary(scanned_image, is_coloured):
    if not is_coloured:
        scanned_image_grey = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)
        scanned_image_thresh_binary = cv2.adaptiveThreshold(scanned_image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, 25, 6)
    else:
        scanned_image_thresh_binary = cv2.adaptiveThreshold(scanned_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, 25, 6)
    return scanned_image_thresh_binary


# get the enhanced document image in RGB
def get_binary_rgb(scanned_image):
    (r, g, b) = cv2.split(scanned_image)
    rn = get_binary(r, True)
    gn = get_binary(g, True)
    bn = get_binary(b, True)
    return cv2.merge((rn, gn, bn))


# get the four corner points
def get_corner_points(image):
    # part 1: setting height and width values
    height, width, _ = image.shape

    # part 2: converting our image to get edges
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    edged_image = cv2.Canny(blurred_image, 50, 100)
    edged_image = cv2.dilate(edged_image, kernel, iterations=1)

    # part 3: finding contours in our edged image
    contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # finding the corner points and reordering them
    rectangle = get_rectangle(contours)
    approx_rectangle = np.float32(rectify(rectangle))

    return approx_rectangle, rectangle, edged_image


def get_scanned_image_manual(image, topLeftX, topLeftY, topRightX, topRightY, bottomLeftX, bottomLeftY, bottomRightX,
                             bottomRightY):
    # part1: getting height and width of image
    height, width, _ = image.shape

    dimensions = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # part2: setting the rectangle with the help of points passed by user
    rectangle = np.array([
        [topLeftX, topLeftY],
        [topRightX, topRightY],
        [bottomLeftX, bottomLeftY],
        [bottomRightX, bottomRightY]
    ])

    # part 3: rectifying the corner points and getting the birds eye view
    approx_rectangle = np.float32(rectify(rectangle))
    matrix = cv2.getPerspectiveTransform(approx_rectangle, dimensions)
    scanned_image = cv2.warpPerspective(image, matrix, (width, height))

    # part 4: obtaining the edged image
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    edged_image = cv2.Canny(blurred_image, 50, 100)
    edged_image = cv2.dilate(edged_image, kernel, iterations=1)

    # part 5: drawing the rectangle detected in our image
    cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 3)

    # part 6: getting the binary image (normal and coloured) of the final scan
    scanned_image_thresh_binary = get_binary(scanned_image, False)
    scanned_image_thresh_coloured = get_binary_rgb(scanned_image)
    images = ([image, scanned_image, edged_image, scanned_image_thresh_binary, scanned_image_thresh_coloured])
    names = (["Initial image", "Scanned image", "Edged image",
              "Scanned threshold, binary", "Scanned threshold, coloured"])

    return scanned_image, scanned_image_thresh_binary, images, names


# the main function, that calls all other functions
def get_scanned_image_auto(image):
    # part1: obtaining height and width
    height, width, _ = image.shape

    # part 2: getting the four corner points as detected through opencv
    approx_rectangle, rectangle, edged_image = get_corner_points(image)

    # part 3: finding the bird's eye view of the document
    dimensions = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(approx_rectangle, dimensions)
    scanned_image = cv2.warpPerspective(image, matrix, (width, height))

    # part 4: drawing the rectangle detected in our image
    cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 3)

    # getting the binary image (normal and coloured) of the final scan
    scanned_image_thresh_binary = get_binary(scanned_image, False)
    scanned_image_thresh_coloured = get_binary_rgb(scanned_image)
    images = ([image, scanned_image, edged_image, scanned_image_thresh_binary, scanned_image_thresh_coloured])
    names = (["Initial image", "Scanned image", "Edged image",
              "Scanned threshold, binary", "Scanned threshold, coloured"])
    return scanned_image, scanned_image_thresh_binary, images, names
