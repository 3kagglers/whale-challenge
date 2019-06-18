"""Image util functions to operate."""

import cv2


def resize_image_scale(path_to_image, new_path_to_image, scale):
    """
    Resizes image to new scale in both x and y.
    Keeps ratio.
    Saves image to new_path_to_image
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    :scale: float : Scale to resize image to.
    """
    img = cv2.imread(path_to_image)
    new_x = img.shape[1] * scale
    new_y = img.shape[0] * scale
    new_img = cv2.resize(img, (int(new_x), int(new_y)))
    cv2.imwrite(new_path_to_image, new_img)


def resize_image_fixed(path_to_image, new_path_to_image, size):
    """
    Resizes image to fixed size.
    Does not keep ratio.
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    :size: Tuple[int, int] : New image size in pixels, (x, y).
    """
    img = cv2.imread(path_to_image)
    new_img = cv2.resize(img, size)
    cv2.imwrite(new_path_to_image, new_img)

def change_to_grayscale(path_to_image, new_path_to_image):
    """
    Change image representation from BGR to grayscale.
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    """
    img = cv2.imread(path_to_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(new_path_to_image, gray)

def clean_image(path_to_image, new_path_to_image):
    """
    Remove the noises. Image blurring. 
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    """
    img = cv2.imread(path_to_image)
    blur = cv2.GaussianBlur(img,(5,5))
    cv2.imwrite(new_path_to_image, blur)

def get_contour(path_to_image, new_path_to_image):
    """
    Get the contours of the images. 
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    """

    # Read 
    img = cv2.imread(path_to_image)
    # Clean 
    blur = cv2.GaussianBlur(img,(5,5),0)
    # Convert to gray
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Get contours
    _, threshold = cv2.threshold(gray, 127, 255, 0)    
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bigger contour
    contour_sizes = [(cv2.contourArea(contour),contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    cv2.drawContours(gray, biggest_contour, -1, (0, 255, 0), 3)

    cv2.imwrite(new_path_to_image, gray)

if __name__ == '__main__':

    #IMAGE = './boop.png' # change here
    IMAGE = 'test1.jpg'
    #TMP_IMG = 'resized_image.png'

    #resize_image_scale(IMAGE, TMP_IMG, 10)
    #cv2.imshow('resized_image.png')
    #cv2.waitKey(0)

    #resize_image_scale(IMAGE, TMP_IMG, (100, 100))
    #cv2.imshow('resized_image.png')
    #cv2.waitKey(0)

    get_contour(IMAGE, 'contour_image.jpg')
    #cv2.imshow('gray.jpg')
    #cv2.waitKey(0)


