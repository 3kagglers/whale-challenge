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
    new_x =img.shape[1] * scale
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


if __name__ == '__main__':

    IMAGE = './boop.png' # change here
    TMP_IMG = 'resized_image.png'

    resize_image_scale(IMAGE, TMP_IMG, 10)
    cv2.imshow('resized_image.png')
    cv2.waitKey(0)

    resize_image_scale(IMAGE, TMP_IMG, (100, 100))
    cv2.imshow('resized_image.png')
    cv2.waitKey(0)
