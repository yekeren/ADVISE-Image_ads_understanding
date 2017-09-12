import numpy as np
import cv2

def image_load(image_path, convert_to_rgb=False):
  image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  if convert_to_rgb:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def image_save(image_path, image, convert_to_bgr=False):
  if convert_to_bgr:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  cv2.imwrite(image_path, image)

def image_uint8_to_float32(image, scale=2.0 / 255, offset=-1.0):
  return image.astype(np.float32) * scale + offset

def image_float32_to_uint8(image, scale=255.0 / 2.0, offset=1.0):
  image = (image + offset) * scale
  image = np.maximum(0.0, np.minimum(255.0, image))
  return image.astype(np.uint8)

def image_crop_and_resize(image, box, crop_size):
  height, width, _ = image.shape

  x1, y1, x2, y2 = box
  x1 = int(x1 * width)
  y1 = int(y1 * height)
  x2 = int(x2 * width)
  y2 = int(y2 * height)
  return cv2.resize(image[y1: y2, x1: x2, :], crop_size)

def image_uint8_to_base64(image, ext='.jpg', disp_size=None):
  if disp_size is not None:
    image = cv2.resize(image, disp_size)
  _, encoded = cv2.imencode(ext, image)
  return encoded.tostring().encode('base64').replace('\n', '')
