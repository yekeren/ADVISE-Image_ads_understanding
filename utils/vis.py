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

def image_draw_text(image, pt, text, 
    color=(255, 255, 255), bkg_color=(0, 255, 0), norm=True):

  height, width, _ = image.shape
  x, y = pt
  if norm:
    x = int(x * width)
    y = int(y * height)

  font_face = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  thickness = 1

  (sx, sy), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

  cv2.rectangle(image, pt1=(x, y), pt2=(x + sx, y + sy + baseline), 
      color=bkg_color, thickness=-1)
  cv2.putText(image, text, org=(x, y + sy), 
      fontFace=font_face, fontScale=font_scale, 
      color=color, thickness=thickness)


def image_draw_bounding_box(image, bndbox, color=(0, 255, 0), norm=True):
  height, width, _ = image.shape
  x1, y1, x2, y2 = bndbox
  if norm:
    x1 = int(x1 * width)
    x2 = int(x2 * width)
    y1 = int(y1 * height)
    y2 = int(y2 * height)
  cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)

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

def image_uint8_to_base64(image, ext='.jpg', disp_size=None, convert_to_bgr=False):
  if convert_to_bgr:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if disp_size is not None:
    image = cv2.resize(image, disp_size)
  _, encoded = cv2.imencode(ext, image)
  return encoded.tostring().encode('base64').replace('\n', '')
