# -*- coding: utf-8 -*-
from imgproc import loadImage
from ocr_pipeline import img_to_text_vietocr
import regex as re
from string import punctuation
import numpy as np
from skimage.transform import rotate
from deskew import determine_skew

def angle_deskew(_img):
  return determine_skew(_img)
    
  
patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}

def convert_to_unsign(text):
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
    return output

sodo = 'thửa đất nhà ở và tài sản khác gắn liền với số tờ bản đồ địa chỉ diện tích hình thức sử dụng mục đích thời hạn nguồn gốc công trình xây dựng rừng xuất là trồng cây lâu năm ghi chú sơ hiệu đỉnh chiều dài những thay đổi sau khi cấp giấy chứng nhận nội dung cơ sở pháp lý xác của quan có thẩm quyền bảng liệt kê tọa độ góc ranh cạnh'
form = convert_to_unsign(sodo).split()
# const
heso_sample_same_form = 0.4
heso_form_same_sample = 0.68
heso_min = 0.1

def is_sodo_straight(image):
  # Đọc ảnh
  sample = img_to_text_vietocr(image,'cuda:0')
  # Chuyển về ký tự in thường
  sample = ' '.join(sample).lower()
  # Xóa các dấu câu
  for c in punctuation:
    sample = sample.replace(c," ")
  # Xử lý các từ thường xuyên đọc sai
  sample = sample.split()
  for i in range(len(sample)):
    if re.search("th.a", sample[i]):
      sample[i] = "thửa" 
    if  (sample[i].isdigit() or
        (sample.count(sample[i]) > 4 and sample[i] != "đất") or
        (len(sample[i]) == 1 or len(sample[i]) > 7)):
      sample[i] = ""

  sample = ' '.join(sample)
  sample = re.sub(' +', ' ', sample)
  # So sánh với sổ đỏ mẫu
  text = sample
  sample = convert_to_unsign(sample).split()
  sample_same_form = sum([word in form for word in set(sample)])
  form_same_sample = sum([word in sample for word in form])
  
  if len(set(sample)) == 0:
    return False, ""
  # print(sample_same_form/len(set(sample)))
  # print(form_same_sample/len(form))
  # print(len(sample))
  return  len(sample) < 350 and len(sample) > 5 and (
          sample_same_form/len(set(sample)) > heso_sample_same_form or form_same_sample/len(form) > heso_form_same_sample) and (
          sample_same_form/len(set(sample)) > heso_min and form_same_sample/len(form) > heso_min), text

def is_sodo(image):
  image = loadImage(image)
  angle = angle_deskew(image)
  for i in range(4):
    rotated = rotate(image, angle + 90 * i, resize=True) * 255
    result, text = is_sodo_straight(rotated.astype(np.uint8))
    if(result):
      return True, text 
  return False, ""
    
  

