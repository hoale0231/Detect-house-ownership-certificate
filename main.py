# -*- coding: utf-8 -*-
from imgproc import loadImage
from ocr_pipeline import img_to_text_vietocr
import regex as re
from string import punctuation

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

def is_sodo(img):
  heso_form_same_sample = 0.6
  heso_sample_same_form = 0.35
  heso_min = 0.1
  # Đọc ảnh
  image = loadImage(img)
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
  
  print(sample_same_form/len(set(sample)))
  print(form_same_sample/len(form))
  print(len(sample))
  if len(set(sample)) == 0:
    return False
  return (sample_same_form/len(set(sample)) > heso_sample_same_form or form_same_sample/len(form) > heso_form_same_sample) and (
          sample_same_form/len(set(sample)) > heso_min and form_same_sample/len(form) > heso_min) and (
          len(sample) < 350), text

count = 0
for i in range(1, 51):
  if(is_sodo("image/image"+ str(i) +".jpg"))[0]:
    count += 1
    
print(count)
