import os,cv2
import numpy as np
from PIL import Image

path = "./resources/image/input"
x = 640
y = 640

def fileInputList():
   fileList = []
   resize_img = []
   img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png'] 
   for(root,dirs,files) in os.walk(path):
    for file_name in files:
        if os.path.splitext(file_name)[1] in img_extension:
            img_path = root + '/' + file_name
            img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
            fileList.append(img_path)
            
    for file in fileList:
       image = Image.open(file)
       resized_image = image.resize((x, y))
       resize_img.append(resized_image)

    return resize_img
