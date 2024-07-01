from PIL import Image


filepath='/home/arslan/DIP Projects/captcha-recognition-pytorch/input/captcha_images_v2'

img=Image.open(filepath+'/2bg48.png')

print(img.width,img.height)