import os

path = 'data/obj/'


imgList = os.listdir(r'C:\Users\Fizics\Desktop\YOLOv3-Series-master\[part 4]OpenLabelling\images')

print(imgList)

textFile = open('train.txt','w')

for img in imgList:
    imgPath = path+img+'\n'
    textFile.write(imgPath)
