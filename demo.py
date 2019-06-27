from siamese_model import SiameseModel
import os

siamese = SiameseModel("MobileNetV2", None)
siamese.load_model("checkpoints/0.12602-MobileNetV2.h5")
filename = r'D:\AAA\Data\myiqa\train\origin\1a1ae8.jpg'
print(siamese.predict(filename))
for i in range(1, 11):
    print(i)
    for j in range(4):
        dis_filename = filename.replace('origin', 'distortion/%d/%d'%(i, j))
        print('  ', siamese.predict(dis_filename))
# for img, score in siamese.predict(r'C:\Users\Darkn\Desktop\1').items():
#     print(img, score)