from siamese_model import SiameseModel

siamese = SiameseModel("MobileNetV2", 1, 2, 3, None)
siamese.load_model("")
filename = r'D:\AAA\Data\myiqa\train\origin\1a0ca6.jpg'
print(siamese.predict(filename))
for i in range(1, 11):
    print(siamese.predict(filename.replace('origin', str(i)+r'\3')))