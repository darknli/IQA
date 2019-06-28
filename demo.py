from siamese_model import SiameseModel
import os

def contrast(img_path, distort_path):
    result = []
    img_name = os.path.basename(img_path)
    print('origin')
    print(siamese.predict(img_path))
    for i in range(1, 11):
        print(i)
        for j in range(4):
            path = os.path.join(distort_path, '%d/%d/%s'%(i, j, img_name))
            print(siamese.predict(path))


siamese = SiameseModel("MobileNetV2", None)
siamese.load_model("checkpoints/0.20000-MobileNetV2.h5")
contrast(r'D:\temp_data\iqa\train\origin\0.0024_1805.jpg', r'D:\temp_data\iqa\train\distortion')
# filename = r'D:\AAA\Data\myiqa\train\origin\1a1ae8.jpg'
# # print(siamese.predict(filename))
# # for i in range(1, 11):
# #     print(i)
# #     for j in range(4):
# #         dis_filename = filename.replace('origin', 'distortion/%d/%d'%(i, j))
# #         print('  ', siamese.predict(dis_filename))
# for img, score in siamese.predict(r'D:\temp_data\iqa\train\distortion\1\3').items():
#     print(img, score)