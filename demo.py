from siamese_model import SiameseModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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


siamese = SiameseModel("Xception", None)
siamese.load_model("no_hid_checkpoints/2019-07-03/ft0.03794-Xception.h5")
# contrast(r'D:\temp_data\iqa\train\origin\0.0024_1805.jpg', r'D:\temp_data\iqa\train\distortion')
# filename = r'D:\AAA\Data\myiqa\train\origin\1a1ae8.jpg'
# # print(siamese.predict(filename))
# # for i in range(1, 11):
# #     print(i)
# #     for j in range(4):
# #         dis_filename = filename.replace('origin', 'distortion/%d/%d'%(i, j))
# #         print('  ', siamese.predict(dis_filename))
for img, score in siamese.predict(r'E:\Data\IQA\tid2013\distorted_images').items():
    print(os.path.basename(img), score)