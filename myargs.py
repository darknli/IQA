import argparse
import os


def get_arg():
    parser = argparse.ArgumentParser(description="keras RankIQA Training")

    parser.add_argument('--train_type', type=int, default=0, help='0: train the ranking data; 1: finetune model with small dataset')
    parser.add_argument('--model_name', type=str, default="MobileNetV2", help='the root path of dataset')
    parser.add_argument('--chepoints_dir', type=str, default=r"no_hid_checkpoints", help='the root direcory saved model')
    parser.add_argument('--epoch', type=int, default=100, help='the number of epoch')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='''batch size is recommended to be set to 2 but 32
                        If the selection 0 of train_type ''')
    parser.add_argument('--img_shape', type=tuple, default=(256, 256), help='inputs shape of images')

    # train_type = 0
    parser.add_argument('--train_dir_ori', type=str, default=r"D:\temp_data\iqa\train\origin",
                        help='the directory of train of the original images.')
    parser.add_argument('--train_dir_dis', type=str, default=r"D:\temp_data\iqa\train\ordistortion",
                        help='the directory of train of the distortion images.')
    parser.add_argument('--val_dir_ori', type=str, default=r"D:\temp_data\iqa\val\origin",
                        help='the directory of validation of the original images.')
    parser.add_argument('--val_dir_dis', type=str, default=r"D:\temp_data\iqa\val\ordistortion",
                        help='the directory of validation of the distortion images.')

    # train_type = 1
    parser.add_argument('--filename', type=str, default=r"E:\Data\IQA\tid2013\mos_with_names.txt",
                        help='name of data list for read images and its mean opinion score.')
    parser.add_argument('--dataset_dir', type=str, default=r"E:\Data\IQA\tid2013\distorted_images",
                        help='the path of dataset')
    parser.add_argument('--model_weights', type=str, default="no_hid_checkpoints/2019-07-02/0.02568-Xception.h5",
                        help='the path of model weights')


    args = parser.parse_args()
    return args