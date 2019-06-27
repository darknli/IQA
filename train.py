from siamese_model import SiameseModel
from data import DataGenerator
# from myargs import get_arg


def train(model_name, batch_size, epoch):
    train_generator = DataGenerator(
        r"D:\AAA\Data\myiqa\train\origin",
        r"D:\AAA\Data\myiqa\train\distortion",
        batch_size,
        (256, 256)
    )
    val_generator = DataGenerator(
        r"D:\AAA\Data\myiqa\val\origin",
        r"D:\AAA\Data\myiqa\val\distortion",
        batch_size,
        (256, 256)
    )

    num_distort, num_level = train_generator.get_num_distort_level()
    # print(num_level, num_distort)
    # for x,y in gendata:
    #     for batch in range(batch_size):
    #         batch_s = batch*num_distort*num_level
    #         for dis in range(num_distort):
    #             step = batch_s+dis*num_level
    #             for i in range(num_level-1):
    #                 for j in range(i+1, num_level):
    #                     a = x[step+i]
    #                     b = x[step+j]
    #                     print(step+i, step+j)
    #             print(' ')
    #     print(x.shape, y.shape)
    #     exit()
    num_data = train_generator.length
    steps_per_train = num_data //1.1
    train_info = (steps_per_train, train_generator)

    num_data = val_generator.length
    steps_per_val = num_data // 1.1
    val_info = (steps_per_val, val_generator)

    model = SiameseModel(model_name, batch_size, num_distort, num_level)
    model.freeze_all_but_top()
    model.fit(epoch, train_info, val_info, 'checkpoints')
    for layer in range(20):
        model.freeze_all_but_mid_and_top(-layer*10)
        model.fit(30, train_info, val_info, 'checkpoints')

def finetune():
    pass

def main():
    # args = get_arg()
    model_type = 0
    batch_size = 1
    epoch = 50
    if model_type == 0:
        train("MobileNetV2", batch_size, epoch)
    elif model_type == 1:
        finetune()
    else:
        raise ValueError("model_type cannot equal to %s" % model_type)



if __name__ == '__main__':
    main()