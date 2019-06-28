from siamese_model import SiameseModel
from data import DataGenerator
# from myargs import get_arg


def train(model_name, batch_size, epoch, chepoints_dir, train_dir, val_dir):
    train_generator = DataGenerator(
        train_dir[0],
        train_dir[1],
        batch_size,
        (256, 256)
    )
    val_generator = DataGenerator(
        val_dir[0],
        val_dir[1],
        batch_size,
        (256, 256)
    )

    num_distort, num_level = train_generator.get_num_distort_level()
    num_data = train_generator.length
    steps_per_train = num_data // 1.5 // batch_size
    train_info = (steps_per_train, train_generator)

    num_data = val_generator.length
    steps_per_val = num_data // 1.5 // batch_size
    val_info = (steps_per_val, val_generator)

    model = SiameseModel(model_name)
    model.set_loss_param(batch_size, num_distort, num_level)
    # model.freeze_all_but_top()
    # model.freeze_all_but_mid_and_top(-2 * 10)
    model.fit(epoch, train_info, val_info, chepoints_dir)
    for layer in range(1, 20):
        model.freeze_all_but_mid_and_top(-layer*10)
        model.fit(30, train_info, val_info, chepoints_dir)

def finetune():
    pass

def main():
    # args = get_arg()
    model_type = 0
    batch_size = 4
    epoch = 10
    train_dir = (r"D:\temp_data\iqa\train\origin", r"D:\temp_data\iqa\train\distortion")
    val_dir = (r"D:\temp_data\iqa\val\origin", r"D:\temp_data\iqa\val\distortion")
    chepoints_dir = 'checkpoints'

    if model_type == 0:
        train("MobileNetV2", batch_size, epoch, chepoints_dir, train_dir, val_dir)
    elif model_type == 1:
        finetune()
    else:
        raise ValueError("model_type cannot equal to %s" % model_type)



if __name__ == '__main__':
    main()