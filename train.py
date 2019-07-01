from siamese_model import SiameseModel
from data import DataGenerator, get_train_val, FTDataGenerator
# from myargs import get_arg


def train(model_name, batch_size, epoch, train_dir, val_dir, img_shape=(256, 256), checkpoints_dir="checkpoints"):
    train_generator = DataGenerator(
        train_dir[0],
        train_dir[1],
        batch_size,
        img_shape
    )
    val_generator = DataGenerator(
        val_dir[0],
        val_dir[1],
        batch_size,
        img_shape
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
    model.compile()
    model.fit(epoch, train_info, val_info, checkpoints_dir)
    for layer in range(1, 20):
        model.freeze_all_but_mid_and_top(-layer*10)
        model.fit(30, train_info, val_info, checkpoints_dir)

def finetune(model_name, model_weights, filename, dataset_dir, epoch, batch_size, img_shape=(256, 256), checkpoints_dir="checkpoints"):
    train_files, val_files = get_train_val(filename)
    train_generator = FTDataGenerator(train_files, dataset_dir, batch_size, img_shape)
    for x, y in train_generator:
        print(y.tolist())
    steps_per_train = train_generator.length // batch_size
    train_info = (steps_per_train, train_generator)

    val_generator = FTDataGenerator(val_files, dataset_dir, batch_size, img_shape)
    steps_per_val = val_generator.length // batch_size
    val_info = (steps_per_val, val_generator)

    model = SiameseModel(model_name, None)
    model.load_model(model_weights)
    # model.compile(loss_type="mse")
    # print(model.model.evaluate_generator(val_generator))
    model.freeze_all_but_top()
    model.compile(loss_type="mae")
    model.fit(epoch, train_info, val_info, save_model_dir=checkpoints_dir, mark="ft")


def main():
    # args = get_arg()
    model_type = 0
    batch_size = 1
    epoch = 30
    model_name = "MobileNetV2"
    chepoints_dir = 'no_hid_checkpoints'
    img_shape = (224, 224)
    if model_type == 0:

        train_dir = (r"D:\temp_data\iqa\train\origin", r"D:\temp_data\iqa\train\distortion")
        val_dir = (r"D:\temp_data\iqa\val\origin", r"D:\temp_data\iqa\val\distortion")
        train(model_name, batch_size, epoch, train_dir, val_dir, img_shape, chepoints_dir)
    elif model_type == 1:
        filename = r"E:\Data\IQA\tid2013\mos_with_names.txt"
        model_weights = "checkpoints/0.02803-MobileNetV2.h5"
        dataset_dir = r"E:\Data\IQA\tid2013\distorted_images"
        finetune(model_name, model_weights, filename, dataset_dir, epoch, batch_size, img_shape, chepoints_dir)
    else:
        raise ValueError("model_type cannot equal to %s" % model_type)


if __name__ == '__main__':
    main()