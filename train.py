from siamese_model import SiameseModel
from data import DataGenerator, get_train_val, FTDataGenerator
from myargs import get_arg


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

    #--------| 临时设置 |--------#
    num_distort = 4
    train_generator.num_distort = num_distort
    val_generator.num_distort = num_distort
    #--------} 临时结束 |--------#
    model = SiameseModel(model_name)
    model.load_model(r'no_hid_checkpoints\2019-07-02\0.02568-Xception.h5')
    model.set_loss_param(batch_size, num_distort, num_level)
    # model.freeze_all_but_top()
    model.freeze_all_but_mid_and_top(1)
    model.compile()
    model.fit(epoch, train_info, val_info, checkpoints_dir)
    for layer in range(2, 20):
        print('start to train model with top %d' % (layer*20))
        model.freeze_all_but_mid_and_top(-layer*10)
        model.compile()
        model.fit(30, train_info, val_info, checkpoints_dir)

def finetune(model_name, model_weights, filename, dataset_dir, epoch, batch_size, img_shape=(256, 256), checkpoints_dir="checkpoints"):
    train_files, val_files = get_train_val(filename)
    train_generator = FTDataGenerator(train_files, dataset_dir, batch_size, img_shape)
    # for x, y in train_generator:
    #     print(y.tolist())
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
    for layer in range(3, 20):
        model.freeze_all_but_mid_and_top(-layer * 10)
        model.compile(loss_type="mae")
        model.fit(30, train_info, val_info, checkpoints_dir, mark="ft")


def main():
    args = get_arg()
    # train_type = 1
    # batch_size = 32
    # epoch = 100
    # model_name = "Xception"
    # chepoints_dir = 'no_hid_checkpoints'
    # img_shape = (256, 256)
    if args.train_type == 0:

        train_dir = (args.train_dir_ori, args.train_dir_dis)
        val_dir = (args.val_dir_ori, args.val_dir_dis)
        train(args.model_name, args.batch_size, args.epoch, train_dir, val_dir, args.img_shape, args.chepoints_dir)
    elif args.train_type == 1:
        # filename = r"E:\Data\IQA\tid2013\mos_with_names.txt"
        # dataset_dir = r"E:\Data\IQA\tid2013\distorted_images"
        # model_weights = "no_hid_checkpoints/2019-07-02/0.02568-Xception.h5"
        finetune(
            args.model_name,
            args.model_weights,
            args.filename,
            args.dataset_dir,
            args.epoch,
            args.batch_size,
            args.img_shape,
            args.chepoints_dir
        )
    else:
        raise ValueError("model_type cannot equal to %s" % args.train_type)


if __name__ == '__main__':
    main()