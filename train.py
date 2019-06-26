from siamese_model import SiameseModel
from data import DataGenerator
# from myargs import get_arg


def train(model_name, batch_size, epoch):
    gendata = DataGenerator(batch_size=batch_size, img_shape=(256, 256))
    num_distort, num_level = gendata.get_num_distort_level()
    num_data = gendata.length
    model = SiameseModel(model_name, batch_size, num_distort, num_level)
    model.freeze_all_but_top()
    steps_per_epoch = num_data //1.1
    model.fit(epoch, steps_per_epoch, gendata)

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