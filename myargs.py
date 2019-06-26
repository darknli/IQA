def get_arg():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Tensorflow RankIQA Training")

    ## Path related arguments
    parser.add_argument('--train_type', type=int, default=0, help='1: train the ranking data; 2: finetune model with small dataset')
    parser.add_argument('--model_name', type=str, default="model", help='the root path of dataset')
    parser.add_argument('--train_list', type=str, default='live_train.txt', help='data list for read image.')
    parser.add_argument('--test_list', type=str, default='live_test.txt', help='data list for read image.')
    parser.add_argument('--ckpt_dir', type=str, default=os.path.abspath('..') + '/experiments',
                        help='the path of ckpt file')
    parser.add_argument('--logs_dir', type=str, default=os.path.abspath('..') + '/experiments',
                        help='the path of tensorboard logs')
    parser.add_argument('--vgg_models_path', type=str,
                        default=os.path.abspath('..') + "/experiments/vgg_models/" + 'vgg16_weights.npz')

    ## models retated argumentss
    parser.add_argument('--save_ckpt_file', type=str2bool, default=True,
                        help="whether to save trained checkpoint file ")

    ## dataset related arguments
    parser.add_argument('--dataset', default='tid2013', type=str, choices=["LIVE", "CSIQ", "tid2013"],
                        help='datset choice')
    parser.add_argument('--crop_width', type=int, default=224, help='train patch width')
    parser.add_argument('--crop_height', type=int, default=224, help='train patch height')

    ## train related arguments
    parser.add_argument('--is_training', type=str2bool, default=True, help='whether to train or test.')
    parser.add_argument('--is_eval', type=str2bool, default=True, help='whether to test.')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--test_step', type=int, default=500)
    parser.add_argument('--summary_step', type=int, default=10)

    ## optimization related arguments
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='init learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7, help='keep neural node')
    parser.add_argument('--iter_max', type=int, default=90000, help='the maxinum of iteration')
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    args = parser.parse_args()
    return args