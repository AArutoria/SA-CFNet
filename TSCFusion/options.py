import argparse
import pathlib


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    # data related
        self.parser.add_argument('--train_data_path', type=str, default='H:\\TSCFusion\\ADRNet-main\\dataa\\rgb2ir\\RoadScene\\fusion\\VIS\\', help='path to train data')
        self.parser.add_argument('--test_data_path', type=str, default='H:\\TSCFusion\\ADRNet-main\\dataa\\rgb2ir\\RoadScene\\fusion\\VIS\\', help='path to test data')
        self.parser.add_argument('--train_ir', default=r'H:\data\RoadScene\Test\vi', type=pathlib.Path)
        self.parser.add_argument('--train_it', default=r'H:\data\RoadScene\Test\ir', type=pathlib.Path)
        self.parser.add_argument('--test_ir', default=r'F:\Code\TSCFusion\ADRNet-main\result\First_experiments\RoadScene\ir_reg', type=pathlib.Path)
        self.parser.add_argument('--test_it', default=r'F:\Code\TSCFusion\ADRNet-main\result\First_experiments\RoadScene\reference_ir', type=pathlib.Path)
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
        self.parser.add_argument('--nThreads', type=int, default=8, help='cpu threads')
        self.parser.add_argument('--train', action='store_true',default=True, help='Train mode or test mode')
        self.parser.add_argument('--dst', default=r'F:\one_layers', help='fuse image save folder',type=pathlib.Path)
    
    # generate data
        self.parser.add_argument('--rotation', type=int, default=30)
        self.parser.add_argument('--translation', type=float, default=0.05)
        self.parser.add_argument('--scaling', type=float, default=0.12)
        self.parser.add_argument('--dim', type=int, default=2)

    # output related
        self.parser.add_argument('--train_img_dir', type=str, default=r'E:\TSCFusion\ADRNet\result\five_layer', help='save path for training process visualization')
        self.parser.add_argument('--model_dir', type=str, default=r'E:\TSCFusion\ADRNet\result\five_layer', help='save path for model weights')
        self.parser.add_argument('--data_name', type=str, default='RoadScene', help='save path for model weights')
    # training related
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='')
        self.parser.add_argument('--n_ep', type=int, default=300, help='epoch')
        self.parser.add_argument('--n_ep_decay', type=int, default=150, help='which epoch starts to reduce the learning rate，-1：不变')

    # test related
        self.parser.add_argument('--train_logs_dir', type=str, default=r'E:\TSCFusion\ADRNet\result\five_layer', help='Save path for training information')
        self.parser.add_argument('--test_logs_dir', type=str, default=r'E:\TSCFusion\ADRNet\result\five_layer', help='Save path for testing information')



    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- train and test settings ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
