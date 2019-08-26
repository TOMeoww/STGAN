import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr_dis', type=int, default=2e-4)
parser.add_argument('--lr_gen', type=int, default=2e-4)
parser.add_argument("--lr_beta", type = tuple, default=(0.5,0.999))
parser.add_argument('--lambda1', type=int, default=1)
parser.add_argument('--lambda2', type=int, default=10)
parser.add_argument('--lambda3', type=int, default=100)
parser.add_argument('--updata_epoch_gen', type=int, default=100)
parser.add_argument('--updata_epoch_dis', type=int, default=100)
parser.add_argument('--batch_size', type = int , default = 512, help="batch_size of training")

opts = parser.parse_args()

opts_name = ['batch_size', 'lambda1', 'lambda2', 'lambda3', 'lr_beta',\
     'lr_dis', 'lr_gen=0.0002', 'updata_epoch_dis', 'updata_epoch_gen']
length_opts = len(opts_name)

hyperparameters = {opts_name[x] : getattr(opts, opts_name[x]) for x in range(length_opts)}

