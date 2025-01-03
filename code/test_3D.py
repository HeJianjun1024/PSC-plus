import argparse
import os
import shutil
from glob import glob
import time
import torch

from networks.vnet import VNet
from networks.vnet_contrast import VNet_Contrast
from utils.val_3D import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTS2019/PSC+', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet_contrast', help='model_name')
parser.add_argument('--patch_n', type=int, default=2, help='patch n in pair shuffle operation')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
parser.add_argument('--labeled_num', type=int, default=25, help='number of labeled data')


def Inference(args):
    snapshot_path = "../model/{}_{}_n{}/{}".format(
        args.exp, args.labeled_num, args.patch_n, args.model)
    num_classes = args.num_classes
    patch_num = args.patch_n
    net = VNet_Contrast(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True, patch_num=patch_num).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(args.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=args.root_path, test_list="test.txt", num_classes=num_classes,
        patch_size=(96, 96, 96), stride_xy=16, stride_z=16)
    print(avg_metric)


if __name__ == '__main__':
    args = parser.parse_args()
    Inference(args)
    