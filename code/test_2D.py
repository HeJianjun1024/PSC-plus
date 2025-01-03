import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC/data/slices/', help='Name of Experiment')
parser.add_argument('--list_dir', type=str,
                    default='../data/ACDC/lists/10%', help='list dir')
parser.add_argument('--exp', type=str,
                    default='ACDC/PSC+', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_contrast', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--patch_n', type=int, default=4,
                    help='patch n in pair shuffle operation')      
parser.add_argument('--setting', type=str, default='10%', help='labeled data ratio')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        spacing = np.ones(pred.ndim)
        sf = compute_surface_distances(gt, pred, spacing_mm=spacing)
        nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
        return dice, hd95, asd, nsd
    else:
        return 0, 0, 0, 0


def test_single_volume(case, net, args):
    h5f = h5py.File(args.root_path + "{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            p, _ = net(input)
            out = torch.argmax(torch.softmax(p, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, args.num_classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def Inference(args):
    with open(args.list_dir + '/test.txt', 'r') as f:
        image_list = f.readlines()
    
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/{}".format(args.exp, args.setting)
    
    net = net_factory(net_type=args.model, in_chns=1,
                      class_num=args.num_classes, patch_num=args.patch_n)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(args.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    metric_list = 0.0
    for case in tqdm(image_list):
        metric_i = test_single_volume(case, net, args)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(image_list)
    for class_i in range(args.num_classes-1):
        print('class_{}_dice'.format(class_i+1), metric_list[class_i, 0])
        print('class_{}_hd95'.format(class_i+1), metric_list[class_i, 1])
        print('class_{}_asd'.format(class_i+1), metric_list[class_i, 2])
        print('class_{}_nsd'.format(class_i+1), metric_list[class_i, 3])
    performance = np.mean(metric_list, axis=0)
    print('test_mean_dice', performance[0])
    print('test_mean_hd95', performance[1])
    print('test_mean_asd', performance[2])
    print('test_mean_nsd', performance[3])


if __name__ == '__main__':
    args = parser.parse_args()
    list_dir = args.list_dir.split('/')
    list_dir[-1] = args.setting
    args.list_dir = '/'.join(list_dir)
    Inference(args)
