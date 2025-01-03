from networks.unet import UNet
from networks.unet_contrast import UNet_Contrast

def net_factory(net_type="unet", in_chns=1, class_num=4, patch_num=4):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_contrast":
        net = UNet_Contrast(in_chns=in_chns, class_num=class_num, patch_num=patch_num).cuda()
    else:
        net = None
    return net
