from networks.vnet import VNet
from networks.vnet_contrast import VNet_Contrast

def net_factory_3d(net_type="vnet", in_chns=1, class_num=2, patch_num=2):
    if net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_contrast":
        net = VNet_Contrast(n_channels=in_chns, n_classes=class_num, normalization='batchnorm',
                   has_dropout=True, patch_num=patch_num).cuda()
    else:
        net = None
    return net
