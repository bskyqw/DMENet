import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from DMENet import DMENet
from dataloader import test_dataset

dataset_path = './datasets'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model = DMENet(32).cuda()
#model.load_state_dict(torch.load('./Checkpoint/DMENet/DMENet_epoch_best.pth'))
model.load_state_dict(torch.load('DMENet_epoch_best.pth'))
model.eval()
#test
test_datasets = ['NEURSDD-AUG']
for dataset in test_datasets:
    save_path = './test_maps/'+dataset+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root  = dataset_path  +'/'+dataset+ '/test/'+'/img/'
    gt_root     = dataset_path +'/'+dataset+ '/test/'+'/GT/'
    depth_root  = dataset_path  +'/'+dataset+'/test/' +'/depth/'

    test_loader = test_dataset(image_root, gt_root,depth_root,224)

    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt      = np.asarray(gt, np.float32)
        gt     /= (gt.max() + 1e-8)
        image   = image.cuda()
        depth   = depth.cuda()
        pre_res = model(image, depth)[0]
        res     = pre_res     
        res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res     = res.data.cpu().numpy().squeeze()
        res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
