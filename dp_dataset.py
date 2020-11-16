# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
#import matplotlib.pyplot as plt
from torch.autograd import Variable
np.set_printoptions(threshold=np.inf)


class DPDataset(data.Dataset):

    def __init__(self, opt):
        super(DPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode
        self.stage = opt.stage
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_1d = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5,), (0.5,))])
        self.transform_1d_x = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5), (0.5))])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "DPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        c_gmm = Image.open(osp.join(self.data_path, 'warp-cloth-gmm', c_name))
        cm_gmm = Image.open(osp.join(self.data_path, 'warp-mask-gmm', c_name))
        c_fw = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
        cm_fw = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
        
        c = self.transform(c)
        c_gmm = self.transform(c_gmm)
        c_fw = self.transform(c_fw)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)
        cm.unsqueeze_(0)
        cm_gmm_array = np.array(cm_gmm)
        cm_gmm_array = (cm_gmm_array >= 128).astype(np.float32)
        cm_gmm = torch.from_numpy(cm_gmm_array)
        cm_gmm.unsqueeze_(0)
        cm_fw_array = np.array(cm_fw)
        cm_fw_array = (cm_fw_array >= 128).astype(np.float32)
        cm_fw = torch.from_numpy(cm_fw_array)
        cm_fw.unsqueeze_(0)

        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im)

        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))

        parse_array = np.array(im_parse)
        im_parse_1d = self.transform_1d(im_parse)
        im_parse_torch = torch.from_numpy(parse_array.astype(np.float32))
        im_parse_torch = im_parse_torch.unsqueeze_(0)

        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)

        parse_dress6 = torch.FloatTensor((parse_array == 6).astype(np.int))
        parse_coat7 = torch.FloatTensor((parse_array == 7).astype(np.int))
        parse_arm1 = torch.FloatTensor((parse_array == 14).astype(np.int))
        parse_arm2 = torch.FloatTensor((parse_array == 15).astype(np.int))
        parse_arm = parse_arm1 + parse_arm2

        arm1_3d = parse_arm1.unsqueeze_(0)
        arm2_3d = parse_arm2.unsqueeze_(0)

        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        shape = self.transform_1d(parse_shape)
        phead = torch.from_numpy(parse_head)
        pcm = torch.from_numpy(parse_cloth)
        im_cm = pcm.unsqueeze_(0)

        parse_cross_entropy2d = im_parse_torch * (1 - im_cm)

        old_label = im_parse_torch
        old_label = old_label * (1 - parse_dress6) + parse_dress6 * 5
        old_label = old_label * (1 - parse_coat7) + parse_coat7 * 5
        im_parse_torch = old_label

        old_label = im_parse_torch
        old_label = old_label * (1 - parse_arm1) + parse_arm1 * 5
        old_label = old_label * (1 - parse_arm2) + parse_arm2 * 5
        label_wo_arm = old_label

        # upper cloth
        im_c = im * pcm + (1 - pcm)
        im_h = im * phead - (1 - phead)

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform_1d(one_map)
            pose_map[i] = one_map[0]

        r_hands = 41
        one_map4 = Image.new('L', (self.fine_width, self.fine_height))
        draw4 = ImageDraw.Draw(one_map4)
        point4x = pose_data[4, 0]
        point4y = pose_data[4, 1]
        if point4x > 1 and point4y > 1:
            draw4.rectangle((point4x - r_hands, point4y - r_hands, point4x + r_hands, point4y + r_hands), 'white', 'white')
        one_map4 = self.transform_1d(one_map4)
        pose_map_4 = one_map4
        pose_map_4 = (pose_map_4 + 1) * 0.5

        one_map7 = Image.new('L', (self.fine_width, self.fine_height))
        draw7 = ImageDraw.Draw(one_map7)
        point7x = pose_data[7, 0]
        point7y = pose_data[7, 1]
        if point7x > 1 and point7y > 1:
            draw7.rectangle((point7x - r_hands, point7y - r_hands, point7x + r_hands, point7y + r_hands), 'white','white')
        one_map7 = self.transform_1d(one_map7)
        pose_map_7 = one_map7
        pose_map_7 = (pose_map_7 + 1) * 0.5

        # just for visualization
        im_pose = self.transform_1d(im_pose)

        agnostic = torch.cat([shape, im_h, pose_map], 0)

        mask_fore = torch.FloatTensor((parse_array > 0).astype(np.int))
        img_fore = im * mask_fore

        img_hole_hand = img_fore * (1 - im_cm) * (1 - parse_arm1) * (1 - parse_arm2) + img_fore * parse_arm1 + img_fore * parse_arm2
        img_hole_hand_wo_arm = img_fore * (1 - im_cm) * (1 - parse_arm1) * (1 - parse_arm2)
        img_hands = img_fore * pose_map_4 * parse_arm2 + img_fore * pose_map_7 * parse_arm1
        arms = im * parse_arm2 + im * parse_arm1
        img_hole_hand_new = img_hole_hand_wo_arm + img_hands

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,  #
            'cloth_gmm': c_gmm,
            'cloth_fw': c_fw,
            'cloth_mask': cm,
            'cloth_gmm_mask': cm_gmm,
            'cloth_fw_mask': cm_fw,
            'image': im,
            'agnostic': agnostic,
            'parse_cloth': im_c,
            'shape': shape,
            'head': im_h,
            'pose_image': im_pose,
            'pose_map': pose_map,
            'grid_image': im_g,
            'parse_cloth_mask': im_cm,
            'im_parse_1d' :im_parse_1d,
            'im_parse_torch': im_parse_torch,
            'label_wo_arm': label_wo_arm,
            'img_hole_hand_wo_arm': img_hole_hand_wo_arm,
            'img_hole_hand': img_hole_hand,
            'parse_arm1': parse_arm1,
            'parse_arm2': parse_arm2,
            'parse_arm': parse_arm,
            'arm1_3d': arm1_3d,
            'arm2_3d': arm2_3d,
            'mask_fore': mask_fore,
            'img_fore': img_fore,
            'parse_cross_entropy2d': parse_cross_entropy2d,
            'img_hands': img_hands,
            'arms': arms,
            'img_hole_hand_new': img_hole_hand_new,
        }

        return result

    def __len__(self):
        return len(self.im_names)


class DPDataLoader(object):
    def __init__(self, opt, dataset):
        super(DPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    opt = parser.parse_args()
    dataset = DPDataset(opt)
    data_loader = DPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()


