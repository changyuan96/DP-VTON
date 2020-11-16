# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import time
from dp_dataset import DPDataset, DPDataLoader
from networks import GMM, feature_warping, Refine, dou_U_Net, Discriminator, VGGLoss, GANLoss, TCLoss, load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
import numpy as np
np.set_printoptions(threshold=np.inf)
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )

    return loss

def generate_discrete_label(inputs, label_nc, onehot=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot: # onehot=False时执行语句
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    return input_label

def encode_input(label_map, clothes_mask, all_clothes_label):
    size = label_map.size()
    oneHot_size = (size[0], 20, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    masked_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    masked_label = masked_label.scatter_(1, (label_map * (1 - clothes_mask)).data.long().cuda(), 1.0)

    c_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    c_label = c_label.scatter_(1, all_clothes_label.data.long().cuda(), 1.0)

    input_label = Variable(input_label)

    return input_label, masked_label, c_label

def train_gmm(opt, train_loader, model_g, model_d, board):
    model_g.cuda()
    model_g.train()
    model_d.cuda()
    model_d.train()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionGAN = GANLoss()
    criterionVGG = VGGLoss()
    tcloss = TCLoss(opt)
    # optimizer
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        input = torch.cat([agnostic, c], 1)

        grid, theta = model_g(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='border')
        warped_cloth = nn.Tanh()(warped_cloth)
        warped_mask = nn.Sigmoid()(warped_mask)

        Ltc = tcloss(grid)
        Ltc = Ltc / (grid.shape[0] * grid.shape[1] * grid.shape[2])

        pred_fake = model_d(torch.cat((input.detach(), warped_cloth), dim=1))
        loss_g_gan = criterionGAN(pred_fake, True)
        L1_loss_cloth = criterionL1(warped_cloth, im_c)
        L1_loss_mask = criterionL1(warped_mask, im_cm)
        L1_loss = torch.mean(L1_loss_cloth + L1_loss_mask)
        vgg_loss = criterionVGG(warped_cloth, im_c)
        loss_g = loss_g_gan + L1_loss + vgg_loss + Ltc

        pred_fake_pool = model_d(torch.cat((input.detach(), warped_cloth.detach()), dim=1))
        loss_d_fake = criterionGAN(pred_fake_pool, False)
        pred_real = model_d(torch.cat((input.detach(), im_c), dim=1))
        loss_d_real = criterionGAN(pred_real, True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if (step + 1) % opt.display_count == 0:
            board.add_scalar('metric', loss_g.item(), step + 1)
            board.add_scalar('metric', loss_d.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_g.item()), flush=True)
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_d.item()), flush=True)
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_gmm_step_%06d.pth' % (step + 1)))
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_gmm_step_%06d.pth' % (step + 1)))

def train_feature_warping(opt, train_loader, model_gmm, model_g, model_d, board):
    model_gmm.cuda()
    model_gmm.eval()
    model_g.cuda()
    model_g.train()
    model_d.cuda()
    model_d.train()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionGAN = GANLoss()
    criterionVGG = VGGLoss()
    # optimizer
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()

        grid, theta = model_gmm(agnostic, c)
        warped_cloth_gmm = F.grid_sample(c, grid, padding_mode='border')
        warped_mask_gmm = F.grid_sample(cm, grid, padding_mode='border')

        warped_cloth_g = model_g(c, theta)
        warped_cloth = warped_cloth_g[:, 0:3, :, :]
        warped_cloth = nn.Tanh()(warped_cloth)
        composition_mask = warped_cloth_g[:, 3, :, :]
        composition_mask = nn.Sigmoid()(composition_mask)
        composition_warped_cloth = warped_cloth.detach() * (1 - composition_mask).unsqueeze(1) + (composition_mask.unsqueeze(1)) * warped_cloth_gmm.detach()

        pred_fake = model_d(torch.cat((c.detach(), warped_cloth), dim=1))
        loss_g_gan = criterionGAN(pred_fake, True)
        L1_loss_cloth = criterionL1(warped_cloth, im_c)
        L1_loss_composition_cloth = criterionL1(composition_warped_cloth, im_c)
        L1_loss_composition_mask = criterionL1(composition_mask.unsqueeze(1), im_cm)
        L1_loss = torch.mean(L1_loss_cloth + L1_loss_composition_cloth + L1_loss_composition_mask)
        loss_g_vgg_cloth = criterionVGG(warped_cloth, im_c)
        loss_g_vgg_composition_cloth = criterionVGG(composition_warped_cloth, im_c)
        loss_g_vgg = loss_g_vgg_cloth + loss_g_vgg_composition_cloth
        loss_g = loss_g_gan + L1_loss + loss_g_vgg

        pred_fake_pool = model_d(torch.cat((c.detach(), warped_cloth.detach()), dim=1))
        loss_d_fake = criterionGAN(pred_fake_pool, False)
        pred_real = model_d(torch.cat((c.detach(), im_c), dim=1))
        loss_d_real = criterionGAN(pred_real, True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if (step + 1) % opt.display_count == 0:
            board.add_scalar('metric', loss_g.item(), step + 1)
            board.add_scalar('metric', loss_d.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_g.item()), flush=True)
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_d.item()), flush=True)
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_fw_step_%06d.pth' % (step + 1)))
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_fw_step_%06d.pth' % (step + 1)))

def train_predict(opt, train_loader, model_g, board):
    model_g.cuda()
    model_g.train()
    # criterion
    criterionL1 = nn.L1Loss()
    # optimizer
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        c_gmm = inputs['cloth_gmm'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        pose_map = inputs['pose_map'].cuda()
        label_wo_arm = inputs['label_wo_arm'].cuda()
        im_parse_torch = inputs['im_parse_torch'].cuda()
        im_parse_torch_3d = im_parse_torch.squeeze(1)

        input_label, masked_label, label_wo_arm= encode_input(im_parse_torch, im_cm, label_wo_arm)
        predict = torch.cat([c_gmm, label_wo_arm, pose_map], 1)

        seg = model_g(predict)
        seg = nn.Sigmoid()(seg)
        seg_map = generate_discrete_label(seg, 20, False)
        CE_loss = cross_entropy2d(seg, im_parse_torch_3d.long())
        L1_loss = criterionL1(seg_map, im_parse_torch)
        loss_g = CE_loss * 10 + L1_loss

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if (step + 1) % opt.display_count == 0:
            board.add_scalar('metric', loss_g.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_g.item()), flush=True)
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_predict_step_%06d.pth' % (step + 1)))

def train_arm(opt, train_loader, model_predict, model_g, model_d, board):
    model_predict.cuda()
    model_predict.eval()
    model_g.cuda()
    model_g.train()
    model_d.cuda()
    model_d.train()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionGAN = GANLoss()
    criterionVGG = VGGLoss()
    # optimizer
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        arms = inputs['arms'].cuda()
        c_gmm = inputs['cloth_gmm'].cuda()
        pose_map = inputs['pose_map'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        label_wo_arm = inputs['label_wo_arm'].cuda()
        im_parse_torch = inputs['im_parse_torch'].cuda()
        img_hands = inputs['img_hands'].cuda()

        input_label, masked_label, label_wo_arm = encode_input(im_parse_torch, im_cm, label_wo_arm)
        input_predict = torch.cat([c_gmm, label_wo_arm, pose_map], 1)
        seg = model_predict(input_predict)
        seg = nn.Sigmoid()(seg)
        seg_map = generate_discrete_label(seg, 20, False)
        seg_arm = torch.FloatTensor((seg_map.cpu().numpy() == 14).astype(np.float)).cuda() + \
                  torch.FloatTensor((seg_map.cpu().numpy() == 15).astype(np.float)).cuda()

        input = torch.cat([img_hands, seg_arm], 1)

        fake_arms = model_g(input)
        fake_arms = nn.Tanh()(fake_arms)

        L1_loss = criterionL1(fake_arms, arms)
        vgg_loss = criterionVGG(fake_arms, arms)
        pred_fake = model_d(torch.cat((input.detach(), fake_arms), dim=1))
        loss_g_gan = criterionGAN(pred_fake, True)
        loss_g = loss_g_gan + L1_loss + vgg_loss

        pred_fake_pool = model_d(torch.cat((input.detach(), fake_arms.detach()), dim=1))
        loss_d_fake = criterionGAN(pred_fake_pool, False)
        pred_real = model_d(torch.cat((input.detach(), arms), dim=1))
        loss_d_real = criterionGAN(pred_real, True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if (step + 1) % opt.display_count == 0:
            board.add_scalar('metric', loss_g.item(), step + 1)
            board.add_scalar('metric', loss_d.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_g.item()), flush=True)
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss_d.item()), flush=True)
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_arm_step_%06d.pth' % (step + 1)))
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_arm_step_%06d.pth' % (step + 1)))

def train_tom(opt, train_loader, model_predict, model_arm, model_g, model_d, board):
    model_predict.cuda()
    model_predict.eval()
    model_arm.cuda()
    model_arm.eval()
    model_g.cuda()
    model_g.train()
    model_d.cuda()
    model_d.train()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionGAN = GANLoss()
    criterionVGG = VGGLoss()
    # optimizer
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        c_gmm = inputs['cloth_gmm'].cuda()
        c_fw = inputs['cloth_fw'].cuda()
        pose_map = inputs['pose_map'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        label_wo_arm = inputs['label_wo_arm'].cuda()
        img_hole_hand_wo_arm = inputs['img_hole_hand_wo_arm'].cuda()
        im_parse_torch = inputs['im_parse_torch'].cuda()
        img_hands = inputs['img_hands'].cuda()

        input_label, masked_label, label_wo_arm= encode_input(im_parse_torch, im_cm, label_wo_arm)
        input_predict = torch.cat([c_gmm, label_wo_arm, pose_map], 1)
        seg = model_predict(input_predict)
        seg = nn.Sigmoid()(seg)
        seg_map = generate_discrete_label(seg, 20, False)
        seg_arm = torch.FloatTensor((seg_map.cpu().numpy() == 14).astype(np.float)).cuda() + \
                  torch.FloatTensor((seg_map.cpu().numpy() == 15).astype(np.float)).cuda()

        input_arm = torch.cat([img_hands, seg_arm], 1)
        fake_arms = model_arm(input_arm)
        fake_arms = nn.Tanh()(fake_arms)

        input = torch.cat([img_hole_hand_wo_arm, fake_arms, seg, c_fw], 1)
        input1 = torch.cat([seg, c_fw], 1)
        input2 = torch.cat([img_hole_hand_wo_arm, fake_arms], 1)

        fake = model_g(input1, input2)
        fake_image = fake[:, 0:3, :, :]
        fake_image = nn.Tanh()(fake_image)
        fake_mask = fake[:, 3, :, :]
        fake_mask = nn.Sigmoid()(fake_mask)
        composition_fake_image = fake_image.detach() * (1 - fake_mask).unsqueeze(1) + (fake_mask.unsqueeze(1)) * c_fw.detach()

        pred_fake = model_d(torch.cat((input.detach(), fake_image), dim=1))
        loss_g_gan = criterionGAN(pred_fake, True)
        L1_loss_cloth = criterionL1(fake_image, im)
        L1_loss_composition_cloth = criterionL1(composition_fake_image, im)
        L1_loss_composition_mask = criterionL1(fake_mask.unsqueeze(1),im_cm)
        L1_loss = torch.mean(L1_loss_cloth + L1_loss_composition_cloth + L1_loss_composition_mask)
        loss_g_vgg_cloth = criterionVGG(fake_image, im)
        loss_g_vgg_composition_cloth = criterionVGG(composition_fake_image, im)
        loss_g_vgg = loss_g_vgg_cloth + loss_g_vgg_composition_cloth
        loss_g = loss_g_gan + L1_loss + loss_g_vgg

        pred_fake_pool = model_d(torch.cat((input.detach(), fake_image.detach()), dim=1))
        loss_d_fake = criterionGAN(pred_fake_pool, False)
        pred_real = model_d(torch.cat((input.detach(), im), dim=1))
        loss_d_real = criterionGAN(pred_real, True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss_g: %.4f, loss_d: %.4f'
                  % (step + 1, t, loss_g.item(), loss_d.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_tom_step_%06d.pth' % (step + 1)))
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_tom_step_%06d.pth' % (step + 1)))

def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # create dataset
    train_dataset = DPDataset(opt)

    # create dataloader
    train_loader = DPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model_g = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_g, opt.checkpoint)
        model_d = Discriminator(28)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_d, opt.checkpoint)
        train_gmm(opt, train_loader, model_g, model_d, board)
        save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_gmm_final.pth'))
        save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_gmm_final.pth'))
    elif opt.stage == 'feature_warping':
        model_gmm = GMM(opt)
        load_checkpoint(model_gmm, 'checkpoints/gmm_train_new/g_gmm_final.pth')
        model_g = feature_warping(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_g, opt.checkpoint)
        model_d = Discriminator(6)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_d, opt.checkpoint)
        train_feature_warping(opt, train_loader, model_gmm, model_g, model_d, board)
        save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_fw_final.pth'))
        save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_fw_final.pth'))
    elif opt.stage == 'predict':
        model_g = Refine(41, 20)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_g, opt.checkpoint)
        train_predict(opt, train_loader, model_g, board)
        save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_predict_final.pth'))
    elif opt.stage == 'arm':
        model_predict = Refine(41, 20)
        load_checkpoint(model_predict, 'checkpoints/predict_train_new/g_predict_final.pth')
        model_g = Refine(4, 3)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_g, opt.checkpoint)
        model_d = Discriminator(7)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_d, opt.checkpoint)
        train_arm(opt, train_loader, model_predict, model_g, model_d, board)
        save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_arm_final.pth'))
        save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_arm_final.pth'))
    elif opt.stage == 'TOM':
        model_predict = Refine(41, 20)
        load_checkpoint(model_predict, 'checkpoints/predict_train_new/g_predict_final.pth')
        model_arm = Refine(4, 3)
        load_checkpoint(model_arm, 'checkpoints/arm_train_new/g_arm_final.pth')
        model_g = dou_U_Net(23, 6, 4)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_g, opt.checkpoint)
        model_d = Discriminator(32)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model_d, opt.checkpoint)
        train_tom(opt, train_loader, model_predict, model_arm, model_g, model_d, board)
        save_checkpoint(model_g, os.path.join(opt.checkpoint_dir, opt.name, 'g_tom_final.pth'))
        save_checkpoint(model_d, os.path.join(opt.checkpoint_dir, opt.name, 'd_tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
