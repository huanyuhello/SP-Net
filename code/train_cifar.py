'''Train CIFAR10/100 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
import pathspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from distiller import Distiller
from model import DR_ResNet110_cifar, DR_ResNet32_cifar
from earlypredict_model import EP_ResNet_cifar
import math
import numpy as np
import warnings
from utils import *
from tensorboardX import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
warnings.filterwarnings('ignore')

print("pidnum:",os.getpid())

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=350, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lrdecay', default=30, type=int,
                    help='epochs to decay lr')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--lossfact', default=2.0, type=float,
                    help='loss factor')
parser.add_argument('--gateloss', default=0.1, type=float,
                    help='loss factor')
parser.add_argument('--cos_margin', default=0., type=float,
                    help='initial learning rate')
parser.add_argument('--cos_loss', default=1e-4, type=float,
                    help='initial learning rate')                               

parser.add_argument('--target', default=0.8, type=float, help='target rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--distillation_momentum', default=0.99, type=float, help='distillation_momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='folder path to save checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--print-freq', '-p', default=25, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--expname', default='give_me_a_name', type=str, metavar='n',
                    help='name of experiment (default: test')
parser.add_argument('--pretrain_model', default='pretrain/cifar10_resnet110_raw.pth', type=str)       

parser.add_argument('--target_list', type=float, nargs='+',default=[0.33,0.33,0.33], help='for early stop ')

parser.add_argument('--dataset', default='cifar??', type=str, metavar='n')
parser.add_argument('--model', default='res??', type=str, metavar='n')

parser.set_defaults(test=False)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print(args)

    save_path = "runs/%s/"%(args.expname)

    if not args.test:
        cp_projects(save_path)
    logger = SummaryWriter(save_path)

    torch.cuda.manual_seed(args.seed)
        
    # set the target rates for each layer
    # the default is to use the same target rate for each layer
    # target_rates_list = ([1,1]+ [args.target] * 14 +[1,1])*3
    target_rates_list = [args.target] * 54
    target_rates = {i:target_rates_list[i] for i in range(len(target_rates_list))}
    print(target_rates)
    print(args.dataset, args.model)

    # Cifar Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    kwargs = {'num_workers': 2, 'pin_memory': True}


    model_path = "None"

    if args.dataset == "cifar10":
        num_class = 10
        ep_model = EP_ResNet_cifar(nclass=10)
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('//4T/wenhu/dataset/cifar', train=True, download=True,
                            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/4T/wenhu/dataset/cifar', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        if args.model == "ResNet110":
            stage = [18,18,18]
            teacher = DR_ResNet110_cifar(nclass=10)
            model = DR_ResNet110_cifar(nclass=10)
            model_path = 'pretrain/cifar10_resnet110.pth'
        elif args.model == "ResNet32":
            stage = [5,5,5]
            teacher = DR_ResNet32_cifar(nclass=10)
            model = DR_ResNet32_cifar(nclass=10)
            model_path = 'pretrain/cifar10_resnet32.pth'

    elif args.dataset == "cifar100":
        num_class = 100
        ep_model = EP_ResNet_cifar(nclass=100)
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/home/huanyu/dataset', train=True, download=True,
                            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/home/huanyu/dataset', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        if args.model == "ResNet110":
            stage = [18,18,18]
            teacher = DR_ResNet110_cifar(nclass=100)
            model = DR_ResNet110_cifar(nclass=100)
            model_path = 'pretrain/cifar100_resnet110.pth'
        elif args.model == "ResNet32":
            stage = [5,5,5]
            teacher = DR_ResNet32_cifar(nclass=100)
            model = DR_ResNet32_cifar(nclass=100)
            model_path = 'pretrain/cifar100_resnet32.pth'
    else:
        print("error!!!!")
    cudnn.benchmark = True

    teacher = torch.nn.DataParallel(teacher).cuda()
    model = torch.nn.DataParallel(model).cuda()
    ep_model = torch.nn.DataParallel(ep_model).cuda()
  
    if args.pretrained:
        print("load pretrain model")
        pretrained_dict = load_student(model_path, model.state_dict())
        model.load_state_dict(pretrained_dict) 

    # define loss function (criterion) 
    CE = nn.CrossEntropyLoss().cuda()
    KD = nn.MSELoss().cuda()
    # KD = nn.KLDivLoss().cuda()
    
    
    if args.resume:
        latest_checkpoint = os.path.join(args.resume, 'ckpt.pth')
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(latest_checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})(acc {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # get the number of model parameters
    print('Number of main_model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        checkpoint = torch.load("model/ckpt.pth")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})(acc {})"
                .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        test_acc, act = validate(valid_loader, model, CE, 60, target_rates,logger)
        sys.exit()

    
    fc1 = 17 * num_class  + (num_class+1) * num_class + (num_class+1) * 2
    fc2 = fc1+ 33 * num_class  + (num_class+1) * num_class + (num_class+1) * 2
    fc3 = fc2+ 65 * num_class

    #4721026* 18 + 3543426 + 4723074 * 17 +  3547522 + 4727170 * 17
    flops_a =        [9 *16*16 *32*32 * 2  + 17*128 + 129*2] * stage[0]\
                    +[9 *16*32 *16*16 + 9 *32*32 *16*16   + 33*128 + 129*2]\
                    +[9 *32*32 *16*16 *2  + 33*128 + 129*2] * (stage[1]-1)\
                    +[9 *32*64 *8*8 + 9 *64*64 *8*8   + 65*128 + 129*2]\
                    +[9 *64*64 *8*8 *2  + 65*128 + 129*2]* (stage[2]-1)
    flops_b = [fc1,fc2,fc3]
    

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! First stage training!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters()],
                            'lr': args.lr, 'weight_decay': args.weight_decay}], 
                            momentum=args.momentum)

    Distiller_model = Distiller(student = model, teacher = teacher, momentum = args.distillation_momentum)

    activation=-1
    gflops = -1
    for epoch in range(args.start_epoch, args.epochs):

        
        adjust_learning_rate(optimizer, epoch)
        # adjust_distillation_momentum(Distiller_model, epoch)
        print("lr:",optimizer.param_groups[0]['lr'])

        # train for one epoch
        train_main(train_loader, Distiller_model, CE, KD, optimizer, epoch, target_rates,  logger, args)

        # evaluate on validation set
        prec1 , new_activation, new_gflops= validate_main(valid_loader, model, CE, epoch, target_rates, logger, flops_a, flops_b)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        activation = new_activation if prec1 > best_prec1 else activation
        gflops = new_gflops if prec1 > best_prec1 else gflops
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'activation': activation,
            'gflops': gflops,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, save_path+"/stage1_model")

        print('Best accuracy: ', best_prec1)
        print('Best activation: ', activation)
        print('Best GFlops: ', gflops)
        logger.add_scalar('best/accuracy', best_prec1, global_step=epoch)
        logger.add_scalar('best/activation', activation, global_step=epoch)
        logger.add_scalar('best/GFlops', gflops, global_step=epoch)

    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!traingning finish!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print('Best accuracy: ', best_prec1)
    print('Best activation: ', activation)
    print('Best GFlops: ', gflops)
    print('\n')


    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! All Finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')




def train_main(train_loader, model, CE, KD, optimizer, epoch, target_rates,  logger, agrs):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_t = AverageMeter()
    losses_k = AverageMeter()
    losses_g = AverageMeter()

    losses_cos = AverageMeter()
    cos_sim0 = AverageMeter()
    cos_sim1= AverageMeter()
    cos_sim2 = AverageMeter()

    top1 = AverageMeter()
    activations = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (inputs, target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        target = target.cuda()
        inputs = inputs.cuda()

        # compute output
        s_out, t_out = model(inputs, temperature=1)

        s_outputs, diff_list, s_activation_rates , s_gate_logits = s_out
        t_outputs, _, t_activation_rates, t_gate_logits = t_out

        loss_f0 = F.cosine_embedding_loss(diff_list[0][0],diff_list[0][1], torch.tensor([1], device=s_outputs[0].device), margin=args.cos_margin)
        loss_f1 = F.cosine_embedding_loss(diff_list[1][0],diff_list[1][1], torch.tensor([1], device=s_outputs[0].device), margin=args.cos_margin)
        loss_f2 = F.cosine_embedding_loss(diff_list[2][0],diff_list[2][1], torch.tensor([1], device=s_outputs[0].device), margin=args.cos_margin)

        # F.cosine_embedding_loss(a,a, torch.tensor([1]), reduction='none')

        cosin0 = torch.cosine_similarity(diff_list[0][0],diff_list[0][1],dim=1).mean()
        cosin1 = torch.cosine_similarity(diff_list[1][0],diff_list[1][1],dim=1).mean()
        cosin2 = torch.cosine_similarity(diff_list[2][0],diff_list[2][1],dim=1).mean()
        cos_sim0.update(to_python_float(cosin0), inputs.size(0))
        cos_sim1.update(to_python_float(cosin1), inputs.size(0))
        cos_sim2.update(to_python_float(cosin2), inputs.size(0))

        
        output = s_outputs[0]

        # A loss
        loss_c1 = CE(s_outputs[0], target) * 1.0
        loss_c2 = CE(s_outputs[1], target) * 0.2
        loss_c3 = CE(s_outputs[2], target) * 0.1

        # loss_k1 = KD(F.softmax(s_outputs[0],dim=-1), F.softmax(t_outputs[0],dim=-1)) * 1.0
        # loss_k2 = KD(F.softmax(s_outputs[1],dim=-1), F.softmax(t_outputs[0],dim=-1)) * 0.2
        # loss_k3 = KD(F.softmax(s_outputs[2],dim=-1), F.softmax(t_outputs[0],dim=-1)) * 0.1
        loss_k1 = KD(s_outputs[0], t_outputs[0]) * 1.0
        loss_k2 = KD(s_outputs[1], t_outputs[0]) * 0.2
        loss_k3 = KD(s_outputs[2], t_outputs[0]) * 0.1
        # loss_k1 = KD( F.log_softmax(s_outputs[0],dim=1), F.softmax(t_outputs[0], dim=1) )* 1.0
        # loss_k2 = KD( F.log_softmax(s_outputs[1],dim=1), F.softmax(t_outputs[1], dim=1) )* 0.2
        # loss_k3 = KD( F.log_softmax(s_outputs[2],dim=1), F.softmax(t_outputs[2], dim=1) )* 0.1
        


        ###### ce  -- kd   -- fe -- gate
        loss_ce = loss_c1 + loss_c2 + loss_c3
        loss_kd = (loss_k1 + loss_k2 + loss_k3) * 0.1
        # loss_gate = KD(F.softmax(s_gate_logits,dim=-1), F.softmax(t_gate_logits,dim=-1)) * 0.1
        loss_gate = KD(s_gate_logits, t_gate_logits) * 0.1
        # loss_gate = KD( F.log_softmax(s_gate_logits,dim=1), F.softmax(t_gate_logits, dim=1) )* 0.1

        
        # target rate loss
        acts = 0
        acts_plot = 0
        for j, act in enumerate(s_activation_rates):
            if target_rates[j] < 1:
                acts_plot += torch.mean(act)
                acts += torch.pow(target_rates[j] - torch.mean(act), 2)
            else:
                acts_plot += 1
        
        # this is important when using data DataParallel
        acts_plot = torch.mean(acts_plot / len(s_activation_rates))
        acts = torch.mean(acts / len(s_activation_rates))

        act_loss = args.lossfact * acts

        loss_cos = loss_f0 * 1. + loss_f1*1. + loss_f2 * 1.
        loss_cos = loss_cos * args.cos_loss

        # loss_kd, loss_gate = loss_ce*0.0, loss_ce*0.0
        loss = loss_ce + act_loss + loss_kd  + loss_gate  + loss_cos

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(to_python_float(loss), inputs.size(0))
        losses_c.update(to_python_float(loss_ce), inputs.size(0))
        losses_t.update(to_python_float(act_loss), inputs.size(0))
        losses_k.update(to_python_float(loss_kd), inputs.size(0))
        losses_g.update(to_python_float(loss_gate), inputs.size(0))
        losses_cos.update(to_python_float(loss_cos), inputs.size(0))


        top1.update(prec1[0], inputs.size(0))
        activations.update(to_python_float(acts_plot), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})k({lossk.avg:.4f})cos({cosloss.avg:.4f})g({lossg.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,lossk = losses_k, cosloss = losses_cos, lossg = losses_g,
                      loss=losses, lossa=losses_t, lossc=losses_c, top1=top1, act=activations))
            
            logger.add_scalar('train-1/losses', losses.avg, global_step=global_step)
            logger.add_scalar('train-1/losses_ce', losses_c.avg, global_step=global_step)
            logger.add_scalar('train-1/losses_kd', losses_k.avg, global_step=global_step)
            logger.add_scalar('train-1/losses_gate', losses_g.avg, global_step=global_step)
            logger.add_scalar('train-1/losses_t', losses_t.avg, global_step=global_step)
            logger.add_scalar('train-1/activations', activations.avg, global_step=global_step)
            logger.add_scalar('train-1/top1', top1.avg, global_step=global_step)
            logger.add_scalar('train-1/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            logger.add_scalar('train-1/distiller_m', model.momentum, global_step=global_step)

            logger.add_scalar('train-1/losses_cosine', losses_cos.avg, global_step=global_step)
            logger.add_scalar('train-1/cos0', cos_sim0.avg, global_step=global_step)
            logger.add_scalar('train-1/cos1', cos_sim1.avg, global_step=global_step)
            logger.add_scalar('train-1/cos2', cos_sim2.avg, global_step=global_step)



def validate_main(valid_loader, model, CE, epoch, target_rates, logger, flops_a, flops_b):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    activations = AverageMeter()
    gflops = AverageMeter()
    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(valid_loader):
        target = target.cuda()
        inputs = inputs.cuda()

        with torch.no_grad():
            outputs, s_features, activation_rates, _ = model(inputs, temperature=1)
           
            output = outputs[0]

            loss = CE(output, target)
        acts = 0
        for j, act in enumerate(activation_rates):
            if target_rates[j] < 1:
                acts += torch.mean(act)
            else:
                acts += 1
        # this is important when using data DataParallel
        acts = torch.mean(acts / len(activation_rates))

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(to_python_float(loss), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        activations.update(to_python_float(acts), 1)

        ep_masks = [torch.zeros(inputs.size(0)).cuda()] * 2 + [torch.ones(inputs.size(0)).cuda()]
        flops, act_rates, ep_rates = get_GFLOPs(activation_rates, ep_masks, flops_a, flops_b)
        gflops.update(to_python_float(flops), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      i, len(valid_loader), batch_time=batch_time, loss=losses,
                      top1=top1, act=activations))

    logger.add_scalar('valid-1/top1', top1.avg, global_step=epoch)
    logger.add_scalar('valid-1/activations', activations.avg, global_step=epoch)
    print(' * Prec@1 {top1.avg:.3f}  Activations: {act.avg:.3f} '.format(top1=top1, act=activations))

    return top1.avg, activations.avg, gflops.avg


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr
    if epoch >= 150:
        lr = 0.1 * lr
    if epoch >= 250:
        lr = 0.1 * lr
    optimizer.param_groups[0]['lr'] = lr

def adjust_distillation_momentum(model, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    m = args.distillation_momentum

    if epoch >= 150:
        m = 1.1 * m
    if epoch >= 250:
        m = 1.1 * m
    model.momentum = m


def get_GFLOPs(activation_rates, ep_masks, flops_a, flops_b):
    flops_base = 9 *3*17  *32*32  *1 
    all_flops = flops_base

    ep_rates = [0] * len(ep_masks)
    for j, ep in enumerate(ep_masks):
        ep_rates[j] += torch.mean(ep)

    act_rates = [0] * len(flops_a)
    for j in range(len(flops_a)):
        act = activation_rates[j]
        if j < len(flops_a)//3:
            ep = torch.ones(ep_masks[0].size(0)).cuda()
        elif j < 2 * len(flops_a)//3:
            ep = 1 - ep_masks[0]
        else:
            ep = (1 - ep_masks[0]) * (1 - ep_masks[1])

        act_rates[j] += torch.mean(act * ep)

    for i in range(len(flops_b)):
        all_flops += ep_rates[i] * flops_b[i]
    
    for i in range(len(flops_a)):
        all_flops += act_rates[i] * flops_a[i]

    all_flops = all_flops * 2 / 1e9 # for * and +  

    return all_flops, act_rates, ep_rates

if __name__ == '__main__':
    main()
