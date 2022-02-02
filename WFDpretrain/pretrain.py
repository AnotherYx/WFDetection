import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from components.dataset_loader import DatasetLoader_TracesAccu as DatasetAccu
from components.meta_eval import meta_test
from components.util import create_model, accuracy, AverageMeter, CategoriesSampler


def init_option():
    option = edict()
    option['batch_size']=512
    option['epochs']=1

    option['learning_rate']=0.1
    option['gamma']=0.2
    option['step_size'] = 30
    option['momentum']=0.9
    option['weight_decay']=0.0005

    option['classifier']="LR"
    option['n_ways']=100
    option['n_shots'] = 1
    option['n_queries']=9
    option['n_gpu']=torch.cuda.device_count()

    return option


def main():
    opt = init_option()
    model_name = 'res50'
    model_path = './models'
    dataset = './data/awf_900w_2500tr_burst_int16.npz'
    dataset_meta_split = './data/tor_900w_2500tr_meta_split.npy'

    save_folder = os.path.join(model_path, model_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    source = np.load(dataset, allow_pickle=False)
    source_data = source['data']
    meta_split_file = np.load(dataset_meta_split, allow_pickle=True)
    meta_split = meta_split_file.item()

    # Load pretrain set
    trainset = DatasetAccu('train', opt, source_data, meta_split, train_aug=True)
    train_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Load meta-val set
    valset = DatasetAccu('val', opt, source_data, meta_split, train_aug=False)
    val_sampler = CategoriesSampler(valset.label, 10, opt.n_ways, opt.n_shots + opt.n_queries)
    meta_valloader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    # Set pretrain class number
    n_cls = trainset.num_class

    # model
    depth = {"res18":18,"res34":34,"res50":50}[model_name]
    model = create_model(n_cls,depth)

    # optimizer and lr_scheduler
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': opt.learning_rate}], \
            momentum=opt.momentum, nesterov=True, weight_decay=opt.weight_decay)
    # Set learning rate scheduler 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, \
        gamma=opt.gamma)        

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True


    trlog = {}
    trlog['max_meta_val_acc']=0.0
    trlog['max_meta_val_acc_epoch']=0
    trlog['max_meta_val_acc_feat'] =0.0
    trlog['max_meta_val_acc_feat_epoch']=0


    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):

        print("==> training...")
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        if lr_scheduler != None:
            lr_scheduler.step()
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        val_acc_feat = meta_validate(model, meta_valloader, opt.classifier, opt)
        # regular saving
        if epoch % 5 == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            }
            save_file = os.path.join(save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # update best saved model
        if val_acc_feat > trlog['max_meta_val_acc_feat']:
            trlog['max_meta_val_acc_feat'] = val_acc_feat
            trlog['max_meta_val_acc_feat_epoch'] = epoch
            print('==> Saving max_meta_val_acc model...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            }
            save_file = os.path.join(save_folder, 'clf_{clf}_max_meta_val_acc_feat_{val_acc_feat}_epoch_{epoch}.pth'.format(epoch=epoch,clf=opt.classifier, val_acc_feat=val_acc_feat))
            torch.save(state, save_file)
        
    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
    }
    save_file = os.path.join(save_folder, '{}_last.pth'.format(model_name))
    torch.save(state, save_file)

def meta_validate(model, meta_valloader, classifier, opt):

    start = time.time()
    val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=True, is_norm=True, 
                                    classifier=classifier, opt= opt)
    val_time = time.time() - start

    print("type(float(val_std_feat[0])): ", type(float(val_std_feat)))
    print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(float(val_acc_feat),
                                                                       float(val_std_feat),
                                                                       val_time))
    return val_acc_feat

def train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        output_logit = output['linear']
        loss = criterion(output_logit, target)

        acc1, acc5 = accuracy(output_logit, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
        # break

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
