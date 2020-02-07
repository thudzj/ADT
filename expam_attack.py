from __future__ import print_function
import os
import argparse
import numpy as np
np.set_printoptions(threshold=3000,linewidth=200)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.distributions.utils import clamp_probs
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss
from generator import define_G, get_scheduler, set_requires_grad
from fs_wideresnet import WideResNet as fs_WideResNet
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8.0/255.0,
                    help='perturbation')
parser.add_argument('--num-steps', default=5,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--net_G', type=str, default='resnet_3blocks', help='net for G')
parser.add_argument('--opt_G', type=str, default='adam', help='optimizer for G')
parser.add_argument('--num_train_G', type=int, default=1, help='# for training G')
parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy_G', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters_G', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--niter_G', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay_G', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1_G', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--no_zero_pad', action='store_true', default=False, help='no_zero_pad')
parser.add_argument('--load_clf', default=None, help='load_clf')
parser.add_argument('--down_G', action='store_true', default=False, help='down_G')
parser.add_argument('--use_relu_G', action='store_true', default=False, help='use_relu')
parser.add_argument('--ngf_G', type=int, default=256, help='# ')
parser.add_argument('--loss_type', type=str, default='normal', choices=['normal', 'trades'], help='Use which loss to produce perturbations')
parser.add_argument('--outs', type=int, default=1, help='# of out samples')
parser.add_argument('--norm_G', type=str, default='batch', choices=['batch', 'cbn', 'instance'], help='Use which to norm')
parser.add_argument('--use_dropout_G', action='store_true', default=False, help='use_dropout_G')
parser.add_argument('--enlarge-factor', type=float, default=1.0, help='Enlarge factor for perturbation')
parser.add_argument('--z_dim', type=int, default=64, help='z_dim')
parser.add_argument('--entropy', type=float, default=0.01, help='entropy weight')
parser.add_argument('--entropy_th', type=float, default=-100., help='entropy_th')
parser.add_argument('--fixed_var', action='store_true', default=False, help='fixed_var')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--pretrained_g', action='store_true', default=False, help='pretrained_g')
parser.add_argument('--dist', type=str, default='gaussian', help='dist')

args = parser.parse_args()
args.use_relu_G = True

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    args.epsilon = 4.0/255.0
    trainset = torchvision.datasets.SVHN(root='../../data', split='train', download=True, transform=transform_test)
    testset = torchvision.datasets.SVHN(root='../../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError
train_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def ListToFormattedString(alist):
    format_list = ['{:.3f}' for item in alist]
    s = ','.join(format_list)
    return s.format(*alist)

def grad_inv(grad):
    if args.loss_type == 'normal':
        return grad.neg()
    elif args.loss_type == 'trades':
        return grad.neg()/args.beta

def generate_adv(adv_raw, tau=None, return_entropy=False):
    entropy = torch.cuda.FloatTensor([0.])
    nc = 3 if args.dist != 'gaussian' else 6
    if args.outs > 1:
        tmp = np.random.randint(0, args.outs)
        adv_raw = adv_raw[:, tmp*nc:(tmp+1)*nc, :, :]
    if args.dist == 'none':
        adv = torch.tanh(adv_raw)
        # flag_pos = (adv > 0.8).float()
        # adv = (adv * (1 - flag_pos) + 1. * flag_pos - adv).detach() + adv
    elif args.dist == 'gaussian':
        adv_mean = adv_raw[:, :3, :, :]
        adv_std = F.softplus(adv_raw[:, 3:, :, :])
        rand_noise = torch.randn_like(adv_std)
        adv = torch.tanh(adv_mean + rand_noise * adv_std) #torch.clamp(adv_mean + torch.randn_like(adv_std)*adv_std, -1., 1.)
        # entropy = -(adv_std+1e-8).log().mean() #F.relu(-adv_std.log().mean()+args.entropy_th)
        logp = -(rand_noise ** 2) / 2. - (adv_std + 1e-8).log() - math.log(math.sqrt(2 * math.pi)) - (-adv ** 2 + 1 + 1e-8).log()
        entropy = logp.mean() #should be called neg entropy
    elif 'bernoulli_repara' in args.dist:
        if tau is None:
            tau = args.tau
        uniforms = clamp_probs(torch.rand_like(adv_raw))
        adv = ((uniforms.log() - (-uniforms).log1p() + adv_raw) / tau).sigmoid()
        if 'hard' in args.dist:
            adv = ((adv >= 0.5).float() - adv).detach() + adv
        adv = adv * 2 - 1
    elif args.dist == 'bernoulli_st':
        adv_p = adv_raw.sigmoid()
        adv = torch.gt(adv_p,torch.rand_like(adv_p)).float()
        adv = (adv-adv_p).detach() + adv_p
        adv = adv * 2 - 1
    else:
        raise NotImplementedError
    if return_entropy:
        return adv, entropy
    else:
        return adv

def train(args, model, device, train_loader, optimizer, epoch, G, optimizer_G):
    g_loss, g_loss_robust = [], []
    for _ in range(args.num_train_G):
        g_loss.append(AverageMeter()); g_loss_robust.append(AverageMeter());
    c_loss, c_loss_robust, entropies, loss1, loss2 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # model.train()
    G.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        bs = len(data)

        model.eval()
        data.requires_grad_()
        loss1_ = F.cross_entropy(model(data), target)
        loss1.update(loss1_, bs)
        grad = torch.autograd.grad(loss1_, [data])[0].detach()
        data.requires_grad_(False)

        x_pgd = torch.clamp(data + args.epsilon * grad.sign(), 0.0, 1.0).detach()
        x_pgd.requires_grad_()
        loss2_ = F.cross_entropy(model(x_pgd), target)
        loss2.update(loss2_, bs)
        grad_2 = torch.autograd.grad(loss2_, [x_pgd])[0].detach()
        x_pgd.requires_grad_(False)

        adv_raw_2 = G(torch.cat([data, grad, grad_2], 1))

        # model.train()
        # optimizer.zero_grad()
        optimizer_G.zero_grad()

        adv2, entropy = generate_adv(adv_raw_2, return_entropy=True)
        entropies.update(entropy.item(), bs)
        #adv.register_hook(grad_inv)
        x_adv2 = torch.clamp(data + args.epsilon * torch.clamp(args.enlarge_factor * adv2, -1, 1), 0.0, 1.0)
        x_adv2.register_hook(grad_inv)
        logits = model(x_adv2)

        loss_robust = F.cross_entropy(logits, target)
        loss = loss_robust

        ((loss + entropy * args.entropy)).backward()
        # optimizer.step()
        optimizer_G.step()

        g_loss[0].update(loss.item(), bs)
        g_loss_robust[0].update(loss_robust.item(), bs)
        c_loss.update(loss.item(), bs)
        c_loss_robust.update(loss_robust.item(), bs)

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}-ite{}\tGenerator loss: {}\trobust loss: {}\n                     \tClassifier loss: {:.3f}\trobust loss: {:.3f}\n                     \tloss1: {}\tloss2: {}\tneg entropy: {}'.format(epoch, batch_idx,
                                    ListToFormattedString([item.avg for item in g_loss]),
                                    ListToFormattedString([item.avg for item in g_loss_robust]),
                                    c_loss.avg, c_loss_robust.avg, loss1.avg, loss2.avg, entropies.avg))
        if batch_idx == len(train_loader) - 1:
            model.eval()
            data = data[0:1]
            data.requires_grad_()
            grad = torch.autograd.grad(F.cross_entropy(model(data), target[0:1]), [data])[0].detach()
            data.requires_grad_(False)
            x_pgd = torch.clamp(data + args.epsilon * grad.sign(), 0.0, 1.0).detach()
            x_pgd.requires_grad_()
            grad_2 = torch.autograd.grad(F.cross_entropy(model(x_pgd), target[0:1]), [x_pgd])[0].detach()
            x_pgd.requires_grad_(False)

            data, grad, grad_2, target = data.repeat(100, 1, 1, 1), grad.repeat(100, 1, 1, 1), grad_2.repeat(100, 1, 1, 1), target[0:1].repeat(100)
            adv_raw_2 = G(torch.cat([data, grad, grad_2], 1))
            adv2, entropy = generate_adv(adv_raw_2, return_entropy=True)
            print('Sample variance: ', adv2.var(0).mean().item())

def eval(model, G, device, train_loader):
    model.eval()
    # G.eval()
    train_loss_adv = 0
    correct_adv = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        data.requires_grad_()
        loss1_ = F.cross_entropy(model(data), target)
        # loss1.update(loss1_, bs)
        grad = torch.autograd.grad(loss1_, [data])[0].detach()
        data.requires_grad_(False)

        x_pgd = torch.clamp(data + args.epsilon * grad.sign(), 0.0, 1.0).detach()
        x_pgd.requires_grad_()
        loss2_ = F.cross_entropy(model(x_pgd), target)
        # loss2.update(loss2_, bs)
        grad_2 = torch.autograd.grad(loss2_, [x_pgd])[0].detach()
        x_pgd.requires_grad_(False)

        adv_raw_2 = G(torch.cat([data, grad, grad_2], 1))

        pred_mask = 1.
        k = 100
        for _ in range(k):
            adv2, entropy = generate_adv(adv_raw_2, return_entropy=True)

            x_adv = torch.clamp(data + args.epsilon * torch.clamp(args.enlarge_factor * adv2, -1, 1), 0.0, 1.0).detach()

            with torch.no_grad():
                output_adv = model(x_adv)
            train_loss_adv += F.cross_entropy(output_adv, target, size_average=False).item()/float(k)
            pred_adv = output_adv.max(1, keepdim=True)[1]
            pred_mask *= pred_adv.eq(target.view_as(pred_adv)).float()
        correct_adv += (pred_mask > 0).sum().item()


    train_loss_adv /= len(train_loader.dataset)
    print('Average loss adv: {:.4f}, Accuracy adv: {}/{} ({:.0f}%)'.format(
        train_loss_adv, correct_adv, len(train_loader.dataset),
        100. * correct_adv / len(train_loader.dataset)))

def main():
    print(args)
    # init model, ResNet18() can be also used here for training
    if args.load_clf == 'fs':
        model = fs_WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        checkpoint = torch.load('./FS/checkpoint-199-ipot.dms')['net']
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            new_state_dict[key[17:]] = value
        model.load_state_dict(new_state_dict)
    elif args.load_clf == 'pgd':
        model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load('./checkpoint_eval/pgd7.pt'))
    elif args.load_clf == 'gaussian':
        model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load('./checkpoint_eval/gaussian-entropy-0.01-bs64.pt'))
    elif args.load_clf == 'implicit':
        model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load('./checkpoint_eval/implicit.pt'))
    elif args.load_clf == 'standard':
        model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load('./checkpoint_eval/standard.pt'))
    elif args.load_clf == 'dist':
        model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load('./checkpoint_eval/dist.pt'))
    elif args.load_clf == 'dist-new':
        model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load('./checkpoint_eval/dist-new.pt'))
    else:
        raise NotImplementedError


    # model = WideResNet(depth=28, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    G = define_G(9, 6, args.ngf_G, args.net_G, not args.no_zero_pad, norm=args.norm_G, no_down_G=not args.down_G, use_relu_atlast=args.use_relu_G, outs=args.outs, use_dropout=args.use_dropout_G)
    # print(G)

    if args.pretrained_g:
        G.load_state_dict(torch.load('generator_cifar10.pt'))
    if args.opt_G == 'adam':
        optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(args.beta1_G, 0.999))
    elif args.opt_G == 'sgd':
        optimizer_G = torch.optim.SGD(G.parameters(), lr=args.lr_G, weight_decay=1e-4)
    else:
        raise NotImplementedError
    scheduler_G = get_scheduler(optimizer_G, args)

    print('  + Number of params of classifier: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('  + Number of params of generator: {}'.format(sum([p.data.nelement() for p in G.parameters()])))

    # if args.load_clf:
    #     model.load_state_dict(torch.load(args.load_clf))
    #     G.load_state_dict(torch.load(args.load_clf.replace('model-wideres', 'generator')))
    #     debug(args, model, device, train_loader, optimizer, 0, G, optimizer_G)

    # eval(model, G, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        # adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, None, epoch, G, optimizer_G)
        # scheduler_G.step()
        if epoch > 400 and epoch % 25 == 0:
            eval(model, G, device, test_loader)

        # save checkpoint
        # if epoch % args.save_freq == 0:
        #     # torch.save(model.state_dict(),
        #     #            os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
        #     # torch.save(optimizer.state_dict(),
        #     #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
        #     torch.save(G.state_dict(),
        #                os.path.join(model_dir, 'generator-epoch{}.pt'.format(epoch)))
        #     torch.save(optimizer_G.state_dict(),
        #                os.path.join(model_dir, 'optG_epoch{}.tar'.format(epoch)))

if __name__ == '__main__':
    main()
