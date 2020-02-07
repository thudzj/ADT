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
from generator import define_G, get_scheduler, set_requires_grad, Encoder
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
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
parser.add_argument('--net_G', type=str, default='resnet_3blocks', 
                    help='net for G')
parser.add_argument('--opt_G', type=str, default='adam', 
                    help='optimizer for G')
parser.add_argument('--lr_G', type=float, default=0.0002, 
                    help='initial learning rate for adam')
parser.add_argument('--lr_policy_G', type=str, default='linear', 
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters_G', type=int, default=30, 
                    help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--niter_G', type=int, default=100, 
                    help='# of iter at starting learning rate')
parser.add_argument('--niter_decay_G', type=int, default=50, 
                    help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1_G', type=float, default=0.5, 
                    help='momentum term of adam')
parser.add_argument('--load_clf', default=None, 
                    help='load_clf')
parser.add_argument('--ngf_G', type=int, default=256, 
                    help='# ')
parser.add_argument('--loss_type', type=str, default='normal', choices=['normal', 'trades'], 
                    help='Use which loss to produce perturbations')
parser.add_argument('--norm_G', type=str, default='batch', choices=['batch', 'cbn', 'instance'], 
                    help='Use which to norm')
parser.add_argument('--z_dim', type=int, default=64, 
                    help='z_dim')
parser.add_argument('--lambda', type=float, default=1., 
                    help='entropy weight')
parser.add_argument('--entropy_th', type=float, default=0.9, 
                    help='entropy_th')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    help='dataset')

args = parser.parse_args()

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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
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

def train(args, model, device, train_loader, optimizer, epoch, G, optimizer_G, encoder, optimizer_encoder):
    g_loss, g_loss_robust = [], []
    g_loss.append(AverageMeter()); g_loss_robust.append(AverageMeter());
    c_loss, c_loss_robust, entropies, loss1, loss2 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    model.train()
    G.train()
    encoder.train()

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

        rand_z = torch.rand(data.size(0), args.z_dim, device='cuda')*2.-1.
        adv = G(torch.cat([data, grad, grad_2], 1), target, rand_z).tanh()

        model.train()
        optimizer.zero_grad()
        optimizer_G.zero_grad()
        optimizer_encoder.zero_grad()

        logits_z = encoder(adv)
        mean_z, var_z = logits_z[:, :args.z_dim], F.softplus(logits_z[:, args.z_dim:])
        neg_entropy_ub = -(-((rand_z - mean_z) ** 2) / (2 * var_z+1e-8) - (var_z+1e-8).log()/2. - math.log(math.sqrt(2 * math.pi))).mean(1).mean(0)
        entropies.update(neg_entropy_ub.item(), bs)

        x_adv = torch.clamp(data + args.epsilon * torch.clamp(adv, -1, 1), 0.0, 1.0)
        x_adv.register_hook(grad_inv)
        logits = model(x_adv)
        if args.loss_type == 'normal':
            loss_robust = F.cross_entropy(logits, target)
            loss = loss_robust
        elif args.loss_type == 'trades':
            logits_n = model(data)
            loss_natural = F.cross_entropy(logits_n, target)
            loss_robust = (1.0 / len(target)) * nn.KLDivLoss(size_average=False)(F.log_softmax(logits, dim=1),
                                                            F.softmax(logits_n, dim=1))
            loss = loss_natural + args.beta * loss_robust
        else:
            raise NotImplementedError

        (loss + F.relu(neg_entropy_ub-args.entropy_th)*args.lambda).backward()
        optimizer.step()
        optimizer_G.step()
        optimizer_encoder.step()

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

def eval(model, G, device, train_loader, log_header='Train'):
    model.eval()
    G.eval()
    train_loss = 0
    correct = 0
    train_loss_adv = 0
    correct_adv = 0
    train_loss_pgd = 0
    correct_pgd = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        train_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if not 'Train' in log_header:
            data.requires_grad_()
            grad = torch.autograd.grad(F.cross_entropy(model(data), target), [data])[0].detach()
            x_adv = torch.clamp(data + args.epsilon * generate_adv(G(torch.cat([data, grad], 1))), 0.0, 1.0).detach()
            output_adv = model(x_adv)
            train_loss_adv += F.cross_entropy(output_adv, target, size_average=False).item()
            pred_adv = output_adv.max(1, keepdim=True)[1]
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

            x_pgd = data.detach() + torch.FloatTensor(*data.shape).uniform_(-args.epsilon, args.epsilon).cuda().detach()
            x_pgd = torch.clamp(x_pgd, 0.0, 1.0)
            for _ in range(args.num_steps):
                x_pgd.requires_grad_()
                with torch.enable_grad():
                    loss = nn.CrossEntropyLoss()(model(x_pgd), target)
                grad = torch.autograd.grad(loss, [x_pgd])[0]
                x_pgd = x_pgd.detach() + args.step_size * torch.sign(grad.detach())
                x_pgd = torch.min(torch.max(x_pgd, data - args.epsilon), data + args.epsilon)
                x_pgd = torch.clamp(x_pgd, 0.0, 1.0)
            output_pgd = model(x_pgd)
            train_loss_pgd += F.cross_entropy(output_pgd, target, size_average=False).item()
            pred_pgd = output_pgd.max(1, keepdim=True)[1]
            correct_pgd += pred_pgd.eq(target.view_as(pred_pgd)).sum().item()


    train_loss /= len(train_loader.dataset)
    train_loss_adv /= len(train_loader.dataset)
    train_loss_pgd /= len(train_loader.dataset)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(log_header,
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    print('{}: Average loss adv: {:.4f}, Accuracy adv: {}/{} ({:.0f}%)'.format(log_header,
        train_loss_adv, correct_adv, len(train_loader.dataset),
        100. * correct_adv / len(train_loader.dataset)))
    print('{}: Average loss pgd: {:.4f}, Accuracy pgd: {}/{} ({:.0f}%)'.format(log_header,
        train_loss_pgd, correct_pgd, len(train_loader.dataset),
        100. * correct_pgd / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    print(args)
    # init model, ResNet18() can be also used here for training
    model = WideResNet(depth=28, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    G = define_G(9 + 1, 3, args.ngf_G, args.net_G, True, norm=args.norm_G, no_down_G=True, use_relu_atlast=True, outs=1, use_dropout=False, z_dim=args.z_dim)
    
    if args.opt_G == 'adam':
        optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(args.beta1_G, 0.999))
    elif args.opt_G == 'sgd':
        optimizer_G = torch.optim.SGD(G.parameters(), lr=args.lr_G, weight_decay=1e-4)
    else:
        raise NotImplementedError
    scheduler_G = get_scheduler(optimizer_G, args)

    encoder = Encoder(args.z_dim).cuda()
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=2e-4)

    print('  + Number of params of classifier: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('  + Number of params of generator: {}'.format(sum([p.data.nelement() for p in G.parameters()])))
    print('  + Number of params of generator: {}'.format(sum([p.data.nelement() for p in encoder.parameters()])))

    if args.load_clf:
        model.load_state_dict(torch.load(args.load_clf))
        G.load_state_dict(torch.load(args.load_clf.replace('model-wideres', 'generator')))
        debug(args, model, device, train_loader, optimizer, 0, G, optimizer_G)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        time_s = time.time()
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, G, optimizer_G, encoder, optimizer_encoder)
        scheduler_G.step()

        print(time.time()-time_s)

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch > 70:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
            torch.save(G.state_dict(),
                       os.path.join(model_dir, 'generator-epoch{}.pt'.format(epoch)))
            torch.save(optimizer_G.state_dict(),
                       os.path.join(model_dir, 'optG_epoch{}.tar'.format(epoch)))
            torch.save(encoder.state_dict(),
                       os.path.join(model_dir, 'encoder-epoch{}.pt'.format(epoch)))
            torch.save(optimizer_encoder.state_dict(),
                       os.path.join(model_dir, 'opt-encoder_epoch{}.tar'.format(epoch)))

if __name__ == '__main__':
    main()
