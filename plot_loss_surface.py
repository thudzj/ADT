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
from generator_implicit import define_G, get_scheduler, set_requires_grad, Encoder
from fs_wideresnet import WideResNet as fs_WideResNet
from collections import OrderedDict
from torch.autograd import Variable
import scipy.io

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load_clf', default=None, help='load_clf')
parser.add_argument('--epsilon', default=8.0/255.0,
                    help='perturbation')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)

def _gradient(_outputs, _inputs, grad_outputs=None, retain_graph=None,
                create_graph=False):
        if torch.is_tensor(_inputs):
            _inputs = [_inputs]
        else:
            _inputs = list(_inputs)
        grads = torch.autograd.grad(_outputs, _inputs, grad_outputs,
                                    allow_unused=False,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)
        grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads,
                                                                             _inputs)]
        return torch.cat([x.contiguous().view(-1) for x in grads])

def _hessian(outputs, inputs, out=None, allow_unused=False,
                 create_graph=False):
        #assert outputs.data.ndimension() == 1

        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)

        n = sum(p.numel() for p in inputs)
        if out is None:
            out = Variable(torch.zeros(n, n)).type_as(outputs)
        ai = 0
        grad_data = []
        for i, inp in enumerate(inputs):
            [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                         allow_unused=allow_unused)
            grad = grad.contiguous().view(-1)
            grad_data.append(grad.data)
            #grad = outputs[i].contiguous().view(-1)

            for j in range(inp.numel()):
                # print('(i, j): ', i, j, grad[j].requires_grad)
                if grad[j].requires_grad:
                    row = _gradient(grad[j], inputs[i:], retain_graph=True)[j:]
                else:
                    n = sum(x.numel() for x in inputs[i:]) - j
                    row = Variable(torch.zeros(n)).type_as(grad[j])
                    #row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

                out.data[ai, ai:].add_(row.clone().type_as(out).data)  # ai's row
                if ai + 1 < n:
                    out.data[ai + 1:, ai].add_(row.clone().type_as(out).data[1:])  # ai's column
                del row
                ai += 1
            del grad
        return grad_data[0], out

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
    model.eval()

    for ii, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        data.requires_grad_()

        loss = F.cross_entropy(model(data), target)
        # grad, hessian = _hessian(loss, data)
        ##
        print(loss)
        # [grad] = torch.autograd.grad(loss, data, create_graph=True,
        #                              allow_unused=False)
        # grad = grad.contiguous().view(-1)
        # n = data.numel()
        # hessian = torch.zeros(n, n).cuda()
        # for i in range(n):
        #     # print(i)
        #     [he] = torch.autograd.grad(grad[i], data, torch.ones_like(grad[i]),
        #                                 allow_unused=False,
        #                                 retain_graph=True)
        #     hessian[i, :] = he.view(-1)
        # grad = grad.data
        ##

        # print(grad.shape, hessian.shape)
        # grad_h = torch.matmul(grad[None, :], hessian.inverse())[0]
        # grad_h = torch.matmul(torch.from_numpy(np.linalg.inv(hessian.data.cpu().numpy())).cuda(), grad)
        # e, v = torch.symeig(hessian, eigenvectors=True)
        # print(e.shape, v.shape)
        # e_idx = e.max(0)[1]
        # print(e_idx)
        # grad_h = v[e_idx]

        [grad] = torch.autograd.grad(loss, data, create_graph=False,
                                      allow_unused=False)
        grad = grad.data.sign()
        while True:
            grad_h = torch.randn_like(grad).sign()
            inner_prod = (grad_h/((grad_h**2).sum().sqrt()) * grad/((grad**2).sum().sqrt())).sum()
            print(inner_prod)
            if inner_prod.abs() < 1e-8:
                break

        # print(torch.matmul((grad/grad.norm())[None, :], grad/grad.norm()))
        # print(torch.matmul((grad/grad.norm())[None, :], grad_h/grad_h.norm()))

        grad = grad/grad.abs().max()*args.epsilon*10
        grad_h = grad_h/grad_h.abs().max()*args.epsilon*10

        axis_x = torch.from_numpy(np.arange(-1., 1., 0.01)).cuda()
        axis_y = torch.from_numpy(np.arange(-1., 1., 0.01)).cuda()
        num = axis_x.shape[0]

        data = data.unsqueeze(0)
        print(data.shape)
        datas = data + grad.view_as(data) * axis_x[None,:, None, None, None] + grad_h.view_as(data) * axis_y[:,None, None, None, None]
        datas = datas.float()

        losses = []
        for j in range(num):
            # print(j)
            with torch.no_grad():
                losses.append(F.cross_entropy(model(datas[j]), target.repeat(num), reduction='none'))
                # print(losses[-1].data.cpu().numpy())
        losses = torch.stack(losses)

        print(losses.shape)
        scipy.io.savemat('mats/{}_losses_{}.mat'.format(args.load_clf, ii), dict(losses=losses.data.cpu().numpy()))

        # e, v = torch.symeig(hessian, eigenvectors=True)
        # print(e)
        # max_es[0].append(e.max().item())




if __name__ == '__main__':
    main()
