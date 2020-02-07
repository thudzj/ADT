from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
import numpy as np
from collections import OrderedDict
from fs_wideresnet import WideResNet as fs_WideResNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--st', type=int, default=0, metavar='N', help='')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    testset = torchvision.datasets.SVHN(root='../../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

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
        for i, inp in enumerate(inputs):
            [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                         allow_unused=allow_unused)
            grad = grad.contiguous().view(-1)
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
        return out

def main():

    model = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    model.load_state_dict(torch.load('./checkpoint_eval/pgd7.pt'))

    model1 = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    model1.load_state_dict(torch.load('./checkpoint_eval/gaussian-entropy-0.01-bs64.pt'))

    model2 = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    model2.load_state_dict(torch.load('./checkpoint_eval/implicit.pt'))


    model4 = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    model4.load_state_dict(torch.load('./checkpoint_eval/standard.pt'))

    model5 = WideResNet(28,num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    model5.load_state_dict(torch.load('./checkpoint_eval/dist-new.pt'))

    model.eval()
    model1.eval()
    model2.eval()
    # model3.eval()
    model4.eval()
    model5.eval()

    data, target = [], []
    for ite in range(args.st*100, (args.st+1)*100):
        img, targ = test_loader.dataset[ite]
        data.append(img)
        target.append(targ)
    data = torch.stack(data)
    target = torch.from_numpy(np.array(target)).long()
    print(data.shape, target.shape)
    # data, target = test_loader.dataset.images[args.st*100:(args.st+1)*100], #next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    if False:
        max_es = [[], [], [], []]
        for ii in range(args.test_batch_size):
            y0 = target[ii:(ii+1)]

            x0 = Variable(data[ii:(ii+1)].data, requires_grad=True)
            loss = F.cross_entropy(model(x0), y0)
            hessian = _hessian(loss, x0)
            # print(hessian[:10, :10])
            e, v = torch.symeig(hessian, eigenvectors=True)
            # print(e)
            max_es[0].append(e.max().item())

            x0 = Variable(data[ii:(ii+1)].data, requires_grad=True)
            loss1 = F.cross_entropy(model1(x0), y0)
            hessian1 = _hessian(loss1, x0)
            # print(hessian1[:10, :10])
            e1, v1 = torch.symeig(hessian1, eigenvectors=True)
            # print(e1)
            max_es[1].append(e1.max().item())

            x0 = Variable(data[ii:(ii+1)].data, requires_grad=True)
            loss2 = F.cross_entropy(model2(x0), y0)
            hessian2 = _hessian(loss2, x0)
            # print(hessian2[:10, :10])
            e2, v2 = torch.symeig(hessian2, eigenvectors=True)
            # print(e1)
            max_es[2].append(e2.max().item())

            x0 = Variable(data[ii:(ii+1)].data, requires_grad=True)
            loss3 = F.cross_entropy(model3(x0), y0)
            hessian3 = _hessian(loss3, x0)
            # print(hessian2[:10, :10])
            e3, v3 = torch.symeig(hessian3, eigenvectors=True)
            # print(e1)
            max_es[3].append(e3.max().item())

            print("Val:")
            print(max_es[0][-1], max_es[1][-1], max_es[2][-1], max_es[3][-1])
            print("Mean:")
            print(np.mean(max_es[0]), np.mean(max_es[1]), np.mean(max_es[2]), np.mean(max_es[3]))
            print("Std:")
            print(np.std(max_es[0], ddof=1), np.std(max_es[1], ddof=1), np.std(max_es[2], ddof=1), np.std(max_es[3], ddof=1))
    else:
        models = [model5, model4, model, model1, model2]
        # data.requires_grad_()
        for num, model_ in enumerate(models):
            data = Variable(data.data, requires_grad=True)
            loss = F.cross_entropy(model_(data), target)
            print(loss)
            [grad] = torch.autograd.grad(loss, data, create_graph=True,
                                         allow_unused=False)
            grad = grad.contiguous().view(data.size(0), -1)
            n = data[0].numel()
            hessian = torch.zeros(data.size(0), n, n).cuda()
            tmp_grad = torch.ones(data.size(0)).cuda()
            for i in range(n):
                # print(i)
                [he] = torch.autograd.grad(grad[:, i], data, tmp_grad,
                                            allow_unused=False,
                                            retain_graph=True)
                hessian[:, i, :] = he.view(data.size(0), -1)

            print('cal eigen values')
            es = np.linalg.eigvalsh(hessian.data.cpu().numpy())
            es_max = np.max(es, 1)
            # es = []
            # with torch.no_grad():
            #     for j in range(data.size(0)):
            #         e, v = torch.symeig(hessian[j], eigenvectors=True)
            #         es.append(e.max().item())
            print(es_max , np.mean(es_max), np.std(es_max, ddof=1))
            del hessian, grad, loss

            with open(str(num)+'.txt', 'a+') as f:
                for item in es_max:
                    f.write(str(item)+'\n')

if __name__ == '__main__':
    main()
