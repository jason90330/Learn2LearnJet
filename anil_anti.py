# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
# from _typeshed import NoneType
import os
import random
import argparse
import wandb
import gc

import numpy as np
import torch
from torch import nn
import torchvision as tv

import learn2learn as l2l
from loss.focal import FocalLoss
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from efficientnet_pytorch import EfficientNet
from dataset.customDataAnilAnti import customData
from torchvision import transforms
from misc.test import Test, TestOri

from statistics import mean
from copy import deepcopy


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def log(trainErr, trainAcc, valErr, valAcc, testErr, testAcc):
    wandb.log({"Meta Train Error": trainErr,
               "Meta Train Accuracy": trainAcc,
               "Meta Valid Error": valErr,
               "Meta Valid Accuracy": valAcc,
               "Meta Test Error": testErr,
               "Meta Test Accuracy": testAcc})

def fast_adapt(batch,
               learner,
               features,
               loss,
               adaptation_steps,#5
               shots,#5
               ways,#2
               device=None):

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data = features.extract_features(data)

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        out = learner(adaptation_data)
        train_error = loss(out, adaptation_labels)
        learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(args
        # ways=2,
        # shots=5,
        # meta_lr=0.001,
        # fast_lr=0.1,
        # adapt_steps=5,
        # meta_bsz=32,
        # iters=1000,
        # cuda=1,
        # seed=42,
        # data_path='../../sinica/CelebA_Data/trainSquareCropped',
        # txt_path='../../sinica/CelebA_Data/metas/intra_test/train_label.txt',
):

    # wandb.login()
    # project_name = args.results_path.split("/")[1]
    # wandb.init(project=project_name, config = args)

    cuda = bool(args.cuda)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')

    # Create Datasets
    pre_processRGB = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    train_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="meta_train")
    valid_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="meta_val")
    
    test_dataset = customData(img_path=args.data_path,
                                    txt_path=args.txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="meta_test")
    
    test_set_dataset = customData(img_path=args.test_data_path,
                                txt_path=args.test_txt_path,
                                    data_transforms=pre_processRGB,
                                    phase="test")

    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        FusedNWaysKShots(train_dataset, n=args.ways, k=2*args.shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=args.ways, k=2*args.shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)
    
    test_transforms = [
        FusedNWaysKShots(test_dataset, n=args.ways, k=2*args.shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)

    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        )
    # Create model

    # features = l2l.vision.models.ConvBase(output_size=64, channels=3, max_pool=True)
    # features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, 256)))
    features = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
    # features = EfficientNet.from_name("efficientnet-b0")
    features.to(device)

    out_feature = features._fc.in_features
    # head = torch.nn.Linear(256, ways)
    head = torch.nn.Linear(out_feature, args.ways)
    head = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1), nn.Dropout(0.2), head)
    head = l2l.algorithms.MAML(head, lr=args.fast_lr)
    head.to(device)

    # Setup optimization
    all_parameters = list(features.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params=all_parameters, lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=0.01)
    loss = nn.CrossEntropyLoss(reduction='mean')
    # loss = FocalLoss(outNum = 2)

    # log the weight of model
    # wandb.watch(features, log_freq=100)
    # wandb.watch(head, log_freq=100)

    for iteration in range(args.iters):#20000
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(args.meta_bsz):#32
            # Compute meta-training loss
            learner = head.clone()
            batch = train_tasks.sample()#20
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = head.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = head.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
            

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / args.meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / args.meta_bsz)
        print('Meta Valid Error', meta_valid_error / args.meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / args.meta_bsz)
        print('Meta Test Error', meta_test_error / args.meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / args.meta_bsz)
        log(meta_train_error / args.meta_bsz,
            meta_train_accuracy / args.meta_bsz,
            meta_valid_error / args.meta_bsz,
            meta_valid_accuracy / args.meta_bsz,
            meta_test_error / args.meta_bsz,
            meta_test_accuracy / args.meta_bsz)

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            if p.grad != None:
                p.grad.data.mul_(1.0 / args.meta_bsz)
        optimizer.step()

        if iteration<100 or iteration%1000==0:
            TestOri(args, features, learner, test_data_loader, iteration)

        if iteration%1000==0:
            torch.save(features.state_dict(), args.results_path+"Feature-"+str(iteration)+".pt")
            torch.save(learner.state_dict(), args.results_path+"Learner-"+str(iteration)+".pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--ways', type=int, default=2, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=5, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('--iters', type=int, default=20000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--adapt_steps', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')   
    parser.add_argument('--meta_bsz', type=int, default=32, metavar='N',
                        help='number of ways (default: 5)')                                                

    parser.add_argument('--meta_lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--fast_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='number of ways (default: 5)')                                               
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')   
    
    # Path
    parser.add_argument('--data_path', default='../../sinica/CelebA_Data/trainSquareCropped', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument("--txt_path", default='../../sinica/CelebA_Data/metas/intra_test/train_label.txt', type=str)
    parser.add_argument('--results_path', type=str, default="output/anil_anti_focal/", metavar='S',
                        help='output dir')
    parser.add_argument('--wandb_name', type=str, default="anil_anti_record_all", metavar='S',
                        help='wandb name')
    # Test
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')   
    parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                    help='number of ways (default: 5)')        
    parser.add_argument('--test_data_path', default='../../sinica/CelebA_Data/testSquareCropped', type=str)
    parser.add_argument("--test_txt_path", default='../../sinica/CelebA_Data/metas/intra_test/test_label.txt', type=str)         
    parser.add_argument('--num_workers', default=10, type=int)   
    parser.add_argument('--dataset1', type=str, default='CelebA')
    parser.add_argument('--tstdataset', type=str, default='CelebA') 
    parser.add_argument('--tst_txt_name', type=str, default='norm_testScore.txt')               

    args = parser.parse_args()

    wandb.login()
    wandb.init(project=args.wandb_name, config = args)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)  

    main(args)