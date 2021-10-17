#!/usr/bin/env python
import torch.utils.data
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tkinter import _flatten

from function import plot_clas_loss, pre_processing
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import network as models
import math
import argparse
import pylib

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# CUDA
device_id = 0 # ID of GPU to use
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

plt.ioff()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default='data/')
    parser.add_argument('-l', '--dataset_file_list', nargs='+', help='<Required> Set flag', required=True, type=str)
    parser.add_argument('--train_num', type=int, default=2)    
    parser.add_argument('--code_save', type=str, default='code_list.pkl')
    parser.add_argument('--take_log', type=bool, default=False)
    parser.add_argument('--standardization', type=bool, default=False)
    parser.add_argument('--scaling', type=bool, default=False)
    parser.add_argument('--plots_dir', type=str, default='plots/')

    parser.add_argument('--code_dim', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of total iterations for training')
    parser.add_argument('--lr_step', type=int, default=10000, help='step decay of learning rates')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='learning rate for network')
    parser.add_argument('--l2_decay', type=float, default=5e-5)
    parser.add_argument('--log_interval', type=int, default=100)

    config = parser.parse_args()
    #print(config)

    data_folder = config.data_folder
    code_save_file = data_folder + config.code_save
    dataset_file_list = [data_folder+f for f in config.dataset_file_list]
    data_num = len(dataset_file_list)  
    train_num = config.train_num
    plots_dir = config.plots_dir    

    # read data
    pre_process_paras = {'take_log': config.take_log, 'standardization': config.standardization, 'scaling': config.scaling}
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)

    # training
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    num_inputs = len(dataset_list[0]['feature'])
    code_dim = config.code_dim

    # construct a DataLoader for each batch
    batch_loader_dict = {}
    for i in range(len(dataset_list)):
        gene_exp = dataset_list[i]['mz_exp'].transpose()
        labels = dataset_list[i]['labels']  

        # construct DataLoader list
        if cuda:
            torch_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(gene_exp).cuda(), torch.LongTensor(labels).cuda())
        else:
            torch_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(gene_exp), torch.LongTensor(labels))
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                                        shuffle=True, drop_last=True)
        batch_loader_dict[i+1] = data_loader

    # create model
    discriminator = models.Discriminator(num_inputs=num_inputs)
    if cuda:
        discriminator.cuda()

    log_interval = config.log_interval
    base_lr = config.base_lr
    lr_step = config.lr_step
    num_epochs = config.num_epochs
    l2_decay = config.l2_decay

    # training
    criterion = nn.CrossEntropyLoss()
    loss_classifier_list = []
    for epoch in range(1, num_epochs + 1):

        # step decay of learning rate
        #learning_rate = base_lr / math.pow(2, math.floor(epoch / lr_step))
        learning_rate = base_lr * math.pow(0.9, epoch / lr_step)
        # regularization parameter between two losses
        gamma_rate = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1

        if epoch % log_interval == 0:
            print('{:}, Epoch {}, learning rate {:.3E}'.format(time.asctime(time.localtime()), epoch, learning_rate))
                
        optimizer = torch.optim.Adam([
            {'params': discriminator.parameters()},
        ], lr=learning_rate, weight_decay=l2_decay)

        discriminator.train()

        iter_data_dict = {}
        for cls in batch_loader_dict:
            iter_data = iter(batch_loader_dict[cls])
            iter_data_dict[cls] = iter_data
        # use the largest dataset to define an epoch
        num_iter = 0
        for cls in batch_loader_dict:
            num_iter = max(num_iter, len(batch_loader_dict[cls]))

        total_clas_loss = 0
        num_batches = 0

        for it in range(0, num_iter):
            data_dict = {}
            label_dict = {}
            code_dict = {}
            reconstruct_dict = {}
            Disc_dict = {}
            for cls in iter_data_dict:
                data, labels = iter_data_dict[cls].next()
                data_dict[cls] = data
                label_dict[cls] = labels
                if it % len(batch_loader_dict[cls]) == 0:
                    iter_data_dict[cls] = iter(batch_loader_dict[cls])
                data_dict[cls] = Variable(data_dict[cls])
                label_dict[cls] = Variable(label_dict[cls])
            
            for cls in range(1,train_num+1):   
                Disc_dict[cls] = discriminator(data_dict[cls])

            optimizer.zero_grad()

            #Loss
            # classifier loss for dignosis
            loss_classification = torch.FloatTensor([0])
            if cuda:
                loss_classification = loss_classification.cuda()
            for cat in range(1,train_num+1):    
                for cls in range(len(label_dict[cat])):
                    loss_classification += F.binary_cross_entropy(torch.squeeze(Disc_dict[cat])[cls], label_dict[cat][cls].float())
                #loss_classification = criterion(Disc_dict[cat], label_dict[cat])
            loss = loss_classification

            loss.backward()
            optimizer.step()

            # update total loss
            num_batches += 1
            total_clas_loss += loss_classification.data.item()

        avg_clas_loss = total_clas_loss / num_batches

        if epoch % log_interval == 0:
            print('Avg_classify_loss {:.3E}'.format(avg_clas_loss))            

        loss_classifier_list.append(avg_clas_loss)

    #scheduler.step()
    plot_clas_loss(loss_classifier_list, plots_dir+'clas_loss.png')

    # testing: extract codes
    discriminator.eval()

    #F_score
    def matric(cluster, labels):
        TP, TN, FP, FN = 0, 0, 0, 0
        n = len(labels)
        for i in range(n):
            if cluster[i]:
                if labels[i]:
                    TP += 1
                else:
                    FP += 1
            elif labels[i]:
                FN += 1
            else:
                TN += 1
        return TP, TN, FP, FN

    #Accuracy
    for pre in range(train_num,len(dataset_list)):
        test_data = torch.from_numpy(dataset_list[pre]['mz_exp'].transpose())
        test_label = torch.from_numpy((np.array(dataset_list[pre]['labels']))).cuda()
        Disc = discriminator(test_data.float().cuda())

        pred = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in Disc])).cuda()
        #pred = torch.max(F.softmax(Disc), 1)[1]
        num_correct = 0
        num_correct += torch.eq(pred, test_label).sum().float().item()
        Acc = num_correct/len(test_label)
        print("Accuracy is ", Acc)

        TP, TN, FP, FN = matric(pred, test_label)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_score = 2 * precision * recall / (precision + recall)
        print("F_score is ",f_score)

        #AUC
        print("AUC is ",roc_auc_score(test_label.cpu(), pred.cpu()))

        #MCC
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print("MCC is ",MCC)

