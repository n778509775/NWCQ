import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
#from collections import Counter

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import scale, minmax_scale, Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from network import Discriminator

num_epochs = 10
batch_size = 128 # batch size for each cluster
base_lr = 1e-3
lr_step = 100  # step decay of learning rates
l2_decay = 5e-5

def read_csv_faster(filename):
	data_df = pd.read_csv(filename,index_col=1)
	dataset = {}
	dataset['labels'] = data_df.iloc[:,0].tolist()
	#dataset['board'] = data_df.iloc[:,1].tolist()
	dataset['mz_exp'] = np.transpose(np.array(data_df.iloc[:,1:]))
	dataset['feature'] = data_df.columns.values.tolist()[1:]
	return dataset

def plot_clas_loss(loss_classifier_list, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(loss_classifier_list)), loss_classifier_list, "b--",linewidth=1)
    ax.legend(['loss_classification'], loc="upper right")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/')
parser.add_argument('--train_file', type=str, default='3.csv')
config = parser.parse_args()

train_file = config.data_folder + config.train_file
#dataset = read_csv_faster('./data/1.csv')
dataset = read_csv_faster(train_file)
FinalData = dataset['mz_exp'].transpose()
AllLabel = dataset['labels']

num_inputs = FinalData.shape[1]
discriminator = Discriminator(num_inputs=num_inputs)

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

scoreA=[]
criterion = torch.nn.CrossEntropyLoss()
models=[]
zzz=np.arange(len(AllLabel)*2).reshape((len(AllLabel), 2))
X=FinalData
y=np.array(AllLabel)
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=random.randint(0,99))
for train_index,test_index in skf.split(zzz,y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	torch_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
	data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	loss_classifier_list = []
	for epoch in range(1, num_epochs + 1):
		learning_rate = base_lr * math.pow(0.9, epoch / lr_step)
		gamma_rate = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1
		optimizer = torch.optim.Adam([{'params': discriminator.parameters()},], lr=learning_rate, weight_decay=l2_decay)

		discriminator.train()
		iter_data = iter(data_loader)
		num_iter = len(data_loader)
		#print(num_iter)
		total_clas_loss = 0
		num_batches = 0
		for it in range(0, num_iter):
			data, label = iter_data.next()
			if it % len(data_loader) == 0:
				iter_data = iter(data_loader)
			data = Variable(torch.FloatTensor(data))
			label = Variable(torch.LongTensor(label))
			Disc_a = discriminator(data)

			optimizer.zero_grad()
			loss_classification = torch.FloatTensor([0])
			for cls in range(len(label)):
				loss_classification += F.binary_cross_entropy(torch.squeeze(Disc_a)[cls], label[cls].float())
			#loss_classification = criterion(Disc_a, label)
			loss = loss_classification
			loss.backward()
			optimizer.step()

			num_batches += 1
			total_clas_loss += loss_classification.data.item()
		avg_clas_loss = total_clas_loss / num_batches
		loss_classifier_list.append(avg_clas_loss)
	plot_clas_loss(loss_classifier_list, 'clas_loss.png')
	discriminator.eval()
	models.append(discriminator.state_dict())

	Disc_b = discriminator(torch.from_numpy(X_test).float())
	pred_b = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in Disc_b]))
	#pred_b = torch.max(F.softmax(Disc_b), 1)[1]
	test_label = torch.from_numpy(y_test)
	num_correct_b = 0
	num_correct_b += torch.eq(pred_b, test_label).sum().float().item()
	Acc_b = num_correct_b/len(test_label)
	scoreA.append(Acc_b)

print(np.mean(scoreA))

