import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from collections import OrderedDict


# including different training method for active learning process (train acc=1, val loss, val acc, epoch)
class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    # epoch as a stopping criterion
    def supervised_train_epoch(self, data):
        n_epoch = 1
        self.clf = self.net().to(self.device) # training from scratch
        self.clf.train()
        if len(data)<=300:
            n_epoch = 20
        else:
            n_epoch = 30
        optimizer = optim.Adam(self.clf.parameters(), lr=0.0001, weight_decay=0.0005)
        # lr = 0.0001
        # lr = 0.000001 * len(data)# adaptive learning rate
        # if(lr > 0.0003):
        #     lr = self.params['optimizer_args']['lr']
        # optimizer = optim.Adam(self.clf.parameters(),lr)#,weight_decay=1e-5

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    # def init_train(self, data):
    #     n_epoch = self.params['n_epoch']
    #     best = {'epoch': 1, 'loss': 10}
    #     self.clf = self.net().to(self.device)
    #     self.clf.train()
    #     optimizer = optim.Adam(self.clf.parameters(), lr=0.005)
    #     criterion = nn.MSELoss()
    #     loader = DataLoader(data, batch_size=16, shuffle=True)
    #     for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
    #         for batch_idx, (x, y, idxs) in enumerate(loader):
    #             self.clf.train()
    #             x = x.view(x.size(0), -1)
    #             x, y = x.to(self.device), y.to(self.device)
    #             optimizer.zero_grad()
    #             out, e1 = self.clf(x)
    #             loss = criterion(out, x)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss  # 一个epoch的loss
    #         trigger += 1
    #         if train_loss / (batch_idx + 1) < best['loss']:
    #             trigger = 0
    #             best['epoch'] = epoch
    #             best['loss'] = train_loss / (batch_idx + 1)
    #             torch.save(self.clf, './autoencoder.pth')
    #         train_loss = 0
    #         if trigger >= args.early_stop:
    #             break

    # Accuracy saturation as a criterion for stopping
    def supervised_train_acc(self, data):
        best = {'epoch': 1, 'acc': 0.5}
        n_epoch = 300
        train_acc = 0
        trigger = 0
        stop = 100
        once = False
        self.clf = self.net().to(self.device)
        self.clf.train()
        # optimizer = optim.Adam(self.clf.parameters(),**self.params['optimizer_args'])#,weight_decay=1e-5
        optimizer = optim.Adam(self.clf.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=0.1, weight_decay=0.01)
        # optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                pred = out.max(1)[1]
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                train_acc += 1.0 * (y == pred).sum().item() / len(x)
            train_acc = train_acc / (batch_idx + 1)
            #             trigger+=1
            if train_acc == 1:
                break
            train_acc = 0

    # Validation loss as a stopping criterion
    def supervised_val_loss(self, data, val_data):
        n_epoch = 150
        trigger = 0
        best = {'epoch': 1, 'loss': 10}
        validation_loss = 0
        self.clf = self.net().to(self.device)
        self.clf.train()
        # if rd==0:
        #     self.clf = self.clf
        # else:
        #     self.clf = torch.load('./model.pth')
        # optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        #         optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        #         optimizer = optim.SGD(self.clf.parameters(), momentum=0.9, lr=0.0003, weight_decay=0.01, nesterov=True)
        optimizer = optim.Adam(self.clf.parameters(), lr=0.0002, eps=0.1, weight_decay=0.01)
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        val_loader = DataLoader(val_data, shuffle=False, **self.params['val_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, (valinputs, valtargets, idxs) in enumerate(val_loader):
                    valinputs, valtargets = valinputs.to(device), valtargets.to(device)
                    valoutputs = self.clf(valinputs)
                    validation_loss += F.cross_entropy(valoutputs, valtargets.long())
            trigger += 1
            # early stopping condition: if the acc not getting larger for over 10 epochs, stop
            if validation_loss / (valbatch_idx + 1) < best['loss']:
                trigger = 0
                best['epoch'] = epoch
                best['loss'] = validation_loss / (valbatch_idx + 1)
                torch.save(self.clf, './model.pth')
            validation_loss = 0
            if trigger >= 10:
                break

    # Validation acc as a stopping criterion
    def supervised_val_acc(self, data, val_data):
        n_epoch = 100
        trigger = 0
        best = {'epoch': 1, 'acc': 0.5}
        val_acc = 0
        self.clf = self.net().to(self.device)
        # if rd==0:
        #     self.clf = self.clf
        # else:
        #     self.clf = torch.load('./model.pth')
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        val_loader = DataLoader(val_data, shuffle=False, **self.params['val_args'])
        #         lr = 0.000001 * len(data)#变化的lr
        #         if(lr > self.params['optimizer_args']['lr']):
        #             lr = self.params['optimizer_args']['lr']
        #         optimizer = optim.Adam(self.clf.parameters(),lr)#,weight_decay=1e-5

        # optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        # optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        optimizer = optim.SGD(self.clf.parameters(), momentum=0.9, lr=0.0003, weight_decay=0.01, nesterov=True)
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                self.clf.train()
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, (valinputs, valtargets, idxs) in enumerate(val_loader):
                    valinputs, valtargets = valinputs.to(device), valtargets.to(device)
                    valoutputs = self.clf(valinputs)
                    #                         validation_loss+=criterion(valoutputs,valtargets.long())
                    pred = valoutputs.max(1)[1]
                    val_acc += 1.0 * (valtargets == pred).sum().item() / len(valinputs)
                # print("epoch: ",epoch,"val_acc: ",val_acc/(valbatch_idx+1))
            trigger += 1
            # early stopping condition: if the acc not getting larger for over 10 epochs, stop
            if val_acc / (valbatch_idx + 1) > best['acc']:
                trigger = 0
                best['epoch'] = epoch
                best['acc'] = val_acc / (valbatch_idx + 1)
                torch.save(self.clf, './model.pth')
            val_acc = 0
            if trigger >= 10:
                break
        print("best performance at Epoch :{}, acc :{}".format(best['epoch'], best['acc']))

    # used for getting prediction results for data
    def predict(self, data):
        #         self.clf = torch.load('./model.pth')
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    # def get_underfit_idx(self, data):  # no use for underfit_idx if train_acc=1
    #     unfit_idxs = []
    #     self.clf.eval()
    #     preds = torch.zeros(len(data), dtype=data.Y.dtype)
    #     loader = DataLoader(data, shuffle=False, **self.params['test_args'])
    #     with torch.no_grad():
    #         for x, y, idxs in loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             out, e1 = self.clf(x)
    #             pred = out.max(1)[1]
    #             preds[idxs] = pred.cpu()
    #             if preds[idxs] != y:
    #                 unfit_idxs.append(idxs)
    #     return unfit_idxs

    # Calculating probability for prediction, used as uncertainty
    def predict_prob(self, data):
        #         self.clf = torch.load('./model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 4])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    # Calculating 10 times probability for prediction, the mean used as uncertainty
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    # Used for Bayesian sampling
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    # def get_autofeature(self, data):
    #     self.clf = torch.load('./autoencoder.pth')
    #     self.clf.eval()
    #     embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
    #     loader = DataLoader(data, shuffle=False, **self.params['test_args'])
    #     with torch.no_grad():
    #         for x, y, idxs in loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             out, e1 = self.clf(x)
    #             embeddings[idxs] = e1.cpu()
    #     return embeddings


# different training network
class Dense_Net(nn.Module):
    def __init__(self):
        super(Dense_Net, self).__init__()
        # get layers of baseline model, loaded with some pre-trained weights on ImageNet
        self.feature_extractor = torchvision.models.densenet201(pretrained=True)
        self.feature_extractor.classifier = nn.Sequential()
        self.fc1 = nn.Linear(1920, 128, bias=True)
        self.fc2 = nn.Linear(128, 2, bias=True)
        # self.fc3 = nn.Linear(50, 2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 1920)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x

class Res_Net(nn.Module):
    def __init__(self):
        super(Res_Net, self).__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        self.fc1 = nn.Linear(512, 2, bias=True)
        # self.fc2 = nn.Linear(128, 2, bias=True)
        # self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        e1 = self.feature_extractor(x)
        x = self.fc1(e1)
        return x


class Inception_V3(nn.Module):
    def __init__(self):
        super(Inception_V3, self).__init__()
        model = torchvision.models.inception_v3(pretrained=True, transform_input=True, aux_logits=False)

        # define our model
        self.inception_layers = nn.Sequential(
            OrderedDict(list(model.named_children())[:-1]))
        self.fc1 = nn.Linear(2048, 4, bias=True)

        for m in self.fc1.parameters():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight(), gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.inception_layers(x)
        x = x.mean((2, 3))
        x = self.fc1(x)
        return x