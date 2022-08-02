import numpy as np
import torch
import glob
import os.path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2

class Data:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def supervised_training_labels(self):
        # used for supervised learning baseline, put all data labeled
        tmp_idxs = np.arange(self.n_pool)
        self.labeled_idxs[tmp_idxs[:]] = True

    def initialize_labels_random(self, num):
        # generate initial labeled pool
        # use idx to distinguish labeled and unlabeled data
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    # def initialize_labels(self, dataset, num): #最开始initialize过程 随机选
    #     # generate initial labeled pool 根据label index来确认 labeled_idxs就是label data反之是unlabel data（x_train+y_train)
    #     tmp_idxs = np.arange(self.n_pool)
    #     np.random.shuffle(tmp_idxs)
    #     net_init = get_net(args.dataset_name, device, init=True)
    #     strategy_init = get_strategy("KMeansSampling")(dataset, net_init, init=True)
    #     rd = 0
    #     strategy_init.train(rd, init=True)
    #     query_idxs = strategy_init.query(args.n_init_labeled)
    #     strategy.update(query_idxs)

    def get_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        print("labeled data", labeled_idxs.shape)
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        # get unlabeled data for active learning selection process
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_val_data(self):
        # get validation dataset if exist
        return self.handler(self.X_val, self.Y_val)

    def get_test_data(self):
        # get test dataset if exist
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        # calculate accuracy for test dataset
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

    def cal_train_acc(self, preds):
        # calculate accuracy for train dataset for early stopping
        return 1.0 * (self.Y_train == preds).sum().item() / self.n_pool

    def add_labeled_data(self, data, label):
        # used for generated adversarial image expansion. Adding generated adversarial image with label to training dataset
        data = torch.reshape(data, (512, 512, 3))
        data = torch.unsqueeze(data, 0)
        self.X_train = torch.tensor(self.X_train)
        self.Y_train = torch.tensor(self.Y_train)
        self.X_train = torch.cat((self.X_train, data), 0)
        self.Y_train = torch.cat((self.Y_train, torch.tensor([label])), 0)
        self.labeled_idxs = np.append(self.labeled_idxs, True)
        self.n_pool += 1

    def update_pseudo_label(self, idx, label):
        # used for pseudo labeling, change the correct label to pseudo label
        self.X_train = torch.tensor(self.X_train)
        self.Y_train[idx] = label

    #     def (self,data,label):
    #         data = tadd_labeled_dataorch.reshape(data, (-1,512,512,3))
    # #         data = torch.unsqueeze(data, 0)
    #         self.X_train = torch.tensor(self.X_train)
    #         self.Y_train = torch.tensor(self.Y_train)
    #         self.X_train = torch.cat((self.X_train, data), 0)
    #         self.Y_train = torch.cat((self.Y_train, label), 0)
    #         for i in range(len(data)):
    #             self.labeled_idxs=np.append(self.labeled_idxs,True)
    #             self.n_pool+=1

    def get_label(self, idx):
        # Get the real label (share lable) for adversarial samples
        self.Y_train = np.array(self.Y_train)
        label = torch.tensor(self.Y_train[idx])
        return label

    # # efficient training method
    # def get_efficient_training_data(self, idx, new_idx):
    #     all_idx = np.concatenate((idx, new_idx), axis=None)
    #     labeled_idxs = np.arange(self.n_pool)[all_idx]
    #     return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])


# Read dataset
def crop(image):
    #used for Messidor dataset preprocessing: crop all images into square shape
    sums = image.sum(axis=0)
    sums = sums.sum(axis=1)
    filter_arr = []
    for s in sums:
        if s == 0:
            filter_arr.append(False)
        else:
            filter_arr.append(True)
    image = image[:, filter_arr]
    h = image.shape[0]
    w = image.shape[1]
    if h < w:
        x = (w - h) // 2
        image = image[:, x:x + h, :]
    elif h > w:
        x = (h - w) // 2
        image = image[x:x + w, :, :]
    else:
        pass
    return image


def create_dataset(image_category, label):
    dataset = []
    for image_path in tqdm(image_category):
        image_size = 512
        # image = PIL.Image.open(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = crop(image)
        image = np.array(image)
        image = cv2.resize(image, (image_size, image_size))  # 299,299,3
        # image = transform(image)
        dataset.append([np.array(image), np.array(label)])
    random.shuffle(dataset)
    return dataset

#methods for splitting dataset
def split_dataset(x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2022)
    return x_train, torch.LongTensor(y_train), x_val, torch.LongTensor(y_val)


def split_train_val(x_train, y_train):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2022)
    return x_train, torch.LongTensor(y_train), x_test, torch.LongTensor(y_test)

def get_Messidor(handler):
    input_path = "../active_learning/data/Messidor_Dataset/Annotation"
    normal = []
    diseased = []
    # glob over all excels
    for i in glob.glob(os.path.join(input_path, "*.xls")):
        df_data = pd.read_excel(i)
        # Get the path of normal and non-normal data
        normal_path = df_data.loc[(df_data["Retinopathy grade"] == 0)]["Image name"].values
        for j in normal_path:
            normal.append("../active_learning/data/Messidor_Dataset/" + i[-10:-4] + "/" + j)
        diseased_path = df_data.loc[(df_data["Retinopathy grade"] != 0)]["Image name"].values
        for k in diseased_path:
            diseased.append("../active_learning/data/Messidor_Dataset/" + i[-10:-4] + "/" + k)

    dataset = create_dataset(normal,1)
    dataset = create_dataset(diseased,0)

    x = np.array([i[0] for i in dataset])
    y = np.array([i[1] for i in dataset])

    # torch.save(torch.Tensor(x),'./data/Messidor_Crop_x.pt')
    # torch.save(torch.Tensor(y),'./data/Messidor_Crop_y.pt')

    # x = torch.load('./data/Messidor_Crop_x.pt')
    # y = torch.load('./data/Messidor_Crop_y.pt')
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    y = y.long()

    x_train, y_train, x_test, y_test = split_dataset(x, y)
    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)
    print("x_train", np.array(x_train).shape)
    print("x_val", np.array(x_val).shape)
    print("x_test", np.array(x_test).shape)
    return x_train, y_train, x_val, y_val, x_test, y_test, handler


def walk_root_dir(DIR):
    wholePathes = []
    for dirpath, subdirs, files in os.walk(DIR):
        for x in files:
            if x.endswith(('.png')):
                wholePathes.append(os.path.join(dirpath, x))
    return wholePathes


def Dataset_loader(input_path, RESIZE):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    wholePathes = walk_root_dir(input_path)
    for PATH in tqdm(wholePathes):
        img = read(PATH)
        img = cv2.resize(img, (RESIZE, RESIZE))
        IMG.append(np.array(img))
    return IMG

def get_Breast(handler):
    # method to read BreakHis
    benign = np.array(Dataset_loader('../active_learning/breast/benign/SOB',512))
    malign = np.array(Dataset_loader('../active_learning/breast/malignant/SOB',512))
    # Create labels
    benign_label = np.zeros(len(benign))
    malign_label = np.ones(len(malign))

    # Merge data
    x = np.concatenate((benign, malign), axis = 0)
    y = np.concatenate((benign_label, malign_label), axis = 0)

    #     torch.save(torch.Tensor(x),'./breast/histology_slides/breast/x.pt')
    #     torch.save(torch.Tensor(y),'./breast/histology_slides/breast/y.pt')

    # x=torch.load('./breast/x.pt')
    # y=torch.load('./breast/y.pt')
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    y = y.long()

    x_train, y_train, x_test, y_test = split_dataset(x, y)
    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)
    print("x_train", np.array(x_train).shape)
    print("x_val", np.array(x_val).shape)
    print("x_test", np.array(x_test).shape)
    return x_train, y_train, x_val, y_val, x_test, y_test, handler


def get_Breast_multi(handler):
    # method to read breast cancer diagnosis
    benign = np.array(Dataset_loader('./Breast_Cancer_Diagnosis_Dataset/ICIAR2018_BACH_Challenge/Photos/Benign', 512))
    insitu = np.array(Dataset_loader('./Breast_Cancer_Diagnosis_Dataset/ICIAR2018_BACH_Challenge/Photos/InSitu', 512))
    invasive = np.array(Dataset_loader('./Breast_Cancer_Diagnosis_Dataset/ICIAR2018_BACH_Challenge/Photos/Invasive', 512))
    normal = np.array(Dataset_loader('./Breast_Cancer_Diagnosis_Dataset/ICIAR2018_BACH_Challenge/Photos/Normal', 512))
    # Create labels
    benign_label = np.zeros(len(benign))
    insitu_label = np.ones(len(insitu))
    invasive_label = np.full(len(invasive), 2)
    normal_label = np.full(len(normal), 3)

    # Merge data
    x = np.concatenate((benign, insitu, invasive, normal), axis=0)
    y = np.concatenate((benign_label, insitu_label, invasive_label, normal_label), axis=0)

    # torch.save(torch.Tensor(x),'./Breast_Cancer_Diagnosis_Dataset/x.pt')
    # torch.save(torch.Tensor(y),'./Breast_Cancer_Diagnosis_Dataset/y.pt')

    # x=torch.load('./Breast_Cancer_Diagnosis_Dataset/x.pt')
    # y=torch.load('./Breast_Cancer_Diagnosis_Dataset/y.pt')
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    y = y.long()

    x_train, y_train, x_test, y_test = split_dataset(x, y)
    # x_train,y_train,x_val,y_val = split_train_val(x_train,y_train)
    print("x_train", np.array(x_train).shape)
    # print("x_val",np.array(x_val).shape)
    print("x_test", np.array(x_test).shape)
    return x_train, y_train, x_test, y_test, handler
