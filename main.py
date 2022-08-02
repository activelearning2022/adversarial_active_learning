import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint

from data import Data
from utils import get_dataset, get_net, get_strategy
from config import parse_args

# args = parse_args.parse_known_args()[0]
args = parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load dataset
X_train, Y_train, X_val, Y_val, X_test, Y_test, handler = get_dataset(args.dataset_name)
dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

net = get_net(args.dataset_name, device) # load network
strategy = get_strategy(args.strategy_name)(dataset, net) # load strategy

# start experiment
dataset.initialize_labels_random(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

print("Round 0")
rd = 0
strategy.train(rd, args.training_name)
accuracy = []
size = []
preds = strategy.predict(dataset.get_test_data()) # get model prediction for test dataset
print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")  # get model performance for test dataset
accuracy.append(dataset.cal_test_acc(preds))
size.append(args.n_init_labeled)
testing_accuracy = 0

for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")
    # query
    if args.strategy_name == "AdaptiveAdversarial":
        query_idxs, generative_top_sample, real_end_sample, fake_label, pseudo_idxs = strategy.query(rd, args.n_query)
        label = dataset.get_label(query_idxs)
        adversarial_sample = []
        adversarial_label = []
        pseudo_adversarial_sample = []
        pseudo_label = []

        # query sample + adversarial sample
        # for one adversarial sample
        for i in range(len(query_idxs)):
            label = dataset.get_label(query_idxs[i])
            dataset.add_labeled_data(generative_top_sample[i], label)
        print("uncertainty data", len(query_idxs))

        # pseudo labeling
        for i in range(len(pseudo_idxs)):
            dataset.update_pseudo_label(pseudo_idxs[i], fake_label[i]) # pseudo labeling
            strategy.update(pseudo_idxs)  # update training dataset and unlabeled dataset for pseudo labeling
        print("pseudo data", len(pseudo_idxs))

    else:
        query_idxs = strategy.query(args.n_query)  # query_idxs为active learning请求标签的数据

    # update labels
    strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning
    strategy.train(rd, args.training_name)

    # efficient training
    # strategy.efficient_train(rd,dataset.get_train_data())

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())

    testing_accuracy = dataset.cal_test_acc(preds)
    print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")

    accuracy.append(testing_accuracy)
    labeled_idxs, _ = dataset.get_labeled_data()
    size.append(len(labeled_idxs))

    unlabeled_idxs, _ = dataset.get_unlabeled_data()
    if len(unlabeled_idxs) < 5:
        break

# save the result
dataframe = pd.DataFrame(
    {'model': 'Inception_V3', 'Method': args.strategy_name, 'Training dataset size': size, 'Accuracy': accuracy})
dataframe.to_csv("./Experiments/performance.csv", index=False, sep=',')
