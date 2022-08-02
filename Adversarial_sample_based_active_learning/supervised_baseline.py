import cv2
from matplotlib import pyplot as plt
# fix random seed
setup_seed(2022)
#supervised learning baseline

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get dataset
X_train, Y_train, X_test, Y_test, handler = get_dataset(args.dataset_name)

# get dataloader
dataset = Data(X_train, Y_train, X_test, Y_test, handler)
# x_train,x_val,y_train,y_val = split_train_val(x_train,y_train)# load dataset
# dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)

# get network
net = get_net(args.dataset_name, device)

# start supervised learning baseline
dataset.supervised_training_labels()
labeled_idxs, labeled_data = dataset.get_labeled_data()
# val_data = dataset.get_val_data()
net.supervised_train_acc(labeled_data)
preds = net.predict(dataset.get_test_data())
print(f"testing accuracy: {dataset.cal_test_acc(preds)}")