from data import get_Messidor, get_Breast, get_Breast_multi
from handlers import Messidor_Handler, Breast_Handler, Breast_multi_Handler
from nets import Net, Dense_Net, Res_Net, Inception_V3
from query_strategies import RandomSampling, EntropySampling, EntropySamplingDropout, BALDDropout, AdversarialDeepFool

# important settings
params = {
    'Messidor':
        {'n_epoch': 100,
         'train_args': {'batch_size': 8, 'num_workers': 4},
         'val_args': {'batch_size': 8, 'num_workers': 4},
         'test_args': {'b §§atch_size': 8, 'num_workers': 4},
         'optimizer_args': {'lr': 0.0002, 'betas': (0.9, 0.999), 'eps': 0.1, 'weight_decay': 0.01}},  # inception paper

    'Breast':
        {'n_epoch': 100,
         'train_args': {'batch_size': 8, 'num_workers': 4},
         'val_args': {'batch_size': 8, 'num_workers': 4},
         'test_args': {'batch_size': 8, 'num_workers': 4},
         'optimizer_args': {'lr': 0.0002, 'betas': (0.9, 0.999), 'eps': 0.1, 'weight_decay': 0.01}},  # inception paper

    'Breast_multi':
        {'n_epoch': 100,
         'train_args': {'batch_size': 8, 'num_workers': 4},
         'val_args': {'batch_size': 8, 'num_workers': 4},
         'test_args': {'batch_size': 8, 'num_workers': 4},
         'optimizer_args': {'lr': 0.0002, 'betas': (0.9, 0.999), 'eps': 0.1, 'weight_decay': 0.01}},  # inception paper
}


# Get data loader
def get_handler(name):
    if name == 'Messidor':
        return Messidor_Handler
    elif name == 'Breast':
        return Breast_Handler
    elif name == 'Breast_multi':
        return Breast_multi_Handler


# Get dataset
def get_dataset(name):
    if name == 'Messidor':
        return get_Messidor(get_handler(name))
    if name == 'Breast':
        return get_Breast(get_handler(name))
    elif name == 'Breast_multi':
        return get_Breast_multi(get_handler(name))
    else:
        raise NotImplementedError


# define network for specific dataset
def get_net(name, device, init=False):
    if name == 'Messidor':
        # return Net(Res_Net, params[name], device)
        # if init==False:
        return Net(Inception_V3, params[name], device)
        # return Net(Dense_Net, params[name], device)
    #         return Net(Res_Net, params[name], device)
    # else:
    #     return Net(Autoencoder, params[name], device)
    elif name == 'Breast':
        # return Net(Res_Net, params[name], device)
        # if init==False:
        return Net(Inception_V3, params[name], device)

    elif name == 'Breast_multi':
        return Net(Inception_V3, params[name], device)
    #         return Net(Res_Net, params[name], device)

    else:
        raise NotImplementedError

# get strategies
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError