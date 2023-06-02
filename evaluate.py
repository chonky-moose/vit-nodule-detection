#%%
'''
Change the dataloader to use CXR_dataset_DR_tech
and see if that leads to accurate results
'''

#%%
# folder containing images for evaluation
folder = '/mnt/d/data/snu_snub_test_normal_nodule_only'
# path to model checkpoint
# checkpoint = './models/glowing-dew-31.pt'
checkpoint = './checkpoint_epoch0013.pt'

# mean, SD for normalization of images
mean, SD = 0.541584312915802, 0.24306267499923706

# diagnoses = {'normal':0, 'nodule':1, 'pneumonia':2, 'pneumothorax':3}

#%%
from train import get_arguments, load_model, make_dataloader
import utils

import numpy as np
from tqdm.notebook import tqdm
from pprint import pprint

import sklearn.metrics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T



# %% Prepare dataloader
# transform = T.Compose([
#     T.Resize((256, 256)),
#     utils.GaussianBlurInference(),
#     T.ToTensor(),
#     T.Normalize(mean, SD),
# ])
# testset = ImageFolder(folder, transform=transform)
# test_sampler = SequentialSampler(testset)
# test_loader = DataLoader(testset, num_workers=0, pin_memory=True)
# print(testset.class_to_idx)


# import CXR_dataset_DR_tech
# testset = CXR_dataset_DR_tech.CXR_Dataset(folder,
#                                           size=1.0,
#                                           transforms=None,
#                                           mode='test')
# test_sampler = SequentialSampler(testset)
# test_loader = DataLoader(testset,
#                          sampler=test_sampler,
#                          batch_size=1,
#                          num_workers=0,
#                          pin_memory=True)
# print(len(testset))

# %% Prepare model
parser = get_arguments()
args = parser.parse_args("")
args.checkpoint = checkpoint
model, _, _ = load_model(args)
test_loader = make_dataloader(args, folder, 1)

#%%
preds, trues = [], []
device = torch.device('cuda')
for i, (x, y) in enumerate(tqdm(test_loader)):
    
    x = x.to(device)    
    with torch.no_grad():
        output = model(x)
        output = output.detach().cpu().numpy()
        pred = output.argmax()
        
        
        preds.append(pred)
        trues.append(y.item())
#%%
trues, preds = np.vstack(trues), np.vstack(preds)
roc_auc = sklearn.metrics.roc_auc_score(trues, preds, average=None)
auc_pr = sklearn.metrics.average_precision_score(trues, preds, average=None)
f1_score = sklearn.metrics.f1_score(trues, np.round(preds), average=None)

print(f"Results for weights at {checkpoint}")
print(f"Using test dataset at {folder}")
print("ROC AUC: ", roc_auc)
print("AUC PR: ", auc_pr)
print("F1: ", f1_score)
# %%
sum(preds)