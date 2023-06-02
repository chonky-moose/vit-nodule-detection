#%%
import vision_transformer as vits
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import os
import argparse
import sys
import math
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import sklearn.metrics
import wandb
#%%
class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
class ViT_Classifier(nn.Module):
    def __init__(self,
                 vit_arch,
                 vit_patch_size,
                 n_last_blocks,
                 avgpool_patchtokens,
                 vit_checkpoint_path,
                 num_labels,
                 linear_clf_checkpoint_path):
        super().__init__()
        device = torch.device('cuda')
        
        # Initialize ViT
        self.vit = vits.__dict__[vit_arch](patch_size=vit_patch_size,
                                           num_classes=0)
        if vit_checkpoint_path:
            self.vit.load_state_dict(torch.load(vit_checkpoint_path), strict=False)
        self.vit = self.vit.to(device)
        embed_dim = self.vit.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        
        # Initialize linear classifier
        self.linear_classifier = LinearClassifier(embed_dim, num_labels)
        if linear_clf_checkpoint_path:
            state_dict = torch.load(linear_clf_checkpoint_path)['state_dict']
            state_dict = {k.replace("module.", ""):v for k,v in state_dict.items()}
            self.linear_classifier.load_state_dict(state_dict, strict=True)
        self.linear_classifier = self.linear_classifier.to(device)
        
        self.n_last_blocks = n_last_blocks

    def forward(self, x):
        intermediate_output = self.vit.get_intermediate_layers(x, self.n_last_blocks)
        output = torch.cat([out[:,0] for out in intermediate_output], dim=-1)
        output = self.linear_classifier(output)
        return output
# %%
def get_arguments():
    parser = argparse.ArgumentParser('vit-nodule-detection',
                                     add_help=False)
    parser.add_argument('--train_data',
                        default='/mnt/d/data/snu_snub_train_normal_nodule_only',
                        help='Path to training data folder')
    parser.add_argument('--valid_data',
                        default='/mnt/d/data/snu_snub_test_normal_nodule_only',
                        help='Path to training data folder')
    parser.add_argument('--checkpoint',
                        default = None,
                        # default = './best.pt',
                        help='Path to pretrained model checkpoint (or None if starting from scratch)')
    # data_snu_snub_train_1900normals
    # MEAN: 0.5263558626174927
    # SD  : 0.25668418407440186
    parser.add_argument('--mean', default = 0.541584312915802,
                        help='mean value to use for dataset normalization')
    parser.add_argument('--sd', default=0.24306267499923706,
                        help='SD value to use for dataset normalization')
    parser.add_argument('--save_dir', default=r'.',
                        help='Path to save trained models')
    parser.add_argument('--save_name', default='best.pt')
    
    # Basic hyperparameters
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--use_fp16', default=True, type=bool)
    parser.add_argument('--log_interval', default=10, type=int)
    
    # hyperparameters for schedulers
    parser.add_argument('--lr', default=0.00002, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.4, type=float)
    
    # hyperparameters for gradient descent
    parser.add_argument('--clip_grad', default=3.0, type=float,
                        help="""Maximal parameter gradient norm if using gradient clipping.
                        Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures.
                        0 for disabling.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output layer fixed.
                        Typically doing so during the first epoch helps training.
                        Try increasing this value if the loss does not decrease.""")
    
    # args = parser.parse_args("")
    return parser
# %%
def make_dataloader(args, data_path, batch_size):
    if not args.mean or not args.sd:
        args.mean, args.sd = utils.calculate_mean_SD(data_path)
    train_transform = T.Compose([
        # T.RandomResizedCrop(256, scale=(0.95, 1), interpolation=Image.BICUBIC),
        T.Resize((256, 256)),
        # T.RandomRotation(degrees=(-15,15)),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5)),
        utils.GaussianBlurInference(),
        T.ToTensor(),
        T.Normalize(args.mean, args.sd),
    ])
    dataset = ImageFolder(root=data_path, transform=train_transform)
    classes = os.listdir(data_path)
    class_counts = {cl:0 for cl in classes}
    for cl in class_counts:
        class_counts[cl] = len(os.listdir(os.path.join(data_path, cl)))
    print("\n Number of files in the dataset for each class: \n", class_counts)
    class_counts = {dataset.class_to_idx[cl]:count for cl, count in class_counts.items()}
    print("class_to_idx: \n", class_counts)
    weights = [1/class_counts[cl] for cl in dataset.targets]
    datasampler = WeightedRandomSampler(weights=weights,
                                        num_samples=len(dataset),
                                        replacement=True)
    dataloader = DataLoader(
        dataset,
        sampler=datasampler,
        batch_size = batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

#%%
def load_model(args):
    device = torch.device('cuda')
    model = ViT_Classifier(vit_arch='vit_small',
                           vit_patch_size=8,
                           n_last_blocks=4,
                           avgpool_patchtokens=False,
                           vit_checkpoint_path=None,
                           num_labels=2,
                           linear_clf_checkpoint_path=None)
    
    param_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(param_groups)
    fp16_scaler = torch.cuda.amp.GradScaler()
    
    if args.checkpoint:
        utils.restart_from_checkpoint(
            ckp_path=args.checkpoint,
            model=model.cuda(),
            optimizer=optimizer,
            fp16_scaler=fp16_scaler
        )
    model.eval()
    
    return model.to(device), optimizer, fp16_scaler

# %%
def train(args):
    wandb.init(project='vit-nodule-detection')
    train_loader = make_dataloader(args, args.train_data, batch_size=args.batch_size)
    valid_loader = make_dataloader(args, args.valid_data, batch_size=1)
    model, optimizer, fp16_scaler = load_model(args)
    model = model.to(args.device)
    # bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
        
    # initialize schedulers
    lr_schedule = utils.cosine_scheduler(
        base_value= args.lr * (args.batch_size * utils.get_world_size()) / 16.,
        final_value= args.min_lr,
        epochs= args.epochs,
        niter_per_ep= len(train_loader),
        warmup_epochs= args.warmup_epochs
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(train_loader)
    )
    
    best_loss = 9999
    cum_avg_loss = 0
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")
        
        # TRAINING LOOP
        for it, (images, labels) in enumerate(tqdm(train_loader)):
            it = len(train_loader) * epoch + it # global training iteration count
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[it]
                if i == 0: # only the first group is regularized
                    param_group['weight_decay'] = wd_schedule[it]
                    
            images = images.to(args.device)
            labels = labels.to(args.device)
                        
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                preds = model(images)
                loss = ce_loss(preds, labels)
                
            optimizer.zero_grad()
            
            if not math.isfinite(loss.item()):
                print(f"Loss is invalid: \n {loss}")
                print("Stopping training.")
                sys.exit(1)
            
            fp16_scaler.scale(loss).backward()                    
        
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            
            cum_avg_loss += loss.item()
            if (it+1) % args.log_interval == 0:
                wandb.log({'train_loss': loss.item()}, step=it)
                cum_avg_loss /= args.log_interval

                if cum_avg_loss <= best_loss:
                    best_loss = cum_avg_loss
                    save_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'fp16_scaler': fp16_scaler.state_dict()
                    }
                    utils.save_on_master(save_dict,
                                         os.path.join(args.save_dir, args.save_name))
                    print(f'Saved {args.save_name} trained on {it+1} batches')
                    print(f'LOSS: {best_loss}')
                cum_avg_loss = 0
        
        
        # SAVE AFTER EVERY EPOCH OF TRAINING
        save_dict = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'fp16_scaler':fp16_scaler.state_dict()
        }
        utils.save_on_master(save_dict, os.path.join(args.save_dir, f'checkpoint_epoch{epoch:04}.pt'))
        
        # EVAL LOOP AFTER EVERY EPOCH OF TRAINING
        print(f"Running evaluation for epoch {epoch}")
        preds, trues = [], []
        epoch_val_loss = 0
        for i, (x,y) in enumerate(tqdm(valid_loader)):
            x, y = x.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                yhat = model(x)
                loss = ce_loss(yhat, y).detach().cpu().numpy()
                yhat = yhat.detach().cpu().numpy().argmax()
                
                preds.append(yhat)
                trues.append(y.item())
                epoch_val_loss += loss
        
        epoch_val_loss /= len(valid_loader)
        trues, preds = np.vstack(trues), np.vstack(preds)
        roc_auc = sklearn.metrics.roc_auc_score(trues, preds, average=None)
        auc_pr = sklearn.metrics.average_precision_score(trues, preds, average=None)
        f1_score = sklearn.metrics.f1_score(trues, np.round(preds), average=None)
        wandb.log({'valid_loss': epoch_val_loss, 'AUROC':roc_auc,
                   'AUC-PR':auc_pr, "F1_score":f1_score}, step=epoch)
#%%
if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args("")
    wandb.login()
    train(args)
    
    print('done')
# %%
