import random
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim, ms_ssim
from util import get_arguments, adj_mat_mask_gene, draw_graph, graph_nms
from dataloaders.loader import *
from model import *
from metrics import TripletsEvaluator

args = get_arguments()
with open(args.json_path) as json_file:
    configs = json.load(json_file)

if args.rand_seed == None:
    seed = configs['random_seed']
else:
    seed = args.rand_seed
torch.backends.cudnn.benchmark = configs['cudnn_benchmark']
torch.backends.cudnn.deterministic = configs['cudnn_deterministic']
restore = configs['restore']
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')

dataset = configs['dataset']
dataset_dir = configs['dataset_dir']
image_size = configs['image_size']
bidirectional = configs['bidirectional']
self_edge = configs['self_edge']
num_nodes = configs['num_nodes']
method = configs['method']
loss_mode = configs['loss_mode']

num_epochs = configs['num_epochs']
number_main_each_batch = configs['number_main_each_batch']
number_argo_each_batch = configs['number_argo_each_batch']
flag_fuse_two_datasets = configs['fuse_two_datasets']
flag_only_argo = configs['only_argo']
lr = configs['lr']
betas = configs['betas']
weight_decay = configs['weight_decay']
adj_threshold = configs['adj_threshold']

checkpoint_path = 'checkpoints/' + method + '_seed' + str(seed) + '.pth.tar'
writer = SummaryWriter(comment='_' + method + '_seed' + str(seed))

# Define dataloaders
val_with_graph = True
train_set = RoadLayoutDataset(dataset_dir+'/train.csv', transform=transforms.Compose([ToTensor()]), thinning=False)
train_set_argo = RoadLayoutDataset('datasets/argoverse-tracking-sorted-thinning/train.csv', transform=transforms.Compose([ToTensor()]), thinning=False)
val_set = RoadLayoutDataset(dataset_dir+'/test.csv', with_graph=val_with_graph,
                              transform=transforms.Compose([ToTensor(with_graph=val_with_graph)]), thinning=False)
train_loader = DataLoader(train_set, batch_size=number_main_each_batch, shuffle=True, num_workers=6, drop_last=True)
train_loader_argo = DataLoader(train_set_argo, batch_size=number_argo_each_batch, shuffle=True, num_workers=6, drop_last=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
argo_iter = iter(train_loader_argo)
dataloaders = {'train': train_loader, 'val': val_loader}


model = IGITrans(num_nodes, bidirectional, self_edge)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

if restore:
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
    else:
        epoch = 0
else:
    epoch = 0

while epoch < num_epochs:
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            evaluator_stu = TripletsEvaluator()
            evaluator_tea = TripletsEvaluator()
            pred_coarse_imgs = []
            pred_imgs = []
            pred_coarse_t_imgs = []
            pred_t_imgs = []
            pred_imgs_from_g = []
            pred_imgs_t_from_g = []
            temp_imgs = []
            stacked_heat_t_imgs = []
            heat_imgs = []
            temp_val_loss = 0.0

        for i, temp_batch in enumerate(dataloaders[phase]):
            temp_img = temp_batch['img'].float().to(device).unsqueeze(1)
            if phase == 'val':
                temp_graph = temp_batch['graph']
            if phase == 'train':
                try:
                    temp_img_argo = next(argo_iter)['img'].float().to(device).unsqueeze(1)
                except StopIteration:
                    argo_iter = iter(train_loader_argo)
                    temp_img_argo = next(argo_iter)['img'].float().to(device).unsqueeze(1)

            with torch.set_grad_enabled(phase == 'train'):

                if phase == 'train' and flag_fuse_two_datasets:
                    temp_img = torch.cat([temp_img, temp_img_argo], dim=0).detach()
                elif phase == 'train' and flag_only_argo:
                    temp_img = temp_img_argo.detach()


                pred_img_t, pred_nodes_t, pred_adj_mat_t, pred_coarse_img_t, node_heat_t, \
                pred_img, pred_nodes, pred_adj_mat, pred_coarse_img, node_heat = model(temp_img, phase, device)

                stacked_heats_t = normalize_and_stack(node_heat_t)
                loss_heat = F.binary_cross_entropy_with_logits(node_heat, (F.interpolate(stacked_heats_t,
                                    size=node_heat.size()[2:4], mode='nearest').detach() > 0.5).float(),
                                                               pos_weight=torch.tensor(1.).to(device))

                loss_ssim = 1 - ms_ssim(pred_img_t, temp_img, win_size=7, data_range=1, size_average=True)
                loss_ssim_C = 1 - ms_ssim(pred_coarse_img_t, temp_img, win_size=7, data_range=1, size_average=True)

                loss_total = loss_ssim + loss_heat + 0.1 * loss_ssim_C


                if phase == 'train':
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

                    global_step = epoch*len(train_set)/number_main_each_batch+i
                    writer.add_scalar(phase + '_loss_ssim', loss_ssim, global_step)
                    writer.add_scalar(phase + '_loss_ssim_C', loss_ssim_C, global_step)
                    writer.add_scalar(phase + '_loss_total', loss_total, global_step)
                    writer.add_scalar(phase + '_loss_heat', loss_heat, global_step)

                else:
                    pred_imgs.append(pred_img)
                    pred_coarse_imgs.append(pred_coarse_img)
                    pred_t_imgs.append(pred_img_t)
                    pred_coarse_t_imgs.append(pred_coarse_img_t)
                    temp_imgs.append(temp_img)
                    stacked_heat_t_imgs.append((stacked_heats_t > 0.5).float())
                    heat_imgs.append(torch.sigmoid(node_heat))

                    temp_val_loss += loss_total.item()

                    pred_graph_t = {'nodes': pred_nodes_t.cpu().numpy(),
                                    'adj': (pred_adj_mat_t.cpu().numpy().
                                            reshape((num_nodes, num_nodes)) > adj_threshold).astype(np.int64)}
                    pred_graph_t = graph_nms(pred_graph_t, image_size, True, bidirectional, self_edge)
                    adj_mat_mask_t = adj_mat_mask_gene(pred_graph_t['nodes'].shape[0], bidirectional, self_edge).numpy()
                    pred_imgs_t_from_g.append(np.expand_dims(draw_graph(pred_graph_t['nodes'], pred_graph_t['adj'],
                                                                        adj_mat_mask_t), axis=0))

                    pred_graph = {'nodes': pred_nodes.cpu().numpy(),
                                  'adj': (pred_adj_mat.cpu().numpy().
                                          reshape((pred_nodes.size(0), pred_nodes.size(0))) > adj_threshold).astype(np.int64)}
                    adj_mat_mask = adj_mat_mask_gene(pred_graph['nodes'].shape[0], bidirectional, self_edge).numpy()
                    pred_imgs_from_g.append(np.expand_dims(draw_graph(pred_graph['nodes'], pred_graph['adj'],
                                                                      adj_mat_mask), axis=0))

                    gt_graph = {'nodes': temp_graph['nodes'][0].numpy()/
                                         np.broadcast_to(np.array([image_size[::-1]]), temp_graph['nodes'][0].size()),
                                'adj': temp_graph['adj'][0].numpy().astype(np.int64)}

                    evaluator_tea.eval_one_pair(pred_graph_t, gt_graph, image_size, bidirectional)
                    evaluator_stu.eval_one_pair(pred_graph, gt_graph, image_size, bidirectional)


        # statistics
        if phase == 'train':
            pass
        else:
            global_step = epoch * len(train_set)/(number_main_each_batch+1)
            temp_val_loss = temp_val_loss / len(val_set)

            P, R, F1 = evaluator_stu.get_stat()
            writer.add_scalar(phase + '_precision', P, global_step)
            writer.add_scalar(phase + '_recall', R, global_step)
            writer.add_scalar(phase + '_F1', F1, global_step)

            P_t, R_t, F1_t = evaluator_tea.get_stat()
            writer.add_scalar(phase + '_precision_teacher', P_t, global_step)
            writer.add_scalar(phase + '_recall_teacher', R_t, global_step)
            writer.add_scalar(phase + '_F1_teacher', F1_t, global_step)

            writer.add_scalar(phase + '_loss_total', temp_val_loss, global_step)
            writer.add_images('pred_img', torch.cat(pred_imgs, dim=0)[:64], global_step)
            writer.add_images('pred_coarse_img', torch.cat(pred_coarse_imgs, dim=0)[:64], global_step)
            writer.add_images('pred_t_img', torch.cat(pred_t_imgs, dim=0)[:64], global_step)
            writer.add_images('pred_t_coarse_img', torch.cat(pred_coarse_t_imgs, dim=0)[:64], global_step)
            writer.add_images('pred_img_from_g', np.expand_dims(np.concatenate(pred_imgs_from_g, axis=0), axis=1)[:64], global_step)
            writer.add_images('pred_img_t_from_g', np.expand_dims(np.concatenate(pred_imgs_t_from_g, axis=0), axis=1)[:64], global_step)
            writer.add_images('temp_img', torch.cat(temp_imgs, dim=0)[:64], global_step)
            writer.add_images('stacked_heats_t', torch.cat(stacked_heat_t_imgs, dim=0)[:64], global_step)
            writer.add_images('heats', torch.cat(heat_imgs, dim=0)[:64], global_step)

    checkpoint_path_to_save = checkpoint_path
    dict_to_dave = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
    if epoch >= num_epochs - 5:
        torch.save(dict_to_dave, checkpoint_path_to_save[:-8] + '_epoch_' + str(epoch) + checkpoint_path_to_save[-8:])
    torch.save(dict_to_dave, checkpoint_path_to_save)
    epoch += 1

writer.close()
