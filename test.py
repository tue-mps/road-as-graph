import random
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from util import get_arguments, adj_mat_mask_gene, draw_graph, img_list_from_csv, graph_nms, gt_label_parse_as_dict
from dataloaders.loader import *
from model import *
from metrics import patch_road_topology_reader

save_img = True
args = get_arguments()
with open(args.json_path) as json_file:
    configs = json.load(json_file)

if args.rand_seed == None:
    seed = configs['random_seed']
else:
    seed = args.rand_seed
epoch = args.test_epoch

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
adj_threshold = configs['adj_threshold']

if epoch is not None:
    checkpoint_path = 'checkpoints/' + method + '_seed' + str(seed) +'_epoch_' + str(epoch) + '.pth.tar'
else:
    checkpoint_path = 'checkpoints/' + method + '_seed' + str(seed) + '.pth.tar'
# print(checkpoint_path)
# load gt as dict
gt_file = dataset_dir + 'argoverse_gt.csv'
# gt_file = dataset_dir + 'argoverse_gt_based_on_sota_pred.csv'
gt_dict = gt_label_parse_as_dict(gt_file)
# Define dataloaders
test_img_list = img_list_from_csv(dataset_dir+'/test.csv')
test_set = RoadLayoutDataset(dataset_dir+'/test.csv', transform=transforms.Compose([ToTensor()]), thinning=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=6, drop_last=False)


model = IGITrans(num_nodes, bidirectional, self_edge)
model.to(device)

if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
else:
    raise Exception('checkpoint not found')

model.eval()
gt_labels = []
pred_labels = []
node_count = 0

for i, temp_batch in enumerate(test_loader):
    temp_img = temp_batch['img'].float().to(device).unsqueeze(1)
    temp_img_file = test_img_list[i]
    with torch.set_grad_enabled(False):
        pred_img_t, pred_nodes_t, pred_adj_mat_t, pred_coarse_img_t, node_heat_t, \
        pred_img, pred_nodes, pred_adj_mat, pred_coarse_img, node_heat = model(temp_img, 'val', device)

        pred_graph_t = {'nodes': pred_nodes_t.cpu().numpy(),
                        'adj': (pred_adj_mat_t.cpu().numpy().
                                reshape((num_nodes, num_nodes)) > adj_threshold).astype(np.int64)}
        pred_graph_t = graph_nms(pred_graph_t, image_size, True, bidirectional, self_edge)
        adj_mat_mask_t = adj_mat_mask_gene(pred_graph_t['nodes'].shape[0], bidirectional, self_edge).numpy()
        pred_img_from_graph_t = draw_graph(pred_graph_t['nodes'], pred_graph_t['adj'], adj_mat_mask_t, draw_node=True)

        pred_graph = {'nodes': pred_nodes.cpu().numpy(),
                      'adj': (pred_adj_mat.cpu().numpy().
                              reshape((pred_nodes.size(0), pred_nodes.size(0))) > adj_threshold).astype(np.int64)}
        adj_mat_mask = adj_mat_mask_gene(pred_graph['nodes'].shape[0], bidirectional, self_edge).numpy()
        pred_img_from_graph = draw_graph(pred_graph['nodes'], pred_graph['adj'], adj_mat_mask, draw_node=True)

        pred_label = patch_road_topology_reader(pred_graph)
        pred_labels.append(pred_label)
        gt_labels.append(gt_dict[temp_img_file.split('/')[-1]])

        node_count += pred_graph['nodes'].shape[0]


        if i < 99999 and save_img:
            io.imsave(temp_img_file.replace('images', 'pred_imgs')[:-4] + '_' + method + '.png',
                      (np.clip(pred_img.cpu().numpy()[0, 0, :, :], 0, 1) * 255).astype(np.uint8), check_contrast=False)
            io.imsave(temp_img_file.replace('images', 'pred_node_heats')[:-4] + '_' + method + '.png',
                      (np.clip(node_heat.cpu().numpy()[0, 0, :, :], 0, 1) * 255).astype(np.uint8), check_contrast=False)
            io.imsave(temp_img_file.replace('images', 'pred_coarse_imgs')[:-4] + '_' + method + '.png',
                      (np.clip(pred_coarse_img.cpu().numpy()[0, 0, :, :], 0, 1) * 255).astype(np.uint8), check_contrast=False)
            io.imsave(temp_img_file.replace('images', 'pred_imgs_from_graph')[:-4] + '_' + method + '.png',
                      pred_img_from_graph, check_contrast=False)
            io.imsave(temp_img_file.replace('images', 'pred_imgs_from_graph_t')[:-4] + '_' + method + '.png',
                      pred_img_from_graph_t, check_contrast=False)
            # np.savetxt(temp_img_file.replace('imgs', 'gt_graphs_txt')[:-4] + '_' + method + '_node.txt',
            #            gt_graph['nodes'])
            # np.savetxt(temp_img_file.replace('imgs', 'gt_graphs_txt')[:-4] + '_' + method + '_adj.txt',
            #            gt_graph['adj'])
            np.savetxt(temp_img_file.replace('images', 'pred_graphs_txt')[:-4] + '_' + method + '_node.txt',
                       pred_graph['nodes'])
            np.savetxt(temp_img_file.replace('images', 'pred_graphs_txt')[:-4] + '_' + method + '_adj.txt',
                       pred_graph['adj'])


accuracy = accuracy_score(gt_labels, pred_labels)
# print('accuracy', accuracy)
# print('confusion matrix')
# print(confusion_matrix(gt_labels, pred_labels))
avg_node_num = node_count / len(test_img_list)
# print('avg_node_num',  avg_node_num)
print(checkpoint_path, 'accuracy', accuracy, 'avg_num_nodes', avg_node_num)
