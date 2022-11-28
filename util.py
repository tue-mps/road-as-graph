import argparse
import time
import torch
import torch.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from metrics import bbox_overlaps
from PIL import ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from skimage import io


def template_img_loader(cls_dict):
    num_temps = len(cls_dict.keys())
    res_cls_dict = dict((v, k) for k, v in cls_dict.items())
    img_list = []
    for i in range(num_temps):
        img = io.imread('misc/'+res_cls_dict[i]+'.png')[:,:,0]
        img_list.append(img)

    return img_list


def get_arguments():
    parser = argparse.ArgumentParser(description="Config file loading")
    parser.add_argument("--json_path", type=str, required=True, help="The path to the json file")
    parser.add_argument("--rand_seed", type=int, required=False, help="The random seed")
    parser.add_argument("--test_epoch", type=int, required=False, help="test which epoch of ckpt")

    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adj_mat_mask_gene(num_nodes, bidirectional, self_edge):
    one_mat = torch.ones((num_nodes, num_nodes))
    if bidirectional and self_edge:
        adj_mat_mask = one_mat.to(torch.bool)
    elif (not bidirectional) and self_edge:
        adj_mat_mask = torch.triu(one_mat, diagonal=0).to(torch.bool)
    elif (not bidirectional) and (not self_edge):
        adj_mat_mask = torch.triu(one_mat, diagonal=1).to(torch.bool)
    elif bidirectional and (not self_edge):
        adj_mat_mask = ~ torch.eye(num_nodes).to(torch.bool)

    return adj_mat_mask


def graph_nms(graph, image_size, node_as_point, bidirectional, self_edge, box_size_ratio=0.1, iou_thresh=0.15):
    nodes = graph['nodes']
    adj_mat = graph['adj']
    h, w = image_size

    if node_as_point:
        boxes = np.concatenate((nodes-box_size_ratio/2, nodes+box_size_ratio/2), axis=1)
        boxes = np.clip(boxes, 0, 1)
        boxes = (boxes * np.array([[w, h, w, h]])).astype(np.int64)
    else:
        raise NotImplementedError

    if (not bidirectional) and (not self_edge):
        adj_mat = torch.triu(torch.from_numpy(adj_mat), diagonal=1)
        adj_mat = adj_mat + torch.transpose(adj_mat, 0, 1)
    else:
        raise NotImplementedError

    ious = bbox_overlaps(torch.from_numpy(boxes), torch.from_numpy(boxes)).numpy()
    delete_idx = []

    for idx in range(adj_mat.size(0)):
        if idx in delete_idx:
            continue
        matched_idx = ious[idx] > iou_thresh
        lookup_idxs = list(np.where(matched_idx == True)[0])
        # delete index of the node itself
        lookup_idxs.remove(idx)

        for lookup_idx in lookup_idxs:

            idx2compare = np.ones((adj_mat.size()[0],), bool)
            idx2compare[idx] = False
            idx2compare[lookup_idx] = False

            differ = adj_mat[idx,:][idx2compare] != adj_mat[lookup_idx,:][idx2compare]
            if sum(differ) == 0:
                delete_idx.append(lookup_idx)

    nodes = np.delete(nodes, delete_idx, 0)
    adj_mat = np.delete(np.delete(adj_mat, delete_idx, 0), delete_idx, 1)

    return {'nodes': nodes, 'adj': adj_mat.numpy()}


def draw_graph(nodes, adj_mat, adj_mat_mask, draw_node=False):
    # rectangle
    N_nodes = nodes.shape[0]
    N_edges = N_nodes * N_nodes

    nodes_mat = np.concatenate([nodes.repeat(N_nodes, axis=0), np.tile(nodes, (N_nodes, 1))], axis=1)
    adj_mat = adj_mat.reshape(-1, 1)
    adj_mat_mask = adj_mat_mask.reshape(N_nodes*N_nodes,).astype(np.bool)
    valid_nodes = nodes_mat[adj_mat_mask,:]
    valid_adj = adj_mat[adj_mat_mask,:].reshape((-1, )).astype(np.bool)

    visible_adj = valid_nodes[valid_adj] * 128

    img = np.zeros((128, 128, 3), dtype=np.uint8)
    PIL_image = Image.fromarray(img)
    draw = ImageDraw.Draw(PIL_image)
    for i in range(visible_adj.shape[0]):
        draw.line([(visible_adj[i, 0], visible_adj[i, 1]), (visible_adj[i, 2], visible_adj[i, 3])],
                  fill=(255, 255, 255), width=7)

    if draw_node:
        for i in range(N_nodes):
            node_x, node_y = nodes[i] * 128
            draw.ellipse((node_x - 8, node_y - 8, node_x + 8, node_y + 8), fill='red', outline='red')

    img = np.asarray(PIL_image, dtype=np.uint8)

    # PIL_image.show()

    return img


def heat_map_gene_batch(graphs, map_size):
    """
    gene a batch of heat maps for node attentions supervision in baseline
    :param graphs: a list of graph dicts
    :param map_size:
    :return: tensor with size (batch_size, 1, h ,w) ranging from 0-1
    """
    h, w = map_size
    batch_size = len(graphs)
    maps = torch.zeros((batch_size, 1, h, w))

    for i in range(batch_size):
        graph = graphs[i]
        heat_map = heat_map_gene(graph, map_size)
        maps[i, 0, ...] = torch.from_numpy(heat_map)

    return maps


def heat_map_gene(graph, map_size):
    """
    generate a heatmap from a single graph
    :param graph: a dict with 'nodes' and 'adj' (not used)
    :param map_size:
    :return: numpy float array, range 0.-1., (h, w)
    """
    nodes = graph['nodes']
    h, w = map_size
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    z = np.zeros((h, w), dtype=np.float)

    for i in range(nodes.shape[0]):
        xc = nodes[i, 0]
        yc = nodes[i, 1]

        z += multivariate_gaussian(pos, np.array([xc, yc]), np.array([[5, 0], [0, 5]]))

    z = z / np.max(z)

    return z


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def img_list_from_csv(csv_file):
    samples = pd.read_csv(csv_file, header=None)
    return list(samples.iloc[:, 0])


def gt_label_parse_as_dict(csv_file):
    gts = pd.read_csv(csv_file, header=None, index_col=None)
    gt_dict = {}
    for i in range(gts.shape[0]):
        file_name = gts.iloc[i, 0]
        one_hot = gts.iloc[i, 1:].values
        label = np.argmax(one_hot)
        gt_dict[file_name] = label

    return gt_dict


if __name__ == '__main__':
    g = {
        'nodes': np.array([[0.1, 0.1],
                           [0.3, 0.11],
                           [0.105, 0.105],
                           [0.0, 0.0]]),
        'adj': np.array([[0, 1, 1, 0],
                         [1, 0, 1, 1],
                         [1, 1, 0, 1],
                         [0, 1, 1, 0]])
    }
    # g = graph_nms(g, (128, 128), True, False, False)
    # draw_graph(g['nodes'], g['adj'], np.array([[0, 1, 1, 1],
    #                                          [0, 0, 1, 1],
    #                                          [0, 0, 0, 1],
    #                                          [0, 0, 0, 0]]))

    # heat_map_gene(g, (128, 128))
    # maps = heat_map_gene_batch([g, g], (128, 128))
    # print(maps.size())

    # g_sorted = graph_reorder(g)



