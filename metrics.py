import torch
import numpy as np


'''
Functions for triplet matching scores
'''

class TripletsEvaluator():
    def __init__(self):
        self.tp_in_pred = 0
        self.tp_in_gt = 0
        self.fp = 0
        self.fn = 0

    def eval_one_pair(self, pred, gt, image_size, bidirectional):
        gt_rels, gt_boxes, gt_classes = g2tensor4eval(gt['nodes'], gt['adj'], image_size, bidirectional=bidirectional)
        pred_rels, pred_boxes, pred_classes = g2tensor4eval(pred['nodes'], pred['adj'], image_size, bidirectional=bidirectional)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                    gt_rels[:, :2],
                                                    gt_classes,
                                                    gt_boxes)

        pred_triplets, pred_triplet_boxes, _ = _triplet(pred_rels[:, 2],
                                                        pred_rels[:, :2],
                                                        pred_classes,
                                                        pred_boxes)

        pred_triplets_rev = pred_triplets[:, [2, 1, 0]]
        pred_triplet_boxes_rev = pred_triplet_boxes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        pred_triplets_dual = np.concatenate((pred_triplets, pred_triplets_rev), axis=0)
        pred_triplet_boxes_dual = np.concatenate((pred_triplet_boxes, pred_triplet_boxes_rev), axis=0)

        # delete redundant matched triplet in predictions (as well as two connected nodes at the same location)
        pred_to_pred_dual = _compute_pred_matches(
            pred_triplets_dual,
            pred_triplets,
            pred_triplet_boxes_dual,
            pred_triplet_boxes,
            0.15,
        )
        delete_idx = []
        num_pred_tri = len(pred_to_pred_dual)
        for match_list in pred_to_pred_dual:
            trasnformed_match_list = [item % num_pred_tri for item in match_list]
            if len(trasnformed_match_list) > 1:
                trasnformed_match_list = sorted(trasnformed_match_list)
                delete_idx += trasnformed_match_list[1:]
        delete_idx = list(dict.fromkeys(delete_idx))
        pred_triplets = np.delete(pred_triplets, delete_idx, 0)
        pred_triplet_boxes = np.delete(pred_triplet_boxes, delete_idx, 0)


        if not bidirectional:
            gt_triplets_rev = gt_triplets[:, [2, 1, 0]]
            gt_triplet_boxes_rev = gt_triplet_boxes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
            gt_triplets = np.concatenate((gt_triplets, gt_triplets_rev), axis=0)
            gt_triplet_boxes = np.concatenate((gt_triplet_boxes, gt_triplet_boxes_rev), axis=0)

            pred_triplets_rev = pred_triplets[:, [2, 1, 0]]
            pred_triplet_boxes_rev = pred_triplet_boxes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
            pred_triplets = np.concatenate((pred_triplets, pred_triplets_rev), axis=0)
            pred_triplet_boxes = np.concatenate((pred_triplet_boxes, pred_triplet_boxes_rev), axis=0)

        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            0.15,
        )

        gt_to_pred = _compute_pred_matches(
            pred_triplets,
            gt_triplets,
            pred_triplet_boxes,
            gt_triplet_boxes,
            0.15,
        )

        num_tri_pred = len(pred_to_gt)
        num_tri_gt = len(gt_to_pred)
        temp_tp_in_pred = 0
        temp_tp_in_gt = 0
        for match_list in pred_to_gt:
            temp_tp_in_pred += len(match_list)

        for match_list in gt_to_pred:
            if len(match_list) >0:
                temp_tp_in_gt += 1

        temp_fp = num_tri_pred - temp_tp_in_pred
        temp_fn = num_tri_gt - temp_tp_in_gt

        self.tp_in_pred += temp_tp_in_pred
        self.tp_in_gt += temp_tp_in_gt
        self.fp += temp_fp
        self.fn += temp_fn

    def get_stat(self):

        if self.tp_in_pred + self.fp > 0:
            precision = self.tp_in_pred / (self.tp_in_pred + self.fp)
        else:
            precision = 0.
        if self.tp_in_gt + self.fn > 0:
            recall = self.tp_in_gt / (self.tp_in_gt + self.fn)
        else:
            recall = 0.
        if precision + recall > 0:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0.

        return precision, recall, F1


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return: a list of lists, outside list represents the predicted triplets, the nested list for each predicted
            triplet is the indexs of the ground truth triplet
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]

        sub_iou = bbox_overlaps(torch.from_numpy(gt_box[None,:4]).contiguous(), torch.from_numpy(boxes[:, :4]).contiguous()).numpy()[0]
        obj_iou = bbox_overlaps(torch.from_numpy(gt_box[None,4:]).contiguous(), torch.from_numpy(boxes[:, 4:]).contiguous()).numpy()[0]

        inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = torch.true_divide(iw * ih, ua)

    return overlaps


def g2tensor4eval(nodes, ajd_mat, image_size, bidirectional=False, node_as_point=True, box_size_ratio=0.1):
    """
    expected outputs:
    relations: [#gt_rel, 3] array of relations, for each row, the elements are
                [h_node_index, w_node_index, relation_label]
    boxes: [#gt_box, 4] array of boxes
    classes: [#gt_box] array of classes, currently values should all be 1, since the relation is binary, and zero
            means no relation (invalid)
    """
    h, w = image_size
    if node_as_point:
        boxes = np.concatenate((nodes-box_size_ratio/2, nodes+box_size_ratio/2), axis=1)
        boxes = np.clip(boxes, 0, 1)
        boxes = (boxes * np.array([[w, h, w, h]])).astype(np.int64)
    else:
        raise NotImplementedError

    if bidirectional:
        raise NotImplementedError
    else:
        idxs_h, idxs_w = np.triu_indices(boxes.shape[0], k=1)
    all_relations = np.transpose(np.vstack((idxs_h, idxs_w, ajd_mat[(idxs_h,idxs_w)])))
    valid_relations_idx = all_relations[:, 2] > 0
    valid_relations = all_relations[valid_relations_idx]

    classes = np.ones(boxes.shape[0])

    return valid_relations, boxes, classes


'''
Functions for road layout topology classification
'''

def if_node_at_edge(nodes, boundary_thresh=0.2):
    # given a list of nodes, assign a corresponding list indicating whether on edge and which edge

    x_coords = nodes[:, 0].reshape(1, -1)
    y_coords = nodes[:, 1].reshape(1, -1)

    # if most left:
    if_at_left = x_coords < boundary_thresh
    if_at_right = x_coords > (1 - boundary_thresh)
    if_at_top = y_coords < boundary_thresh
    if_at_bottom = y_coords > (1 - boundary_thresh)

    # nodes_location is of size (num_nodes, 4), each line indicate if the node is on (left, top, right, bottom)
    nodes_location = np.concatenate((if_at_left, if_at_top, if_at_right, if_at_bottom)).transpose()

    return nodes_location

# sample graph implemented as a dictionary


def graph_format_converter(adj_matrix):
    # input format is nodes_num * nodes_num matrix with binary element value indicating the connectivity of two nodes
    # output format is dictionary indicated in the following bfs_connected_component
    # NOTE: self-connection and bidirectional connection should be addressed beforehand, this function process general
    # adjacency matrices
    num_nodes = adj_matrix.shape[0]
    graph_dict = {}

    for i in range(num_nodes):
        # loop over rows
        row = adj_matrix[i, :]
        connected_nodes = np.where(row > 0.5)[0].tolist()
        graph_dict[i] = connected_nodes

    return graph_dict


# visits all the nodes of a graph (connected component) using BFS
def bfs_connected_component(graph, start):
    # graph format should be like this
    # graph = {'A': ['B', 'C', 'E'],
    #          'B': ['A', 'D', 'E'],
    #          'C': ['A', 'F', 'G'],
    #          'D': ['B'],
    #          'E': ['A', 'B', 'D'],
    #          'F': ['C'],
    #          'G': ['C']}
    # keep track of all visited nodes
    explored = []
    # keep track of nodes to be checked
    queue = [start]

    levels = {}         # this dict keeps track of levels
    levels[start]= 0    # depth of start node is 0

    visited= [start]     # to avoid inserting the same node twice into the queue

    # keep looping until there are nodes still to be checked
    while queue:
       # pop shallowest node (first node) from queue
        node = queue.pop(0)
        explored.append(node)
        neighbours = graph[node]

        # add neighbours of node to queue
        for neighbour in neighbours:
            if neighbour not in visited:
                queue.append(neighbour)
                visited.append(neighbour)

                levels[neighbour]= levels[node]+1
                # print(neighbour, ">>", levels[neighbour])

    return explored


def patch_road_topology_reader(graph):
    nodes = graph['nodes']
    adj = graph['adj']

    # nodes_location is of size (num_nodes, 4), each line indicate if the node is on (left, top, right, bottom)
    nodes_location = if_node_at_edge(nodes)

    # if nodes exist at bottom edge, then we can define one of the 8 classes, otherwise the label is unknown
    bottom_status = nodes_location[:, 3]
    bottom_node_idxes = np.where(bottom_status > 0.5)[0]
    num_bottom_nodes = len(bottom_node_idxes)
    if num_bottom_nodes == 0:
        # cannot find node at bottom edge, which is not logical for ego-centric based road layout, return 8 (others)
        return 8

    elif num_bottom_nodes >= 1:
        bottom_nodes = nodes[bottom_node_idxes]
        x_coords = bottom_nodes[:, 0]
        distance_to_center = np.abs(x_coords - 0.5)
        bottom_node_idx_of_idx = np.argsort(distance_to_center)[0]
        bottom_node_idx = bottom_node_idxes[bottom_node_idx_of_idx]
    else:
        bottom_node_idx = bottom_node_idxes[0]

    # given the found index of node at bottom,
    # we use BFS to find out the connectivity between bottom node and other nodes
    # first, convert the adj matrix to the format of dictionary
    graph_dict = graph_format_converter(adj)
    visited_nodes = bfs_connected_component(graph_dict, bottom_node_idx)

    # if different edges are reachable
    left_status_for_each_node = nodes_location[:, 0]
    if True in left_status_for_each_node[visited_nodes]:
        left_status = '1'
    else:
        left_status = '0'
    top_status_for_each_node = nodes_location[:, 1]
    if True in top_status_for_each_node[visited_nodes]:
        top_status = '1'
    else:
        top_status = '0'
    right_status_for_each_node = nodes_location[:, 2]
    if True in right_status_for_each_node[visited_nodes]:
        right_status = '1'
    else:
        right_status = '0'

    overall_status_code = left_status + top_status + right_status
    status2label_dict = {
        '000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7
    }
    label = status2label_dict[overall_status_code]

    return label


if __name__ == '__main__':

    evaluator = TripletsEvaluator()

    gt_1 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.75, 0.25]]), 'adj': np.array([[0, 1, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 0]])}
    pred_1 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0.25, 0.75]], dtype=np.float32), 'adj': np.array([[0, 1, 1, 0, 1],
                                                                        [0, 0, 1, 1, 1],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0]])}

    gt_2 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.75, 0.25]]), 'adj': np.array([[0, 1, 1],
                                                      [1, 0, 0],
                                                      [1, 1, 0]])}
    pred_2 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0.25, 0.75]], dtype=np.float32), 'adj': np.array([[0, 0, 1, 0, 1],
                                                                        [0, 0, 1, 1, 0],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0]])}

    gt_3 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.75, 0.25]]), 'adj': np.array([[0, 0, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 0]])}
    pred_3 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0.25, 0.75]], dtype=np.float32), 'adj': np.array([[0, 1, 1, 0, 1],
                                                                        [0, 0, 1, 1, 1],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0]])}
    gt_4 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.75, 0.25]]), 'adj': np.array([[0, 0, 1],
                                                      [1, 0, 0],
                                                      [1, 1, 0]])}
    pred_4 = {'nodes': np.array([[0.25, 0.75],
                     [0.75, 0.75],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0.25, 0.75]], dtype=np.float32), 'adj': np.array([[0, 1, 1, 0, 1],
                                                                        [0, 0, 1, 1, 1],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0]])}

    evaluator.eval_one_pair(pred_1, gt_1, (128, 128), False)
    evaluator.eval_one_pair(pred_2, gt_2, (128, 128), False)
    evaluator.eval_one_pair(pred_3, gt_3, (128, 128), False)
    evaluator.eval_one_pair(pred_4, gt_4, (128, 128), False)
    print(evaluator.get_stat())


    temp_road_graph = {'nodes': np.array([[0.5, 0.1],
                     [0.05, 0.5],
                     [0.5, 0.5],
                     [0.95, 0.5],
                     [0.5, 0.95],
                     [0.7, 0.95]], dtype=np.float32), 'adj': np.array([[0, 0, 1, 0, 0, 0],
                                                                        [0, 0, 1, 0, 0, 0],
                                                                        [1, 1, 0, 1, 1, 0],
                                                                        [0, 0, 1, 0, 0, 0],
                                                                        [0, 0, 1, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0, 0]])}

    graph_in_dict = graph_format_converter(temp_road_graph['adj'])
    print(patch_road_topology_reader(temp_road_graph))


