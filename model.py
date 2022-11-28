import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_pool, nms
from layout import masks_to_layout
from layers import Conv, Pool, Residual


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class EncoderAttentionNodesTeacher(nn.Module):
    def __init__(self, num_nodes, bidirectional, self_edge):
        super(EncoderAttentionNodesTeacher, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_nodes * num_nodes
        self.bidirectional = bidirectional
        self.self_edge = self_edge
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=True, relu=True),
            Pool(2, 2),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 64),
        )

        self.node_att = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_nodes, 3, padding=0),
        )

        self.coord_trans = DiffCoordinatesTrans()


    def forward(self, x):

        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)
        f = self.pre(x)
        nodes_heat = self.node_att(f)
        f_h, f_w = nodes_heat.size(2), nodes_heat.size(3)
        nodes_heat = F.softmax(nodes_heat.view(batch_size, self.num_nodes, -1), dim=2)
        nodes_heat = nodes_heat.view(batch_size, self.num_nodes, f_h, f_w)
        nodes = self.coord_trans(nodes_heat)

        return nodes.view(-1, 2), nodes_heat


class EncoderAttentionNodes(nn.Module):
    def __init__(self, num_nodes, bidirectional, self_edge):
        super(EncoderAttentionNodes, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_nodes * num_nodes
        self.bidirectional = bidirectional
        self.self_edge = self_edge

        self.num_grid = 16
        grid_size = int(128/self.num_grid)
        self.rect_mask_temp = torch.zeros((self.num_grid*self.num_grid, 1, 128, 128))
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                self.rect_mask_temp[i*self.num_grid+j, 0, grid_size*i:grid_size*(i+1), grid_size*j:grid_size*(j+1)] = 1

        self.node_att = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=0),
            # nn.Sigmoid()
        )

        self.coord_trans = DiffCoordinatesTrans()

    def forward(self, x, phase):

        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)
        nodes_heat_raw = self.node_att(x)
        nodes_heat_sigmoid = torch.sigmoid(nodes_heat_raw)

        grid = F.max_pool2d(nodes_heat_sigmoid, kernel_size=int(128/self.num_grid), stride=int(128/self.num_grid)).view(batch_size, 1, -1)
        sorted_grid, _ = torch.sort(grid, dim=2, descending=True)
        if phase == 'train':
            thresh = sorted_grid[:, 0, 16].view(batch_size, 1, 1)
            thresh = torch.maximum(thresh, 0.9*torch.ones_like(thresh).to(thresh.device))
        else:
            thresh = 0.7
        valid_grid_map = grid > thresh
        valid_grid = torch.nonzero(valid_grid_map.view(batch_size, 1, self.num_grid, self.num_grid))

        num_valid_grid = valid_grid[:, 0].size(0)

        atts = torch.index_select(nodes_heat_sigmoid, dim=0, index=valid_grid[:, 0])
        grid_msks = torch.index_select(self.rect_mask_temp.to(grid.device), dim=0, index=valid_grid[:, 2]*self.num_grid+valid_grid[:, 3])
        masked_node_heat = atts*grid_msks

        f_h, f_w = masked_node_heat.size(2), masked_node_heat.size(3)
        nodes_heat = masked_node_heat/torch.sum(masked_node_heat, dim=(2, 3), keepdim=True).expand(masked_node_heat.size())
        nodes_heat = nodes_heat.view(num_valid_grid, 1, f_h, f_w)
        nodes = self.coord_trans(nodes_heat).view(-1, 2)

        nodes_per_image = torch.bincount(valid_grid[:, 0], minlength=batch_size)

        if phase == 'val':
            boxes = torch.cat((torch.maximum(nodes-0.1, torch.zeros_like(nodes).to(nodes.device)), torch.minimum(nodes+0.1, torch.ones_like(nodes).to(nodes.device))), dim=1)*128
            scores = grid[grid > thresh]
            keep_idx = nms(boxes, scores, 0.1)
            nodes = nodes[keep_idx]
            nodes_per_image[0] = nodes.size(0)


        return nodes, nodes_heat_raw, nodes_per_image


class Nodes2Adj(nn.Module):
    def __init__(self):
        super(Nodes2Adj, self).__init__()

        self.adjmat_gene = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, nodes, nodes_per_image, device):

        batch_size = img.size(0)
        H, W = img.size(2), img.size(3)
        rois = []
        accu_idx = 0
        batch_idx = 0
        roi_batch_idxs = []
        for i in nodes_per_image:
            temp_nodes = nodes[accu_idx:accu_idx+i]
            temp_rois = torch.cat((torch.repeat_interleave(temp_nodes, i, dim=0), temp_nodes.repeat(i, 1)), dim=1)
            rois.append(temp_rois)
            temp_rois_batch_idxs = torch.ones((i*i,))*batch_idx
            roi_batch_idxs.append(temp_rois_batch_idxs)
            accu_idx = accu_idx + i
            batch_idx = batch_idx + 1

        roi_batch_idxs = torch.cat(roi_batch_idxs)
        rois_raw = torch.cat(rois)

        # make rois bbox in the right order
        rois_x = torch.index_select(rois_raw, 1, torch.tensor([0, 2]).to(rois_raw.device))
        rois_y = torch.index_select(rois_raw, 1, torch.tensor([1, 3]).to(rois_raw.device))
        rois_x_sorted, _ = torch.sort(rois_x, dim=1)
        rois_y_sorted, _ = torch.sort(rois_y, dim=1)

        rois = torch.cat((rois_x_sorted[:, 0].unsqueeze(1), rois_y_sorted[:, 0].unsqueeze(1),
                          rois_x_sorted[:, 1].unsqueeze(1), rois_y_sorted[:, 1].unsqueeze(1)), dim=1)

        # rescale
        rois = rois * torch.tensor([W, H, W, H]).to(nodes.device).view(1, 4).repeat(rois.size(0), 1)

        roi_batch_idxs = torch.unsqueeze(roi_batch_idxs, 1).to(rois.device)
        rois_w_imgidx = torch.cat((roi_batch_idxs, rois), dim=1)

        f_edges = roi_pool(img, rois_w_imgidx, (16, 16))
        f_edges_flip = torch.flip(f_edges, [3])

        slope = (rois_raw[:, 3] - rois_raw[:, 1]) / (rois_raw[:, 2] - rois_raw[:, 0] + 1e-6)
        right_order = slope > 0
        right_order = right_order.float().view(-1, 1, 1, 1).expand(f_edges.size())

        f_edges = f_edges * right_order + f_edges_flip * (1-right_order)

        focus_region = F.max_pool2d(torch.eye(16).view(1, 1, 16, 16), 5, stride=1, padding=2).to(device)
        focus_region = focus_region.expand_as(f_edges)
        f_edges = f_edges * focus_region


        if f_edges.size(0) == 0:
            adj_mat = torch.empty((0, 1)).to(f_edges.device)
        else:
            adj_mat = self.adjmat_gene(f_edges)

        adj_mat = torch.minimum(torch.maximum(torch.zeros_like(adj_mat, device=device), adj_mat),
                                torch.ones_like(adj_mat, device=device))

        return adj_mat


class G2ITranslator(nn.Module):
    def __init__(self):
        super(G2ITranslator, self).__init__()

    def forward(self, nodes, adj_mat, nodes_per_image, device):


        # two templates for lines, rect template is for line vanishing problem
        line_template = F.max_pool2d(torch.eye(128).view(1, 1, 128, 128), 11, stride=1, padding=5).view(128, 128).to(device)
        rect_template = torch.ones_like(line_template).to(device)

        rois = []
        accu_idx = 0
        batch_idx = 0
        roi_batch_idxs = []
        valid_idx_list = []
        for i in nodes_per_image:
            temp_nodes = nodes[accu_idx:accu_idx+i]
            if i > 1:
                temp_rois = torch.cat((torch.repeat_interleave(temp_nodes, i, dim=0), temp_nodes.repeat(i, 1)), dim=1)
                rois.append(temp_rois)
                temp_rois_batch_idxs = torch.ones((i*i,))*batch_idx
                roi_batch_idxs.append(temp_rois_batch_idxs)

                one_mat = torch.ones((i, i))
                adj_mat_mask = torch.triu(one_mat, diagonal=1).to(torch.bool).view(-1)
                valid_idx_list.append(adj_mat_mask)
            else:
                temp_rois = torch.cat((torch.repeat_interleave(temp_nodes, i, dim=0), temp_nodes.repeat(i, 1)), dim=1)
                rois.append(temp_rois)
                temp_rois_batch_idxs = torch.ones((i*i,))*batch_idx
                roi_batch_idxs.append(temp_rois_batch_idxs)

                adj_mat_mask = torch.zeros((i, i)).to(torch.bool).view(-1)
                valid_idx_list.append(adj_mat_mask)

            accu_idx = accu_idx + i
            batch_idx = batch_idx + 1

        roi_batch_idxs = torch.cat(roi_batch_idxs).to(device)
        rois = torch.cat(rois)
        valid_idx = torch.cat(valid_idx_list).to(device)

        obj_to_img = roi_batch_idxs[valid_idx][:, None].to(torch.int64)
        boxes = rois[valid_idx]
        adj_mat = adj_mat[valid_idx]
        N_patch = boxes.size(0)

        if obj_to_img.size(0):
            # apply perturbations to avoid pure vertical/horizontal problem and
            # choose templates based on the sigmoid function
            dual_slope = torch.abs((boxes[:, 3] - boxes[:, 1]) / (torch.abs(boxes[:, 2] - boxes[:, 0]) + 1e-6)) + \
                         torch.abs((boxes[:, 2] - boxes[:, 0]) / (torch.abs(boxes[:, 3] - boxes[:, 1]) + 1e-6))
            template_idx = 2 * torch.sigmoid(0.03 * (dual_slope - 2)) - 1
            template_idx = torch.minimum(torch.maximum(torch.zeros_like(template_idx, device=device), template_idx), torch.ones_like(template_idx, device=device))
            template_idx = template_idx[:, None, None].expand((-1, 128, 128))
            masks = (1 - template_idx) * line_template[None, :, :].expand((N_patch, -1, -1)) + \
                    template_idx * rect_template[None, :, :].expand((N_patch, -1, -1))
            img = masks_to_layout(adj_mat, boxes, masks, obj_to_img, 128, pooling='sum')
            if img.size(0) != nodes_per_image.size(0):
                img = torch.cat((img, torch.zeros(nodes_per_image.size(0)-img.size(0), 1, 128, 128).to(img.device)), dim=0)
            img = torch.minimum(torch.maximum(torch.zeros_like(img, device=device), img), torch.ones_like(img, device=device))
        else:
            img = torch.zeros((len(nodes_per_image), 1, 128, 128)).to(device)
        return img


class DiffCoordinatesTrans(nn.Module):
    def __init__(self):
        super(DiffCoordinatesTrans, self).__init__()

    def forward(self, heat_maps):
        (batch, chs, h, w) = heat_maps.size()
        x_reference = torch.linspace(0, 1, steps=w).view((1, 1, 1, w)).expand(batch, chs, h, -1).to(heat_maps)
        y_reference = torch.linspace(0, 1, steps=h).view((1, 1, h, 1)).expand(batch, chs, -1, w).to(heat_maps)

        x_coord = torch.sum(x_reference * heat_maps, (2, 3)).view((batch, chs, 1))
        y_coord = torch.sum(y_reference * heat_maps, (2, 3)).view((batch, chs, 1))

        nodes = torch.cat((x_coord, y_coord), dim=2)

        return nodes


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.minimum(torch.maximum(torch.zeros_like(x, device=x.device), x), torch.ones_like(x, device=x.device))
        return x


class IGITrans(nn.Module):
    def __init__(self, num_nodes, bidirectional=True, self_edge=False):
        super(IGITrans, self).__init__()

        self.num_nodes = num_nodes
        self.bidirectional = bidirectional
        self.self_edge = self_edge

        self.encoder_teacher = EncoderAttentionNodesTeacher(num_nodes, bidirectional, self_edge)
        self.encoder = EncoderAttentionNodes(num_nodes, bidirectional, self_edge)
        self.nodes2adj = Nodes2Adj()
        self.g2itranslator = G2ITranslator()
        self.decoder = Decoder()

    def forward(self, img, phase, device):

        batch_size = img.size(0)

        nodes_t, node_heat_t = self.encoder_teacher(img)
        fixed_nodes_per_image = (self.num_nodes * torch.ones((batch_size, ), dtype=torch.int64)).to(device)
        if phase == 'train':
            nodes_t = torch.clamp(nodes_t + (torch.rand(nodes_t.size()).to(nodes_t.device) - 0.5) * 0.03, 0, 1)
        adj_mat_t = self.nodes2adj(img, nodes_t, fixed_nodes_per_image, device)
        coarse_img_t = self.g2itranslator(nodes_t, adj_mat_t, fixed_nodes_per_image, device)

        nodes, node_heat, nodes_per_image = self.encoder(img, phase)
        adj_mat = self.nodes2adj(img, nodes.detach().clone(), nodes_per_image, device)
        if adj_mat.size(0) == 0:
            coarse_img = torch.zeros_like(coarse_img_t).to(device)
        else:
            coarse_img = self.g2itranslator(nodes.detach().clone(), adj_mat, nodes_per_image, device)
        out = self.decoder(coarse_img)
        out_t = self.decoder(coarse_img_t)

        return out_t, nodes_t, adj_mat_t, coarse_img_t, node_heat_t, \
               out, nodes, adj_mat, coarse_img, node_heat


def normalize_and_stack(node_heat_t):
    temp_batch_size, channels, h, w = node_heat_t.size()
    heat_maps_flat = node_heat_t.view(temp_batch_size, channels, h * w)
    max_value, max_idx = torch.max(heat_maps_flat, 2, keepdim=True)
    max_value = max_value.view(temp_batch_size, channels, 1, 1).expand(temp_batch_size, channels, h, w)
    normalized_heat_maps_t = node_heat_t / (max_value + 1e-6)
    stacked_heats_t = torch.sum(normalized_heat_maps_t, dim=1, keepdim=True)

    return stacked_heats_t
