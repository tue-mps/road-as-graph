import pandas as pd
import os
import cv2
import torch
import scipy
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class RoadLayoutDataset(Dataset):

    def __init__(self, csv_file, with_graph=False, transform=None, thinning=False):
        self.examples = pd.read_csv(csv_file, header=None)
        self.with_graph = with_graph
        self.transform = transform
        self.thinning = thinning

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        img = io.imread(self.examples.iloc[item, 0])
        max_value = np.max(img)
        if max_value > 0:
            img = img/np.max(img) # rescale range to [0, 1]

        if self.with_graph:
            graph = np.load(self.examples.iloc[item, 1], allow_pickle=True)[()]
            example = {'img': img,
                       'graph': graph
                       }
        else:
            example = {'img': img}

        if self.transform:
            example = self.transform(example)

        return example


class ToTensor(object):
    def __init__(self, with_graph=False):
        self.with_graph = with_graph

    def __call__(self, sample):
        img = sample['img']

        img = torch.from_numpy(img)
        if self.with_graph:
            graph = sample['graph']
            return {'img': img,
                    'graph': graph
                    }
        else:
            return {'img': img}


# class Rescale(object):
#
#     def __init__(self, output_size, with_graph=False):
#         self.output_size = output_size
#         self.with_graph = with_graph
#
#     def __call__(self, sample):
#         img = sample['img']
#         img = transform.resize(img, self.output_size, preserve_range=True, anti_aliasing=True)
#
#         if self.with_graph:
#             graph = sample['graph']
#             return {'img': img,
#                     'graph': graph
#                     }
#         else:
#             return {'img': img}


def collate_graphs(batch):
    imgs = [item['img'].unsqueeze(0) for item in batch]
    graphs = [item['graph'] for item in batch]
    return [imgs, graphs]


def collate_no_graphs(batch):
    imgs = [item['img'].unsqueeze(0) for item in batch]
    return [imgs]


if __name__ == '__main__':
    with_graph = False
    val_set = RoadLayoutDataset('datasets/road_layout/val.csv', with_graph=with_graph,
                            transform=transforms.Compose([ToTensor(with_graph=with_graph)]))
    print('number of val examples:', len(val_set))
    print(val_set[0]['img'].shape)
    if with_graph:
        print(val_set[0]['graph'])

    if with_graph:
        val_loader = DataLoader(val_set, batch_size=2, shuffle=True, num_workers=8, collate_fn=collate_graphs)
    else:
        val_loader = DataLoader(val_set, batch_size=2, shuffle=True, num_workers=8, collate_fn=collate_no_graphs)

    print('show 3 examples')
    for i, temp_batch in enumerate(val_loader):
        print(temp_batch[0])
        if with_graph:
            print(temp_batch[1])
        if i == 1:
            break

    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
    print('show 3 examples')
    for i, temp_batch in enumerate(val_loader):
        print(temp_batch['img'])
        if with_graph:
            print(temp_batch['graph'])
        if i == 1:
            break
