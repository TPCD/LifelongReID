from PIL import Image
import os.path as osp
import _pickle as cPickle
import json
import errno
import numpy as np
import torch
import os
import pickle as pk

def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pk.dump(data, f, pk.HIGHEST_PROTOCOL)



def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pk.load(f)
    return data


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    _obj = obj.copy()
    for k, v in _obj.items():
        if isinstance(v, np.ndarray):
            _obj.pop(k)
    with open(fpath, 'w') as f:
        json.dump(_obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, filename):
    torch.save(state, filename)

class PersonSearchDataset(object):

    def __init__(self, root, transforms, mode='train'):
        super(PersonSearchDataset, self).__init__()
        self.root = root
        self.data_path = self.get_data_path()
        self.transforms = transforms
        self.mode = mode
        # test = gallery + probe
        assert self.mode in ('train', 'test', 'probe')

        self.imgs = self._load_image_set_index()
        if self.mode in ('train', 'test'):
            self.record = self.gt_roidb()
        else:
            self.record = self.load_probes()

    def get_data_path(self):
        raise NotImplementedError

    def _load_image_set_index(self):
        raise NotImplementedError

    def gt_roidb(self):
        raise NotImplementedError

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, idx):
        # since T.to_tensor() is in transforms, the returned image pixel range is in (0, 1)
        # label_pids.min() = 1 if mode = 'train', else label_pids unchanged.
        sample = self.record[idx]
        im_name = sample['im_name']
        img_path = osp.join(self.data_path, im_name)
        img = Image.open(img_path).convert('RGB')

        boxes = torch.as_tensor(sample['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(sample['gt_pids'], dtype=torch.int64)

        target = dict(boxes=boxes,
                      labels=labels,
                      flipped='False',
                      im_name=im_name
                      )

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.mode == 'train':
            target['labels'] = \
                self._adapt_pid_to_cls(target['labels'])

        return img, target

    def __len__(self):
        return len(self.record)

    def _adapt_pid_to_cls(self, label_pids, upid=5555):
        raise NotImplementedError

    def load_probes(self):
        raise NotImplementedError
