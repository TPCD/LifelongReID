from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
from lreid.data.datasets import ImageDataset


class IncrementalSamples4mix(IncrementalPersonReIDSamples):
    """mix datase  for CRL

    Dataset statistics:
    The mixed dataset (CRL-person) contains 2,494 training identities.
    Specifically, the three datasets contribute 751, 702 and 1,041 training identities respectively.
    In total, 59,706 training images of the 2,494 identities are employed as the training set.
    The training set is split into 5 subsets and 10 subsets for 5-step and 10-step continual representation learning respectively.
    After applying the two strategies, the final testing set has 11,351 query images and 19,576 gallery images of 4,512 identities.
        - train_identities: 2494
        - test_identities; 4512
        - images: 59706 (train) + 11351 (query) + 19576 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'mix'

    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.dataset_dir = osp.join(datasets_root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MIX')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "train" under '
                '"MIX".'
            )

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        self.source_txt = osp.join(self.data_dir, 'source.txt')
        self.split_5_txt = osp.join(self.data_dir, 'split_5_step.txt')
        self.split_10_txt = osp.join(self.data_dir, 'split_10_step.txt')


        train, pid2label = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        self.source_split_5_list = self.read_split_txt(self.split_5_txt)
        self.source_split_10_list = self.read_split_txt(self.split_10_txt)
        self.source_txt_list = self.read_source_txt(self.source_txt)

        self.source_split_5_list = self.relabel(self.source_split_5_list, pid2label)
        self.source_split_10_list = self.relabel(self.source_split_10_list, pid2label)

        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)


    def process_dir(self, dir_path, relabel=False):
        # 0000_0_00_000 pid_(train=0, query=1, gallery=2)_camid_index
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_(\d)_([\d]+)_([\d]+)')

        pid_container = set()

        for img_path in img_paths:
            # pid, _,  = map(int, pattern.search(img_path).groups())
            pid, state, camid, index = pattern.search(img_path).groups()
            pid_container.add(int(pid))
            # data.append((img_path, pid, camid, 'mix', pid))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, state, camid, index = pattern.search(img_path).groups()
            if relabel:
                pid = pid2label[int(pid)]
            data.append([img_path, int(pid), int(camid), 'mix', int(pid)])

        if relabel:
            return data, pid2label
        else:
            return data


    def read_split_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            obj = f.readlines()
        pid_list = [list(map(int,step_item.split(':')[1].replace('[', '').replace(']', '').replace(' ', '').split(','))) for step_item in obj]
        return pid_list


    def read_source_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            obj = f.readlines()
        dict = { line.split('..')[0].replace(' ', ''): line.split('..')[1].replace(' ', '').replace('\n', '') for line in obj }
        return dict


    def relabel(self, split_list, pid2label_dict):
        new_list = []
        for step_temp in split_list:
            new_list_step = []
            for original_pid in step_temp:
                new_list_step.append(pid2label_dict[int(original_pid)])
            new_list.append(new_list_step)
        return new_list


class MIX(ImageDataset):
    """mix datase  for CRL

    Dataset statistics:
    The mixed dataset (CRL-person) contains 2,494 training identities.
    Specifically, the three datasets contribute 751, 702 and 1,041 training identities respectively.
    In total, 59,706 training images of the 2,494 identities are employed as the training set.
    The training set is split into 5 subsets and 10 subsets for 5-step and 10-step continual representation learning respectively.
    After applying the two strategies, the final testing set has 11,351 query images and 19,576 gallery images of 4,512 identities.
        - train_identities: 2494
        - test_identities; 4512
        - images: 59706 (train) + 11351 (query) + 19576 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'mix'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MIX')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "train" under '
                '"MIX".'
            )

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        self.source_txt = osp.join(self.data_dir, 'source.txt')
        self.split_5_txt = osp.join(self.data_dir, 'split_5_step.txt')
        self.split_10_txt = osp.join(self.data_dir, 'split_10_step.txt')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir,
            self.source_txt, self.split_5_txt, self.split_10_txt
        ]

        self.check_before_run(required_files)

        train, pid2label = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        self.source_split_5_list = self.read_split_txt(self.split_5_txt)
        self.source_split_10_list = self.read_split_txt(self.split_10_txt)
        self.source_txt_list = self.read_source_txt(self.source_txt)

        self.source_split_5_list = self.relabel(self.source_split_5_list, pid2label)
        self.source_split_10_list = self.relabel(self.source_split_10_list, pid2label)

        super(MIX, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        # 0000_0_00_000 pid_(train=0, query=1, gallery=2)_camid_index
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_(\d)_([\d]+)_([\d]+)')

        pid_container = set()

        for img_path in img_paths:
            # pid, _,  = map(int, pattern.search(img_path).groups())
            pid, state, camid, index = pattern.search(img_path).groups()
            pid_container.add(int(pid))
            # data.append((img_path, pid, camid, 'mix', pid))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, state, camid, index = pattern.search(img_path).groups()
            if relabel:
                pid = pid2label[int(pid)]
            data.append((img_path, pid, camid, 'mix', pid))

        if relabel:
            return data, pid2label
        else:
            return data


    def read_split_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            obj = f.readlines()
        pid_list = [list(map(int,step_item.split(':')[1].replace('[', '').replace(']', '').replace(' ', '').split(','))) for step_item in obj]
        return pid_list


    def read_source_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            obj = f.readlines()
        dict = { line.split('..')[0].replace(' ', ''): line.split('..')[1].replace(' ', '').replace('\n', '') for line in obj }
        return dict


    def relabel(self, split_list, pid2label_dict):
        new_list = []
        for step_temp in split_list:
            new_list_step = []
            for original_pid in step_temp:
                new_list_step.append(pid2label_dict[int(original_pid)])
            new_list.append(new_list_step)
        return new_list