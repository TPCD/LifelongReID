import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable
from easydict import EasyDict
import random
from collections import defaultdict
def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class PersonReIDSamples:

    def _relabels(self, samples, label_index, is_mix=False):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        pid2label = {}
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()

        # reorder
        for sample in samples:
            pid2label[sample[label_index]] = ids.index(sample[label_index])
            sample[label_index] = ids.index(sample[label_index])
        if is_mix:
            return samples, pid2label
        else:
            return samples

    def _load_images_path(self, folder_dir, is_mix=False):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name, is_mix=is_mix)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name, is_mix=False):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''

        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        if is_mix:
            identi_id, camera_id = int(split_list[0]), int(split_list[2])
        else:
            identi_id, camera_id = int(split_list[0]), int(split_list[1])

        return identi_id, camera_id

    def _show_info(self, train, query, gallery, name=None):
        def analyze(samples):
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        train_info = analyze(train)
        query_info = analyze(query)
        gallery_info = analyze(gallery)

        # please kindly install prettytable: ```pip install prettyrable```
        table = PrettyTable(['set', 'images', 'identities', 'cameras'])
        table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
        table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
        table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
        table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
        print(table)


class Samples4MIX(PersonReIDSamples):
    '''
    Market Dataset
    '''
    def __init__(self, datasets_root, relabel=True, combineall=False):

        # parameters
        self.mix_path = os.path.join(datasets_root, 'mix/MIX/')
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.mix_path, 'train/')
        query_path = os.path.join(self.mix_path, 'query/')
        gallery_path = os.path.join(self.mix_path, 'gallery/')
        self.source_txt = os.path.join(self.mix_path, 'source.txt')
        self.split_5_txt = os.path.join(self.mix_path, 'split_5_step.txt')
        self.split_10_txt = os.path.join(self.mix_path, 'split_10_step.txt')

        # load
        train = self._load_images_path(train_path, is_mix=True)
        query = self._load_images_path(query_path, is_mix=True)
        gallery = self._load_images_path(gallery_path, is_mix=True)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)

        # reorder person identities
        if self.relabel:
            train, pid2label = self._relabels(train, 1, is_mix=True)

        self.source_split_5_list = self.read_split_txt(self.split_5_txt)
        self.source_split_10_list = self.read_split_txt(self.split_10_txt)
        self.source_split_1_list = [[]]
        for _list in self.source_split_10_list:
            self.source_split_1_list[0] += _list

        self.source_txt_list = self.read_source_txt(self.source_txt)

        self.source_split_5_list = self.relabel4txt(self.source_split_5_list, pid2label)
        self.source_split_10_list = self.relabel4txt(self.source_split_10_list, pid2label)
        self.source_split_1_list = self.relabel4txt(self.source_split_1_list, pid2label)

        self.continual_5_pids_list, self.continual_5_global2local_dict, self.continual_5_num_pid_per_step = self.relabel4every_step(self.source_split_5_list)

        self.continual_10_pids_list, self.continual_10_global2local_dict, self.continual_10_num_pid_per_step = self.relabel4every_step(self.source_split_10_list)
        self.continual_1_pids_list, self.continual_1_global2local_dict, self.continual_1_num_pid_per_step = self.relabel4every_step(
            self.source_split_1_list)

        # self.check_pids_in_list_and_source_txt(train)

        # add global pid
        self.continual_5_train = self.add_global_pid_and_step(train, self.continual_5_global2local_dict)
        self.continual_10_train = self.add_global_pid_and_step(train, self.continual_10_global2local_dict)
        self.continual_1_train = self.add_global_pid_and_step(train, self.continual_1_global2local_dict)
        # add for task incremental
        self.source_split_task_list = self.generate_task_incremental_list()
        self.continual_task_pids_list, self.continual_task_global2local_dict, self.continual_task_num_pid_per_step = self.relabel4every_step(
            self.source_split_task_list)
        self.continual_task_train = self.add_global_pid_and_step(train, self.continual_task_global2local_dict)

        self.create_validation_set_for_forget()

        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def generate_task_incremental_list(self):

        def _show_info(pid_list_dict):
            info_array = []
            info_list_dict = {}
            print(f'The number of ids: {len(pid_list_dict.keys())}\n')
            for index, l in enumerate(pid_list_dict.values()):
                info_list_dict[index] = len(l)
                info_array.append(len(l))
            return info_list_dict, info_array

        msmt_pid_list_dict = defaultdict(list)
        market_pid_list_dict = defaultdict(list)
        duke_pid_list_dict = defaultdict(list)
        for item in self.continual_10_train:
            source_path_list = self.source_txt_list[item[0].split('/')[-1]].split('/')
            if 'MSMT' in source_path_list[2]:
                item = item[:3]
                item.append('msmt')
                msmt_pid_list_dict[item[1]].append(item)
            elif 'Market' in source_path_list[2]:
                item = item[:3]
                item.append('market')
                market_pid_list_dict[item[1]].append(item)
            elif 'Duke' in source_path_list[2]:
                item = item[:3]
                item.append('duke')
                duke_pid_list_dict[item[1]].append(item)

        list_step_0_3 = []
        msmt_part = sorted(list(msmt_pid_list_dict.keys()))
        market_part = sorted(list(market_pid_list_dict.keys()))
        duke_part = sorted(list(duke_pid_list_dict.keys()))
        for i in range(4):
            list_step_0_3.append(msmt_part[i * 250 : (i+1) * 250])
        list_step_4_6 = []
        for i in range(3):
            list_step_4_6.append(market_part[i * 250: (i + 1) * 250])
        list_step_7_8 = []
        for i in range(2):
            list_step_7_8.append(duke_part[i * 250: (i + 1) * 250])
        list_step_9 = []
        list_step_9.extend(msmt_part[1000:])
        list_step_9.extend(market_part[750:])
        list_step_9.extend(duke_part[500:])
        list_task = list_step_0_3 + list_step_4_6 + list_step_7_8 + [list_step_9]

        return list_task


    def create_validation_set_for_forget(self):

        def _generate_validation_query_gallery(continual_train_list):
            global_pid_image_dict = defaultdict(list)
            local_pid_image_dict = defaultdict(list)
            for path, global_pid, camid, local_pid, step_id in continual_train_list:
                if step_id == 0:
                    # validation_set_list.append([path, global_pid, camid, local_pid, step_id])
                    global_pid_image_dict[global_pid].append([path, global_pid, camid, local_pid, step_id])
                    local_pid_image_dict[local_pid].append([path, global_pid, camid, local_pid, step_id])
            validation_query_global = [global_pid_image_dict[g_p].pop() for g_p in global_pid_image_dict.keys()]
            validation_gallery_global = []
            for item in global_pid_image_dict.values():
                validation_gallery_global.extend(item)
            validation_query_local = [local_pid_image_dict[g_p].pop() for g_p in local_pid_image_dict.keys()]
            validation_gallery_local = []
            for item in local_pid_image_dict.values():
                validation_gallery_local.extend(item)
            return validation_query_global, validation_gallery_global, validation_query_local, validation_gallery_local

        self.validation_5_query_global, self.validation_5_gallery_global, self.validation_5_query_local, self.validation_5_gallery_local = _generate_validation_query_gallery(
            self.continual_5_train)
        self.validation_10_query_global, self.validation_10_gallery_global, self.validation_10_query_local, self.validation_10_gallery_local = _generate_validation_query_gallery(
            self.continual_10_train)
        self.validation_task_query_global, self.validation_task_gallery_global, self.validation_task_query_local, self.validation_task_gallery_local = _generate_validation_query_gallery(
            self.continual_task_train)


    def relabel4txt(self, split_list, pid2label_dict):
        new_list = []
        for step_temp in split_list:
            new_list_step = []
            for original_pid in step_temp:
                new_list_step.append(pid2label_dict[int(original_pid)])
            new_list.append(new_list_step)
        return new_list

    def relabel4every_step(self, total_list):
        global2local_dict = defaultdict(dict)
        _total_list = copy.deepcopy(total_list)
        for step, step_list in enumerate(total_list):

            for i, p in enumerate(step_list):
                global2local_dict[step].update({p: i})
                _total_list[step][i] = i

        return _total_list, global2local_dict, [len(i) for i in _total_list]

    def add_global_pid_and_step(self, train, global2local_dict):
        _train = []
        for path, global_pid, camid in train:
            for _step, _dict in global2local_dict.items():
                if global_pid in _dict.keys():
                    _train.append([path, global_pid, camid, _dict[global_pid], _step])
                    break;
        assert len(_train) == len(train)
        return _train


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

    def check_pids_in_list_and_source_txt(self, train):
        tatol_list = []
        pid_set = set()
        for step_list in self.source_split_5_list:
            tatol_list.extend(step_list)
        for _, pid, _ in train:
            pid_set.add(pid)
            assert pid in tatol_list
        for ii in tatol_list:
            assert ii in pid_set

        tatol_list = []
        pid_set = set()
        for step_list in self.source_split_10_list:
            tatol_list.extend(step_list)
        for _, pid, _ in train:
            pid_set.add(pid)
            assert pid in tatol_list
        for ii in tatol_list:
            assert ii in pid_set


class Samples4Market(PersonReIDSamples):
    '''
    Market Dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'bounding_box_train/')
        query_path = os.path.join(self.market_path, 'query/')
        gallery_path = os.path.join(self.market_path, 'bounding_box_test/')

        # load
        train = self._load_images_path(train_path)
        query = self._load_images_path(query_path)
        gallery = self._load_images_path(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)

        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

class Samples4Duke(PersonReIDSamples):
    '''
    Duke dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'bounding_box_train/')
        query_path = os.path.join(self.market_path, 'query/')
        gallery_path = os.path.join(self.market_path, 'bounding_box_test/')

        # load
        train = self._load_images_path(train_path)
        query = self._load_images_path(query_path)
        gallery = self._load_images_path(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)
        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id



def combine_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples = []
    max_pid, max_cid = 0, 0
    for samples in samples_list:
        for a_sample in samples:
            img_path = a_sample[0]
            pid = max_pid + a_sample[1]
            cid = max_cid + a_sample[2]
            all_samples.append([img_path, pid, cid])
        max_pid = max([sample[1] for sample in all_samples])
        max_cid = max([sample[2] for sample in all_samples])
    return all_samples





class PersonReIDDataSet:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])
        this_sample.append(this_sample[0])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')




class ContinualReIDDataSet:
    def __init__(self, samples, total_step, transform):
        self.samples = samples
        self.transform = transform
        self.total_step = total_step

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])
        this_sample.append(this_sample[0])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
