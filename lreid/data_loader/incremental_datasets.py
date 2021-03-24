import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable
from easydict import EasyDict
import random
from collections import defaultdict, OrderedDict
import operator



def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class IncrementalPersonReIDSamples:

    def _relabels_incremental(self, samples, label_index, is_mix=False):
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
            sample = list(sample)
            pid2label[sample[label_index]] = ids.index(sample[label_index])
        new_samples = copy.deepcopy(samples)
        for i, sample in enumerate(samples):
            new_samples[i] = list(new_samples[i])
            new_samples[i][label_index] = pid2label[sample[label_index]]
        if is_mix:
            return samples, pid2label
        else:
            return new_samples

    def _load_images_path(self, folder_dir, domain_name='market', is_mix=False):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name, is_mix=is_mix)
                samples.append([root_path + file_name, identi_id, camera_id, domain_name])
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

    def _show_info(self, train, query, gallery, name=None, if_show=True):
        if if_show:
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
        else:
            pass




def Incremental_combine_test_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''

    all_gallery, all_query = [], []

    def _generate_relabel_dict(s_list):
        pids_in_list, pid2relabel_dict = [], {}
        for new_label, samples in enumerate(s_list):
            if str(samples[1]) + str(samples[3]) not in pids_in_list:
                pids_in_list.append(str(samples[1]) + str(samples[3]))
        for i, pid in enumerate(sorted(pids_in_list)):
            pid2relabel_dict[pid] = i
        return pid2relabel_dict
    def _replace_pid2relabel(s_list, pid2relabel_dict, pid_dimension=1):
        new_list = copy.deepcopy(s_list)
        for i, sample in enumerate(s_list):
            new_list[i] = list(new_list[i])
            new_list[i][pid_dimension] = pid2relabel_dict[str(sample[pid_dimension])+str(sample[pid_dimension + 2])]
        return new_list

    for samples_class in samples_list:
        all_gallery.extend(samples_class.gallery)
        all_query.extend(samples_class.query)
    pid2relabel_dict = _generate_relabel_dict(all_gallery)
    # pid2relabel_dict2 = _generate_relabel_dict(all_query)

    # assert len(list(pid2relabel_dict2.keys())) == sum([1 for query_key in pid2relabel_dict2.keys() if query_key in pid2relabel_dict.keys()])
    #print(pid2relabel_dict)
    #print(pid2relabel_dict2)
    # assert operator.eq(pid2relabel_dict, _generate_relabel_dict(all_query))
    gallery = _replace_pid2relabel(all_gallery, pid2relabel_dict, pid_dimension=1)
    query = _replace_pid2relabel(all_query, pid2relabel_dict, pid_dimension=1)


    return query, gallery

def Incremental_combine_train_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples, new_samples = [], []
    all_pid_per_step, all_cid_per_step, output_all_per_step = OrderedDict(), OrderedDict(), defaultdict(dict)
    max_pid, max_cid = 0, 0
    for step, samples in enumerate(samples_list):
        for a_sample in samples:
            img_path = a_sample[0]
            local_pid = a_sample[1]
            try:
                dataset_name = a_sample[3]
                global_pid = max_pid + a_sample[1]
                # global_cid = str(dataset_name) + ':' + str(a_sample[2])
                global_cid = max_cid + int(a_sample[2])
            except:
                print(a_sample)
                assert False
            all_samples.append([img_path, global_pid, global_cid, dataset_name, local_pid])
            if step in all_pid_per_step.keys():
                all_pid_per_step[step].add(global_pid)
            else:
                all_pid_per_step[step] = set()
                all_pid_per_step[step].add(global_pid)

            if step in all_cid_per_step.keys():
                all_cid_per_step[step].add(global_cid)
            else:
                all_cid_per_step[step] = set()
                all_cid_per_step[step].add(global_cid)
        for k, v in all_cid_per_step.items():
            all_cid_per_step[k] = sorted(v)
        max_pid = sum([len(v) for v in all_pid_per_step.values()])
        max_cid = sum([len(v) for v in all_cid_per_step.values()])

    return all_samples, all_pid_per_step, all_cid_per_step


class IncrementalReIDDataSet:
    def __init__(self, samples, total_step, transform):
        self.samples = samples
        self.transform = transform
        self.total_step = total_step

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])
        this_sample = list(this_sample)
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
