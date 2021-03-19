import sys
sys.path.append('../')

from .dataset import *
from .loader import *
import torch
import torchvision.transforms as transforms
from lreid.data_loader.transforms2 import RandomErasing



class ReIDLoaders:

    def __init__(self, config):

        # resize --> flip --> pad+crop --> colorjitor(optional) --> totensor+norm --> rea (optional)
        transform_train = [
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size)]
        if config.use_colorjitor: # use colorjitor
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.use_rea: # use rea
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        # resize --> totensor --> norm
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['mix', 'market', 'duke']

        # dataset
        self.market_path = config.market_path
        self.duke_path = config.duke_path
        self.mix_path = config.mix_path
        self.combine_all = config.combine_all
        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset
        for a_train_dataset in self.train_dataset:
            assert a_train_dataset in self.datasets

        # batch size
        self.p = config.p
        self.k = config.k
        self.if_init_show_loader = config.output_featuremaps

        self.use_local_label4validation = config.use_local_label4validation
        if config.continual_step == '5':
            self.total_step = 5
        elif config.continual_step == '10':
            self.total_step = 10
        elif config.continual_step == 'task':
            self.total_step = 10
        elif config.continual_step == '1':
            self.total_step = 1
        else:
            print(config.continual_step)
            assert 0, 'error for config.continual_step'

        # load
        self._load(config)
        self._init_device()

        if config.continual_step == '5':
            self.continual_train_iter_dict = self.continual_5_train_iter_dict
            self.continual_local2global_dict = self.continual_5_local2global_dict
        elif config.continual_step == '10':
            self.continual_train_iter_dict = self.continual_10_train_iter_dict
            self.continual_local2global_dict = self.continual_10_local2global_dict
        elif config.continual_step == 'task':
            self.continual_train_iter_dict = self.continual_task_train_iter_dict
            self.continual_local2global_dict = self.continual_task_local2global_dict
        elif config.continual_step == '1':
            self.continual_train_iter_dict = self.continual_1_train_iter_dict
            self.continual_local2global_dict = self.continual_1_local2global_dict
        else:
            assert 0, 'error for config.continual_step'
        print(f'Show continual_train_iter_dict (size = {len(self.continual_train_iter_dict)}): \n {self.continual_train_iter_dict} \n--------end \n')
        print(f'Show continual_local2global_dict (size = {len(self.continual_local2global_dict)}): \n {self.continual_local2global_dict} \n--------end \n')

    def reversed_dict(self, global2local_dict):
        local2global_dict = defaultdict(dict)
        for step, global_local_dict in global2local_dict.items():
            for _global, _local in global_local_dict.items():
                local2global_dict[step].update({_local: _global})
        return local2global_dict

    def local2global(self, local_dict, concatenate=False):
        if concatenate is False:
            global_return = defaultdict(list)
            for key, _list in local_dict.items():
                for _local in _list:
                    global_return[key].append(self.continual_local2global_dict[int(key.split(':')[1])][_local.item()])
        else:
            global_return = []
            for key, _list in local_dict.items():
                for _local in _list:
                    global_return.append(self.continual_local2global_dict[int(key.split(':')[1])][_local.item()])
        return torch.tensor(global_return, dtype=torch.long)


    def _init_device(self):
        self.device = torch.device('cuda')

    def _load(self, config):

        '''init train dataset'''
        train_samples = self._get_train_samples(self.train_dataset)
        dataset_class = self._get_dataset_class(self.train_dataset)
        # list_task = self.generate_task_incremental_list(dataset_class)
        # np.save('smst.10', np.array(smst))
        # np.save('market.10', np.array(market))
        # np.save('duke.10', np.array(duke))
        # np.save('smst.10.dict', np.array(smst_dict))
        # np.save('market.10.dict', np.array(market_dict))
        # np.save('duke.10.dict', np.array(duke_dict))

        self.continual_1_num_pid_per_step = dataset_class.continual_1_num_pid_per_step
        self.continual_5_num_pid_per_step = dataset_class.continual_5_num_pid_per_step
        self.continual_10_num_pid_per_step = dataset_class.continual_10_num_pid_per_step
        self.continual_task_num_pid_per_step = dataset_class.continual_task_num_pid_per_step
        # train_samples.source_split_5_list
        self.continual_1_train_iter_dict = {}
        self.continual_10_train_iter_dict = {}
        self.continual_5_train_iter_dict = {}
        self.continual_task_train_iter_dict = {}
        self.continual_task_local2global_dict = self.reversed_dict(dataset_class.continual_task_global2local_dict)
        self.continual_5_local2global_dict = self.reversed_dict(dataset_class.continual_5_global2local_dict)
        self.continual_10_local2global_dict = self.reversed_dict(dataset_class.continual_10_global2local_dict)
        self.continual_1_local2global_dict = self.reversed_dict(dataset_class.continual_1_global2local_dict)
        for number, one_step_pid_list in enumerate(dataset_class.source_split_1_list):
            self.continual_1_train_iter_dict[number] = self._get_uniform_continual_iter(dataset_class.continual_1_train,
                                                                                        self.transform_train,
                                                                                        self.p, self.k, one_step_pid_list)

        for number, one_step_pid_list in enumerate(dataset_class.source_split_5_list):
            self.continual_5_train_iter_dict[number] = self._get_uniform_continual_iter(dataset_class.continual_5_train,
                                                                                        self.transform_train,
                                                                                        self.p, self.k, one_step_pid_list)

        for number, one_step_pid_list in enumerate(dataset_class.source_split_10_list):
            self.continual_10_train_iter_dict[number] = self._get_uniform_continual_iter(dataset_class.continual_10_train,
                                                                                         self.transform_train,
                                                                                         self.p, self.k, one_step_pid_list)

        for number, one_step_pid_list in enumerate(dataset_class.source_split_task_list):
            self.continual_task_train_iter_dict[number] = self._get_uniform_continual_iter(dataset_class.continual_task_train,
                                                                                         self.transform_train,
                                                                                         self.p, self.k, one_step_pid_list)
        # self.train_iter = self._get_uniform_iter(train_samples, self.transform_train, self.p, self.k)
        if self.if_init_show_loader:
            self.train_vae_iter = self._get_uniform_iter(train_samples, self.transform_test, 4, 2)
        '''init test dataset'''
        if self.test_dataset == 'market':
            self.market_query_samples, self.market_gallery_samples = self._get_test_samples('market')
            self.market_query_loader = self._get_loader(self.market_query_samples, self.transform_test, 128)
            self.market_gallery_loader = self._get_loader(self.market_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'duke':
            self.duke_query_samples, self.duke_gallery_samples = self._get_test_samples('duke')
            self.duke_query_loader = self._get_loader(self.duke_query_samples, self.transform_test, 128)
            self.duke_gallery_loader = self._get_loader(self.duke_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'mix':
            self.mix_query_samples, self.mix_gallery_samples, self.mix_validation_dict = self._get_test_samples('mix')
            self.mix_query_loader = self._get_loader(self.mix_query_samples, self.transform_test, 128)
            self.mix_gallery_loader = self._get_loader(self.mix_gallery_samples, self.transform_test, 128)
            if self.use_local_label4validation:
                if config.continual_step == '5':
                    self.mix_validation_query_loader = self._get_loader(
                        self.mix_validation_dict['validation_5_query_local_samples'], self.transform_test, 128)
                    self.mix_validation_gallery_loader = self._get_loader(
                        self.mix_validation_dict['validation_5_gallery_local_samples'], self.transform_test, 128)
                elif config.continual_step == '10':
                    self.mix_validation_query_loader = self._get_loader(
                        self.mix_validation_dict['validation_10_query_local_samples'], self.transform_test, 128)
                    self.mix_validation_gallery_loader = self._get_loader(
                        self.mix_validation_dict['validation_10_gallery_local_samples'], self.transform_test, 128)
                elif config.continual_step == 'task':
                    self.mix_validation_query_loader = self._get_loader(
                        self.mix_validation_dict['validation_task_query_local_samples'], self.transform_test, 128)
                    self.mix_validation_gallery_loader = self._get_loader(
                        self.mix_validation_dict['validation_task_gallery_local_samples'], self.transform_test, 128)

            else:
                if config.continual_step == '5':
                    self.mix_validation_query_loader = self._get_loader(
                        self.mix_validation_dict['validation_5_query_global_samples'], self.transform_test, 128)
                    self.mix_validation_gallery_loader = self._get_loader(
                        self.mix_validation_dict['validation_5_gallery_global_samples'], self.transform_test, 128)
                elif config.continual_step == '10':
                    self.mix_validation_query_loader = self._get_loader(
                        self.mix_validation_dict['validation_10_query_global_samples'], self.transform_test, 128)
                    self.mix_validation_gallery_loader = self._get_loader(
                        self.mix_validation_dict['validation_10_gallery_global_samples'], self.transform_test, 128)
                elif config.continual_step == 'task':
                    self.mix_validation_query_loader = self._get_loader(
                        self.mix_validation_dict['validation_task_query_global_samples'], self.transform_test, 128)
                    self.mix_validation_gallery_loader = self._get_loader(
                        self.mix_validation_dict['validation_task_gallery_global_samples'], self.transform_test, 128)

    def _get_train_samples(self, train_dataset):
        '''get train samples, support multi-dataset'''
        samples_list = []
        for a_train_dataset in train_dataset:
            if a_train_dataset == 'market':
                samples = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'duke':
                samples = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'mix':
                samples = Samples4MIX(self.mix_path, relabel=True, combineall=self.combine_all).train

            samples_list.append(samples)
        if len(train_dataset) > 1:
            samples = combine_samples(samples_list)
            samples = PersonReIDSamples._relabels(None, samples, 1)
            PersonReIDSamples._show_info(None, samples, samples, samples, name=str(train_dataset))
        return samples

    def _get_dataset_class(self, train_dataset):
        '''get train samples, support multi-dataset'''
        for a_train_dataset in train_dataset:
            if a_train_dataset == 'market':
                dataset_class = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all)
            elif a_train_dataset == 'duke':
                dataset_class = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all)
            elif a_train_dataset == 'mix':
                dataset_class = Samples4MIX(self.mix_path, relabel=True, combineall=self.combine_all)
        return dataset_class

    def _get_test_samples(self, test_dataset):
        if test_dataset == 'market':
            market = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all)
            query_samples = market.query
            gallery_samples = market.gallery
            return query_samples, gallery_samples
        elif test_dataset == 'duke':
            duke = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all)
            query_samples = duke.query
            gallery_samples = duke.gallery
            return query_samples, gallery_samples
        elif test_dataset == 'mix':
            mix = Samples4MIX(self.mix_path, relabel=True, combineall=self.combine_all)
            query_samples = mix.query
            gallery_samples = mix.gallery
            validation_dict = {'validation_5_query_global_samples': mix.validation_5_query_global,
                               'validation_5_gallery_global_samples': mix.validation_5_gallery_global,
                               'validation_5_query_local_samples': mix.validation_5_query_local,
                               'validation_5_gallery_local_samples': mix.validation_5_gallery_local,
                               'validation_10_query_global_samples': mix.validation_10_query_global,
                               'validation_10_gallery_global_samples': mix.validation_10_gallery_global,
                               'validation_10_query_local_samples': mix.validation_10_query_local,
                               'validation_10_gallery_local_samples': mix.validation_10_gallery_local,
                               'validation_task_query_global_samples': mix.validation_task_query_global,
                               'validation_task_gallery_global_samples': mix.validation_task_gallery_global,
                               'validation_task_query_local_samples': mix.validation_task_query_local,
                               'validation_task_gallery_local_samples': mix.validation_task_gallery_local
                               }
            return query_samples, gallery_samples, validation_dict



    def _get_uniform_continual_iter(self, samples, transform, p, k, pid_list):
        '''
               load person reid data_loader from images_folder
               and uniformly sample according to class for continual
               '''
        # dataset.sample is list  dataset.transform
        dataset = PersonReIDDataSet(samples, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False,
                                 sampler=ClassUniformlySampler4continual(dataset, class_position=1, k=k, pid_list=pid_list))
        iters = IterLoader(loader)
        return iters

    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        # dataset.sample is list  dataset.transform
        dataset = PersonReIDDataSet(samples, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters


    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        # dataset.sample is list  dataset.transform
        dataset = PersonReIDDataSet(samples, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters

    def _get_random_iter(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

