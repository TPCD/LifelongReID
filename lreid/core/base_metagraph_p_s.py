import sys
sys.path.append('..')
import copy
import torch
import torch.optim as optim
import os
import os.path as osp
import numpy as np
from collections import OrderedDict
from torch.nn import functional as F
from lreid.models import LwFNet, MetaGraph_fd
from lreid.losses import CrossEntropyLabelSmooth, PlasticityLoss
from lreid.tools import os_walk, make_dirs
from .lr_schedulers import WarmupMultiStepLR, torch16_MultiStepLR
import cv2
from torch.autograd import Variable
import torch.nn as nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10

#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


class Base_metagraph_p_s(object):
    '''
    a base module includes model, optimizer, loss, and save/resume operations.
    '''

    def __init__(self, config, loader):

        self.config = config
        self.loader = loader
        # Model Config
        self.mode = config.mode
        self.cnnbackbone = config.cnnbackbone
        self.pid_num = config.pid_num
        self.t_margin = config.t_margin
        # Logger Configuration
        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.output_dirs_dict = {'logs': os.path.join(self.output_path, 'logs/'),
                                 'models': os.path.join(self.output_path, 'models/'),
                                 'images': os.path.join(self.output_path, 'images/'),
                                 'features': os.path.join(self.output_path, 'features/')}

        # make directions
        for current_dir in self.output_dirs_dict.values():
            make_dirs(current_dir)
        # Train Configuration


        # resume_train_dir
        self.resume_train_dir = config.resume_train_dir

        # init
        self._init_device()
        self._init_model()
        self._init_criterion()
        self._init_optimizer()
        if config.output_featuremaps:
            self._init_fixed_values()
        if config.fp_16:
            self.amp = amp
            self.model_list = list(self.model_dict.values())
            self.optimizer_list = list(self.optimizer_dict.values())
            self.model_list, self.optimizer_list = self.amp.initialize(self.model_list, self.optimizer_list, opt_level="O1")
        else:
            self.amp = None
            self.model_list = None
            self.optimizer_list = None

    def _init_device(self):
        self.device = torch.device('cuda')


    def _init_model(self):
        pretrained = False if self.mode != 'train' else True

        self.model_dict = torch.nn.ModuleDict()

        num_class_list = self.loader.continual_num_pid_per_step


        self.model_dict['tasknet'] = LwFNet(class_num_list=num_class_list,
                                            pretrained=pretrained)

        self.model_dict['metagraph'] = MetaGraph_fd(hidden_dim=self.model_dict['tasknet'].feature_dim,
                                                 input_dim=self.model_dict['tasknet'].feature_dim,
                                                 sigma=2.0,
                                                 proto_graph_vertex_num=self.config.p,
                                                 meta_graph_vertex_num=self.config.meta_graph_vertex_num)


        for name, module in self.model_dict.items():
            if 'net' in name:
                # summary(module, torch.zeros((2, 3, self.config.image_size[0], self.config.image_size[1])))
                module = module.to(self.device)
            elif 'graph' in name:
                # summary(module, torch.zeros((self.config.p*self.config.k, 2048)))
                module = module.to(self.device)
            else:
                # summary(module, torch.zeros((2, 512, 16, 8)))
                module = module.to(self.device)


    def _init_criterion(self):
        self.ide_criterion = CrossEntropyLabelSmooth(self.pid_num)
        self.triplet_criterion = PlasticityLoss(self.t_margin, self.config.t_metric, self.config.t_l2)
        self.reconstruction_criterion = torch.nn.L1Loss()

    def _init_optimizer(self):
        self.lr_scheduler_dict = {}
        self.optimizer_dict = {}
        for name, module in self.model_dict.items():
            if 'net' in name:
                self.optimizer_dict[name] = optim.Adam(module.parameters(), lr=self.config.task_base_learning_rate,
                                                       weight_decay=self.config.weight_decay)
                if self.config.warmup_lr:
                    self.lr_scheduler_dict[name] = WarmupMultiStepLR(self.optimizer_dict[name],
                                                                     self.config.task_milestones,
                                                                     gamma=self.config.task_gamma,
                                                                     warmup_factor=0.01,
                                                                     warmup_iters=10)

                else:
                    self.lr_scheduler_dict[name] = torch16_MultiStepLR(self.optimizer_dict[name],
                                                                       self.config.task_milestones,
                                                                       gamma=self.config.task_gamma)

            else:
                self.optimizer_dict[name] = optim.Adam(module.parameters(), lr=self.config.new_module_learning_rate,
                                                       weight_decay=self.config.weight_decay)
                if self.config.warmup_lr:
                    self.lr_scheduler_dict[name] = WarmupMultiStepLR(self.optimizer_dict[name],
                                                                     self.config.new_module_milestones,
                                                                     gamma=self.config.new_module_gamma,
                                                                     warmup_factor=0.01,
                                                                     warmup_iters=10)
                else:
                    self.lr_scheduler_dict[name] = torch16_MultiStepLR(self.optimizer_dict[name],
                                                                       self.config.new_module_milestones,
                                                                       gamma=self.config.new_module_gamma)


    def save_model(self, save_step, save_epoch):
        '''save model as save_epoch'''
        # save model
        models_steps_path = os.path.join(self.output_dirs_dict['models'], str(save_step))
        if not osp.exists(models_steps_path):
            make_dirs(models_steps_path)
        for module_name, module in self.model_dict.items():
            torch.save(module.state_dict(),
                       os.path.join(models_steps_path, f'model_{module_name}_{save_epoch}.pkl'))
        for optimizer_name, optimizer in self.optimizer_dict.items():
            torch.save(optimizer.state_dict(),
                       os.path.join(models_steps_path, f'optimizer_{optimizer_name}_{save_epoch}.pkl'))
        if self.config.fp_16 and self.amp:
            torch.save(self.amp.state_dict(),
                       os.path.join(models_steps_path, f'amp_{save_epoch}.pkl'))


        # if saved model is more than max num, delete the model with smallest epoch
        if self.max_save_model_num > 0:
            root, _, files = os_walk(models_steps_path)

            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

            # remove the bad-case and get available indexes
            model_num = len(self.model_dict)
            optimizer_num = len(self.optimizer_dict)
            available_indexes = copy.deepcopy(indexes)
            for element in indexes:
                if indexes.count(element) < model_num + optimizer_num:
                    available_indexes.remove(element)

            available_indexes = sorted(list(set(available_indexes)), reverse=True)
            unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

            # delete all unavailable models
            for unavailable_index in unavailable_indexes:
                try:
                    # os.system('find . -name "{}*_{}.pth" | xargs rm  -rf'.format(self.config.save_models_path, unavailable_index))
                    for module_name in self.model_dict.keys():
                        os.remove(os.path.join(root, f'model_{module_name}_{unavailable_index}.pkl'))
                    for optimizer_name in self.optimizer_dict.keys():
                        os.remove(os.path.join(root, f'optimizer_{optimizer_name}_{unavailable_index}.pkl'))
                    if self.config.fp_16 and self.amp:
                        os.remove(os.path.join(root, f'amp_{unavailable_index}.pkl'))
                except:
                    pass

            # delete extra models
            if len(available_indexes) >= self.max_save_model_num:
                for extra_available_index in available_indexes[self.max_save_model_num:]:
                    # os.system('find . -name "{}*_{}.pth" | xargs rm  -rf'.format(self.config.save_models_path, extra_available_index))
                    for mudule_name, mudule in self.model_dict.items():
                        os.remove(os.path.join(root, f'model_{mudule_name}_{extra_available_index}.pkl'))
                    for optimizer_name, optimizer in self.optimizer_dict.items():
                        os.remove(os.path.join(root, f'optimizer_{optimizer_name}_{extra_available_index}.pkl'))
                    if self.config.fp_16 and self.amp:
                        os.remove(os.path.join(root, f'amp_{extra_available_index}.pkl'))

    def resume_last_model(self):
        '''resume model from the last one in path self.output_path'''
        # find all files in format of *.pkl

        if self.resume_train_dir == '':
            root, dir, files = os_walk(self.output_dirs_dict['models'])
        else:
            root, dir, files = os_walk(os.path.join(self.resume_train_dir, 'models'))
        if len(dir) > 0:
            resume_step = max(dir)
        else:
            return 0, 0
        _, _, files = os_walk(os.path.join(root, resume_step))
        for file in files:
            if '.pkl' not in file:
                files.remove(file)
        # find the last one
        if len(files) > 0:
            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            # resume model from the latest model
            self.resume_model(resume_step, indexes[-1])
            #
            start_train_epoch = indexes[-1]
            start_train_step = resume_step
            return int(start_train_step), start_train_epoch
        else:
            return 0, 0


    def resume_model(self, resume_step, resume_epoch):
        '''resume model from resume_epoch'''
        if self.resume_train_dir == '':
            model_path = os.path.join(self.output_dirs_dict['models'], resume_step,
                                      f'amp_{resume_epoch}.pkl')
        else:
            model_path = os.path.join(self.resume_train_dir, 'models', resume_step,
                                      f'amp_{resume_epoch}.pkl')
        try:
            self.amp.load_state_dict(torch.load(model_path))
        except:
            print(('fail resume amp from {}'.format(model_path)))
            pass
        else:
            print(('successfully resume amp from {}'.format(model_path)))
        for module_name, module in self.model_dict.items():
            if self.resume_train_dir == '':
                model_path = os.path.join(self.output_dirs_dict['models'], resume_step, f'model_{module_name}_{resume_epoch}.pkl')
            else:
                model_path = os.path.join(self.resume_train_dir, 'models', resume_step, f'model_{module_name}_{resume_epoch}.pkl')
            try:
                module.load_state_dict(torch.load(model_path), strict=False)
            except:
                print(('fail resume model from {}'.format(model_path)))
                pass
            else:
                print(('successfully resume model from {}'.format(model_path)))

        for optimizer_name, optimizer in self.optimizer_dict.items():
            if self.resume_train_dir == '':
                model_path = os.path.join(self.output_dirs_dict['models'], resume_step, f'optimizer_{optimizer_name}_{resume_epoch}.pkl')
            else:
                model_path = os.path.join(self.resume_train_dir, 'models', resume_step, f'optimizer_{optimizer_name}_{resume_epoch}.pkl')
            try:
                optimizer.load_state_dict(torch.load(model_path))
            except:
                print(('fail resume optimizer from {}'.format(model_path)))
                pass
            else:
                print(('successfully resume optimizer from {}'.format(model_path)))



    def resume_from_model(self, models_dir):
        '''resume from model. model_path shoule be like /path/to/model.pkl'''
        # self.model.load_state_dict(torch.load(model_path), strict=False)
        # print(('successfully resume model from {}'.format(model_path)))
        '''resume model from resume_epoch'''
        for module_name, module in self.model_dict.items():
            model_path = os.path.join(models_dir, 'models', f'model_{module_name}.pkl')
            state_dict = torch.load(model_path)
            model_dict = module.state_dict()
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # discard module.
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    matched_layers.append(k)
                else:
                    discarded_layers.append(k)
            model_dict.update(new_state_dict)
            module.load_state_dict(model_dict)
            if len(discarded_layers) > 0:
                print('discarded layers: {}'.format(discarded_layers))

            print(('successfully resume model from {}'.format(model_path)))

    ## set model as train mode
    def set_all_model_train(self):
        for name, module in self.model_dict.items():
            module = module.train()
            module.training = True

    ## set model as eval mode
    def set_all_model_eval(self):
        for name, module in self.model_dict.items():
            module = module.eval()
            module.training = False

    def set_specific_models_train(self, models_list):
        copy_list = copy.deepcopy(list(self.model_dict.keys()))
        print(f'****** open following modules for training! ******')
        for specific_name in models_list:
            if specific_name in copy_list:
                self.model_dict[specific_name] = self.model_dict[specific_name].train()
                self.model_dict[specific_name].training = True
                copy_list.remove(specific_name)
                print(f'open < {specific_name} > modules !')
        print(f'**************************************************\n')
        print(f'****** close the other modules for training! ******')
        for non_specific_name in copy_list:
            self.model_dict[non_specific_name] = self.model_dict[non_specific_name].eval()
            self.model_dict[non_specific_name].training = False
            print(f'close < {non_specific_name} > modules !')
        print(f'**************************************************\n')

    def close_all_layers(self, model):
        r"""Opens all layers in model for training.

        Examples::
            >>> from torchreid.utils import open_all_layers
            >>> open_all_layers(model)
        """
        model.train()
        for p in model.parameters():
            p.requires_grad = False

    def open_specified_layers(self, model, open_layers):
        r"""Opens specified layers in model for training while keeping
        other layers frozen.

        Args:
            model (nn.Module): neural net model.
            open_layers (str or list): layers open for training.

        Examples::
            >>> from torchreid.utils import open_specified_layers
            >>> # Only model.classifier will be updated.
            >>> open_layers = 'classifier'
            >>> open_specified_layers(model, open_layers)
            >>> # Only model.fc and model.classifier will be updated.
            >>> open_layers = ['fc', 'classifier']
            >>> open_specified_layers(model, open_layers)
        """
        if isinstance(model, nn.DataParallel):
            model = model.module

        if isinstance(open_layers, str):
            open_layers = [open_layers]

        for layer in open_layers:
            assert hasattr(
                model, layer
            ), '"{}" is not an attribute of the model, please provide the correct name'.format(
                layer
            )

        for name, module in model.named_children():
            if name in open_layers:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    ## set model as eval mode
    def close_specific_layers(self, model_name, layers_list):
        if isinstance(self.model_dict[model_name], nn.DataParallel):
            model = self.model_dict[model_name].module
        else:
            model = self.model_dict[model_name]
        if isinstance(layers_list, str):
            layers_list = [layers_list]

        for layer in layers_list:
            assert hasattr(
                model, layer
            ), '"{}" is not an attribute of the model, please provide the correct name'.format(
                layer
            )
        for name, module in model.named_children():
            if name in layers_list:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
                print(f'****** close {name} layers and set it as eval mode! ******')


    def set_specific_models_eval(self, models_list):
        copy_list = copy.deepcopy(list(self.model_dict.keys()))
        print(f'****** close following modules for testing! ******')
        for specific_name in models_list:
            if specific_name in copy_list:
                self.model_dict[specific_name] = self.model_dict[specific_name].eval()
                self.model_dict[specific_name].training = False
                copy_list.remove(specific_name)
                print(f'close < {specific_name} > modules !')
        print(f'**************************************************\n')

        print(f'****** open the other modules for testing! ******')
        for non_specific_name in copy_list:
            self.model_dict[non_specific_name] = self.model_dict[non_specific_name].train()
            self.model_dict[non_specific_name].training = True
            print(f'close < {specific_name} > modules !')
        print(f'**************************************************\n')

    def set_bn_to_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()

    def set_bn_to_train(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.train()

    def set_model_and_optimizer_zero_grad(self, mode=['model', 'optimizer']):
        if 'model' in mode:
            for name, module in self.model_dict.items():
                module.zero_grad()
        if 'optimizer' in mode:
            for name, optimizer in self.optimizer_dict.items():
                optimizer.zero_grad()

    def make_onehot(self, label):
        onehot_vec = torch.zeros(label.size()[0], self.config.class_num)
        for i in range(label.size()[0]):
            onehot_vec[i, label[i]] = 1
        return onehot_vec

    def get_current_learning_rate(self):
        str_output = 'current learning rate: '
        dict_output = {}
        for name, optim in self.optimizer_dict.items():
            str_output += f" <{name}> = <{optim.param_groups[0]['lr']}>; "
            dict_output[name] = optim.param_groups[0]['lr']

        return str_output + f'\n', dict_output

    def featuremaps2heatmaps(self, original_images, featuremaps, image_paths, current_epoch, if_save=False, if_fixed=False, if_fake=False):
        height = original_images.size(2)
        width = original_images.size(3)
        imgs = original_images
        outputs = featuremaps.sum(1)
        # outputs = (outputs ** 2).sum(1)
        #b, h, w = outputs.size()
        #outputs = outputs.view(b, h * w)
        # outputs = F.normalize(outputs, p=2, dim=1)
        #outputs = outputs.view(b, h, w)

        imgs, outputs = imgs.cpu(), outputs.cpu()
        grid_img_tensor = []
        if if_save:
            save_dir = osp.join(self.output_dirs_dict['images'], str(current_epoch))
            make_dirs(save_dir)
            if if_fixed:
                if if_fake:
                    save_dir = osp.join(self.output_dirs_dict['images'], str(current_epoch), 'fake')
                    make_dirs(save_dir)
                else:
                    save_dir = osp.join(self.output_dirs_dict['images'], str(current_epoch), 'true')
                    make_dirs(save_dir)
            else:
                save_dir = osp.join(self.output_dirs_dict['images'], str(current_epoch))
        for j in range(outputs.size(0)):
            # get image name
            path = image_paths[j]
            imname = osp.basename(osp.splitext(path)[0])

            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, IMAGENET_MEAN, IMAGENET_STD):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)


            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8
            )
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:,
            width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            grid_img_tensor.append(grid_img)
            if if_save:
                cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)
        grid_img_tensor = np.transpose(np.stack(grid_img_tensor, axis=0), (0, 3, 1, 2))
        return torch.from_numpy(grid_img_tensor)

    def _init_fixed_values(self):  # for visualization
        self.fixed_images, self.fixed_ids, self.fixed_camids, self.fixed_paths = self.loader.train_vae_iter.next_one()
        self.fixed_images = self.fixed_images.to(self.device)
        self.fixed_ids = self.fixed_ids.to(self.device)

    def copy_model_and_frozen(self, model_name='tasknet'):
        old_model = copy.deepcopy(self.model_dict[model_name])
        old_model = old_model.to(self.device)
        return old_model.train()


    def MultiClassCrossEntropy(self, new_cls_score_list, old_cls_score_list, T):
        # new_logit = torch.cat(cls_score_list, dim=1)
        # old_logit = torch.cat(old_cls_score_list, dim=1)
        # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
        assert len(new_cls_score_list) == len(old_cls_score_list)
        loss = 0.
        n = len(new_cls_score_list)
        for logits, labels in zip(new_cls_score_list, old_cls_score_list):
            labels = Variable(labels.data, requires_grad=False).to(self.device)
            outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
            labels = torch.softmax(labels / T, dim=1)
            # print('outputs: ', outputs)
            # print('labels: ', labels.shape)
            outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
            loss += -torch.mean(outputs, dim=0, keepdim=False)
        # print('OUT: ', outputs)
        loss = Variable(loss.data, requires_grad=True).to(self.device) / n
        return loss

    def loss_fn_kd(self, scores, target_scores, T=2.):
        """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

        Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
        'Hyperparameter': temperature"""

        device = scores.device

        log_scores_norm = F.log_softmax(scores / T, dim=1)
        targets_norm = F.softmax(target_scores / T, dim=1)

        # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
        n = scores.size(1)
        if n > target_scores.size(1):
            n_batch = scores.size(0)
            zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
            zeros_to_add = zeros_to_add.to(device)
            targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        KD_loss_unnorm = -(targets_norm * log_scores_norm)
        KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

        # normalize
        KD_loss = KD_loss_unnorm * T ** 2

        return KD_loss




    def loss_fn_fkd(self, new_feature, old_feature, l2_normal=True):
        """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

        Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
        'Hyperparameter': temperature"""
        assert old_feature.size(0) == new_feature.size(0)
        n = new_feature.size(0)
        # if l2_normal:
        #     old_feature = F.normalize(old_feature, dim=1)
        #     new_feature = F.normalize(new_feature, dim=1)

        old_mm = F.softmax(old_feature.mm(old_feature.t()), dim=1)
        new_mm = F.log_softmax(new_feature.mm(new_feature.t()), dim=1)

        KD_loss = F.kl_div(new_mm, old_mm, reduction='mean')


        return KD_loss