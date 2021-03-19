import ast
from lreid.tools import time_now
from lreid.core import Base_metagraph_fd_incremental
from lreid.data_loader import IncrementalReIDLoaders
from lreid.visualization import visualize, Logger, VisdomPlotLogger, VisdomFeatureMapsLogger
from lreid.operation import (train_incremental_metagraph_graphfd_an_epoch, fast_test_incremental_metagraph_graphfd,
                                 test_continual_neck,
                                plot_prerecall_curve, output_featuremaps_from_fixed)

def main(config):

    # init loaders and base
    loaders = IncrementalReIDLoaders(config)
    base = Base_metagraph_fd_incremental(config, loaders)

    # init logger
    logger = Logger(os.path.join(base.output_dirs_dict['logs'], 'log.txt'))
    logger(config)
    if config.visualize_train_by_visdom:
        port = 8097
        visdom_dict = {'feature_maps_fake': VisdomFeatureMapsLogger('image', pad_value=1, nrow=8, port=port, env=config.running_time, opts={'title': f'featuremaps fake'}),
                       'feature_maps_true': VisdomFeatureMapsLogger('image', pad_value=1, nrow=8, port=port, env=config.running_time, opts={'title': f'featuremaps true'}),
                       'feature_maps': VisdomFeatureMapsLogger('image', pad_value=1, nrow=8, port=port, env=config.running_time, opts={'title': f'featuremaps'})}


    assert config.mode in ['train', 'test', 'visualize']
    if config.mode == 'train':  # train mode
        # automatically resume model from the latest one
        if config.auto_resume_training_from_lastest_steps:
            start_train_step, start_train_epoch = base.resume_last_model()
        # continual loop
        for current_step in range(start_train_step, loaders.total_step):
            current_total_train_epochs = config.total_continual_train_epochs if current_step > 0 else config.total_train_epochs
            if current_step > 0:
                logger(f'save_and_frozen old model in {current_step}')
                old_model = base.copy_model_and_frozen(model_name='tasknet')
                old_graph_model = base.copy_model_and_frozen(model_name='metagraph')
            else:
                old_model = None
                old_graph_model = None
            for current_epoch in range(start_train_epoch, current_total_train_epochs):
                visdom_result_dict = {}
                # save model
                base.save_model(current_step, current_epoch)
                # train
                str_lr, dict_lr = base.get_current_learning_rate()
                logger(str_lr)
                if current_epoch < config.epoch_start_joint:
                    results = train_incremental_metagraph_graphfd_an_epoch(config, base, loaders, current_step, old_model,old_graph_model, current_epoch, output_featuremaps=config.output_featuremaps)

                if config.output_featuremaps:
                    results_dict, results_str, heatmaps = results
                    if config.output_featuremaps_from_fixed:
                        heatmaps_true, heatmaps_fake = output_featuremaps_from_fixed(base, current_epoch)
                        visdom_dict['feature_maps_fake'].images(heatmaps_fake)
                        visdom_dict['feature_maps_true'].images(heatmaps_true)
                    else:
                        visdom_dict['feature_maps'].images(heatmaps)
                else:
                    results_dict, results_str = results
                logger('Time: {};  Step: {}; Epoch: {};  {}'.format(time_now(), current_step, current_epoch, results_str))

                if config.test_frequency > 0 and current_epoch % config.test_frequency == 0:
                    rank_map_dict, rank_map_str = fast_test_incremental_metagraph_graphfd(config, base, loaders, current_step, if_test_forget=config.if_test_forget)
                    logger(
                        f'Time: {time_now()}; Test Dataset: {config.test_dataset}: {rank_map_str}')
                    visdom_result_dict.update(rank_map_dict)


                if current_epoch == config.total_train_epochs - 1:
                    rank_map_dict, rank_map_str = fast_test_incremental_metagraph_graphfd(config, base, loaders, current_step, if_test_forget=config.if_test_forget)
                    logger(
                        f'Time: {time_now()}; Step: {current_step}; Epoch: {current_epoch} Test Dataset: {config.test_dataset}, {rank_map_str}')
                    # plot_prerecall_curve(config, pres, recalls, thresholds, mAP, CMC, 'none', current_step)
                    print(f'Current step {current_step} is finished.')
                    start_train_epoch = 0
                    visdom_result_dict.update(rank_map_dict)

                if config.visualize_train_by_visdom:
                    visdom_result_dict.update(results_dict)
                    visdom_result_dict.update(dict_lr)
                    if current_step > 0:
                        global_current_epoch = current_epoch + (current_step-1) * current_total_train_epochs + config.total_train_epochs
                    else:
                        global_current_epoch = current_epoch
                    for name, value in visdom_result_dict.items():
                        if name in visdom_dict.keys():
                            visdom_dict[name].log(global_current_epoch, value, name=str(current_step))
                        else:
                            visdom_dict[name] = VisdomPlotLogger('line', port=port, env=config.running_time,
                                                                 opts={'title': f'train {name}'})
                            visdom_dict[name].log(global_current_epoch, value, name=str(current_step))

            if current_step > 0:
                del old_model

    elif config.mode == 'test':	# test mode
        base.resume_from_model(config.resume_test_model)
        mAP, CMC, pres, recalls, thresholds = test_continual_neck(config, base, loaders, 0)
        logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, mAP, CMC))
        logger('Time: {}; Test Dataset: {}, \nprecision: {} \nrecall: {}\nthresholds: {}'.format(
            time_now(), config.test_dataset, mAP, CMC, pres, recalls, thresholds))
        plot_prerecall_curve(config, pres, recalls, thresholds, mAP, CMC, 'none')


    elif config.mode == 'visualize': # visualization mode
        base.resume_from_model(config.resume_visualize_model)
        visualize(config, base, loaders)


if __name__ == '__main__':
    import time
    import argparse
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    running_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    # cfg.data.save_dir = osp.join(cfg.data.save_dir, running_time)

    parser = argparse.ArgumentParser()

    parser.add_argument('--fp_16', type=bool, default=False)
    parser.add_argument('--running_time', type=str, default=running_time)
    parser.add_argument('--visualize_train_by_visdom', type=bool, default=True)
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='trian_10, train_5, train, test or visualize')
    parser.add_argument('--output_path', type=str, default=f'results/{running_time}', help='path to save related informations')
    parser.add_argument('--continual_step', type=str, default='5',
                        help='10 or 5 or task')
    parser.add_argument('--num_identities_per_domain', type=int, default=500,
                        help='250 for 10 steps, 500 for 5 steps, -1 for all aviliable identities')
    parser.add_argument('--joint_train', type=bool, default=False,
                        help='joint all dataset')
    parser.add_argument('--re_init_lr_scheduler_per_step', type=bool, default=False,
                        help='after_previous_step if re_init_optimizers')
    parser.add_argument('--warmup_lr', type=bool, default=False,
                        help='0-10 epoch warmup')


    # dataset configuration
    machine_dataset_path = '/home/prometheus/Experiments/Datasets'
    # machine_dataset_path = '/home/r2d2/r2d2/Datasets/'
    parser.add_argument('--datasets_root', type=str, default=machine_dataset_path, help='mix/market/duke/')
    parser.add_argument('--combine_all', type=ast.literal_eval, default=False, help='train+query+gallery as train')
    parser.add_argument('--train_dataset', nargs='+', type=str,
                        default=['market','subcuhksysu','duke','msmt17','cuhk03'])
    parser.add_argument('--test_dataset', nargs='+', type=str,
                        default=['duke','market','cuhk03','allgeneralizable','cuhk01','cuhk02','viper','ilids','prid','grid','sensereid'])

    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parser.add_argument('--p', type=int, default=16, help='person count in a batch')
    parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')
    parser.add_argument('--use_local_label4validation', type=bool, default=True,
                        help='validation use global pid label or not')

    # data augmentation
    parser.add_argument('--use_rea', type=ast.literal_eval, default=True)
    parser.add_argument('--use_colorjitor', type=ast.literal_eval, default=False)

    # model configuration
    parser.add_argument('--cnnbackbone', type=str, default='res50', help='res50, res50ibna')
    parser.add_argument('--pid_num', type=int, default=2494, help='for CRL mix dataset:2494(combineall-2494 + 4512)market:751(combineall-1503), duke:702(1812), msmt:1041(3060), njust:spr3869(5086),win,both(7729)')

    # train configuration
    parser.add_argument('--steps', type=int, default=150, help='150 for 5s32p4k, 75 for 10s32p4k')
    parser.add_argument('--task_milestones', nargs='+', type=int, default=[25],
                        help='task_milestones for the task learning rate decay')
    parser.add_argument('--task_gamma', type=float, default=0.1,
                        help='task_gamma for the task learning rate decay')
    parser.add_argument('--new_module_milestones', nargs='+', type=int, default=[50,100,150,200],
                        help='milestones for the VAE learning rate decay')
    parser.add_argument('--new_module_gamma', type=float, default=0.5,
                        help='vae_gamma for the VAE learning rate decay')

    parser.add_argument('--task_base_learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--new_module_learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--total_train_epochs', type=int, default=50)
    parser.add_argument('--total_continual_train_epochs', type=int, default=50)


    parser.add_argument('--epoch_start_joint', type=int, default=80, help='start epoch for start joint sampled')

    # resume and save

    parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
    parser.add_argument('--max_save_model_num', type=int, default=2, help='0 for max num is infinit')
    parser.add_argument('--resume_train_dir', type=str, default='',
                        help='****************************************************************@@@@@@@@@@@@')
    parser.add_argument('--fast_test', type=bool,
                        default=True,
                        help='test during train using Cython')

    parser.add_argument('--test_frequency', type=int,
                        default=11,
                        help='test during trai, i <= 0 means do not test during train')
    parser.add_argument('--if_test_forget', type=bool,
                        default=True,
                        help='test during train for forgeting')
    parser.add_argument('--if_test_metagraph', type=bool,
                        default=True,
                        help='test during train for forgeting')

    # test configuration
    parser.add_argument('--resume_test_model', type=str, default='/path/to/pretrained/model.pkl', help='')
    parser.add_argument('--test_mode', type=str, default='all', help='inter-camera, intra-camera, all')
    parser.add_argument('--test_metric', type=str, default='cosine', help='cosine, euclidean')

    # visualization configuration
    parser.add_argument('--resume_visualize_model', type=str, default='/path/to/pretrained/model.pkl',
                        help='only availiable under visualize model')
    parser.add_argument('--visualize_dataset', type=str, default='',
                        help='market, duke, only  only availiable under visualize model')
    parser.add_argument('--visualize_mode', type=str, default='inter-camera',
                        help='inter-camera, intra-camera, all, only availiable under visualize model')
    parser.add_argument('--visualize_mode_onlyshow', type=str, default='pos', help='pos, neg, none')
    parser.add_argument('--visualize_output_path', type=str, default='results/visualization/',
                        help='path to save visualization results, only availiable under visualize model')
    parser.add_argument('--output_featuremaps', type=bool, default=False,
                        help='During training visualize featuremaps')
    parser.add_argument('--save_heatmaps', type=bool, default=False,
                        help='During training visualize featuremaps and save')
    parser.add_argument('--output_featuremaps_from_fixed', type=bool, default=False,
                        help='alternative from fixed or training sample')


    # losses configuration
    parser.add_argument('--weight_x', type=float, default=1, help='weight for cross entropy loss')
    # for graph

    parser.add_argument('--meta_graph_vertex_num', type=int, default=64,
                        help='meta_graph_vertex_num')
    parser.add_argument('--weight_r', type=float, default=0.03, help='weight for fd loss')


    # for embed net
    parser.add_argument('--weight_t', type=float, default=1, help='weight for triplet loss')
    parser.add_argument('--t_margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    parser.add_argument('--t_metric', type=str, default='euclidean', help='euclidean, cosine')
    parser.add_argument('--t_l2', type=bool, default=False, help='if l2 normal for the triplet loss with batch hard')

    # for classifier disstilation

    parser.add_argument('--weight_kd', type=float, default=1, help='weight for cross entropy loss')
    parser.add_argument('--kd_T', type=float, default=2, help='weight for cross entropy loss')




    # main
    config = parser.parse_args()
    main(config)



