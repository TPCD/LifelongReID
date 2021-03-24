import torch
from lreid.tools import time_now, CatMeter
from lreid.evaluation import (ReIDEvaluator, PrecisionRecall, fast_evaluate_rank, compute_distance_matrix)
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def fast_test_continual_neck(config, base, loaders, current_step, if_test_forget=True):
    # using Cython test during train
    # return mAP, Rank-1
    base.set_all_model_eval()
    print(f'****** start perform fast testing! ******')
    # meters
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
    query_metagraph_features_meter, query_metagraph_pids_meter, query_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_metagraph_features_meter, gallery_metagraph_pids_meter, gallery_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
    # init dataset
    if config.test_dataset == 'market':
        loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
    elif config.test_dataset == 'duke':
        loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
    elif config.test_dataset == 'mix':
        if if_test_forget:
            loaders_validation = [loaders.mix_validation_query_loader, loaders.mix_validation_gallery_loader]
        loaders = [loaders.mix_query_loader, loaders.mix_gallery_loader]
    else:
        assert 0, 'test dataset error, expect mix/market/duke/, given {}'.format(config.test_dataset)

    print(time_now(), ' feature start ')

    # compute query and gallery features
    def _cmc_map(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()

        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=True)

        return CMC[0] * 100, mAP * 100

    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                # compute feautres
                images, pids, cids, _ = data
                images = images.to(base.device)
                features, featuremaps = base.model_dict['tasknet'](images, current_step)
                if config.if_test_metagraph:
                    features_metagraph, _ = base.model_dict['metagraph'](featuremaps=featuremaps, label=None, current_step=current_step)
                    # save as query features
                if loader_id == 0:
                    query_features_meter.update(features.data)
                    if config.if_test_metagraph:
                        query_metagraph_features_meter.update(features_metagraph.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features.data)
                    if config.if_test_metagraph:
                        gallery_metagraph_features_meter.update(features_metagraph.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)


    print(time_now(), 'feature done')
    results_dict = {}
    rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
    results_dict['tasknet_mAP'], results_dict['tasknet_Rank1'] = map, rank1
    if config.if_test_metagraph:
        rank1, map = _cmc_map(query_metagraph_features_meter, gallery_metagraph_features_meter)
        results_dict['metagraph_mAP'], results_dict['metagraph_Rank1'] = map, rank1

    if if_test_forget:
        # meters
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_metagraph_features_meter, query_metagraph_pids_meter, query_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_metagraph_features_meter, gallery_metagraph_pids_meter, gallery_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        print(time_now(), 'validation feature start')
        with torch.no_grad():
            for loader_id, loader in enumerate(loaders_validation):
                for data in loader:
                    # compute feautres
                    images, cids = data[0], data[2]
                    if config.use_local_label4validation:
                        pids = data[3]
                    else:
                        pids = data[1]

                    images = images.to(base.device)
                    features, featuremaps = base.model_dict['tasknet'](images, current_step)
                    if config.if_test_metagraph:
                        features_metagraph, _ = base.model_dict['metagraph'](featuremaps=featuremaps, label=None, current_step=current_step)
                    # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            query_metagraph_features_meter.update(features_metagraph.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            gallery_metagraph_features_meter.update(features_metagraph.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)
        print(time_now(), 'validation feature done')

        rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
        results_dict['tasknet_validation_mAP'], results_dict['tasknet_validation_Rank1'] = map, rank1

        if config.if_test_metagraph:
            rank1, map = _cmc_map(query_metagraph_features_meter, gallery_metagraph_features_meter)
            results_dict['metagraph_validation_mAP'], results_dict['metagraph_validation_Rank1'] = map, rank1

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    return results_dict, results_str




def test_continual_neck(config, base, loaders, current_step):

    base.set_all_model_eval()
    print(f'****** start perform full testing! ******')
    # meters
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    # init dataset
    if config.test_dataset == 'market':
        loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
    elif config.test_dataset == 'duke':
        loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
    elif config.test_dataset == 'mix':
        loaders = [loaders.mix_query_loader, loaders.mix_gallery_loader]
    else:
        assert 0, 'test dataset error, expect mix/market/duke/, given {}'.format(config.test_dataset)

    print(time_now(), 'feature start')

    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                # compute feautres
                images, pids, cids, _ = data
                images = images.to(base.device)
                features, _ = base.model_dict['tasknet'](images, current_step)
                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)

    print(time_now(), 'feature done')

    #
    query_features = query_features_meter.get_val_numpy()
    gallery_features = gallery_features_meter.get_val_numpy()

    # compute mAP and rank@k
    mAP, CMC = ReIDEvaluator(dist=config.test_metric, mode=config.test_mode).evaluate(
        query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
        gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy())

    # compute precision-recall curve
    thresholds = np.linspace(1.0, 0.0, num=101)
    pres, recalls, thresholds = PrecisionRecall(dist=config.test_metric, mode=config.test_mode).evaluate(
        thresholds, query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
        gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy())

    return mAP, CMC[0: 150], pres, recalls, thresholds


def plot_prerecall_curve(config, pres, recalls, thresholds, mAP, CMC, label, current_step):

    plt.plot(recalls, pres, label='{model},map:{map},cmc135:{cmc}'.format(
        model=label, map=round(mAP, 2), cmc=[round(CMC[0], 2), round(CMC[2], 2), round(CMC[4], 2)]))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.output_path, f'precisio-recall-curve-{current_step}.png'))



def output_featuremaps_from_fixed(base, current_epoch):
    base.set_all_model_train()
    print(f'****** start perform generation! ******')
    with torch.no_grad():
        features, cls_score, feature_maps = base.model_dict['tasknet'](base.fixed_images)
        fake_feature_maps, means, variances, current_sampled_z = base.model_dict['vae'](feature_maps)
        visdom_tensor_true = base.featuremaps2heatmaps(base.fixed_images,
                                                       feature_maps,
                                                       base.fixed_paths,
                                                       current_epoch,
                                                       if_save=True,
                                                       if_fixed=True,
                                                       if_fake=False)
        visdom_tensor_fake = base.featuremaps2heatmaps(base.fixed_images,
                                                       fake_feature_maps,
                                                       base.fixed_paths,
                                                       current_epoch,
                                                       if_save=True,
                                                       if_fixed=True,
                                                       if_fake=True)
    print(f'****** end perform generation! ******')
    return visdom_tensor_true.detach().cpu(), visdom_tensor_fake.detach().cpu()
