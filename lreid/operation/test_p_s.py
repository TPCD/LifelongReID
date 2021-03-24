import torch
from lreid.tools import time_now, CatMeter
from lreid.evaluation import (fast_evaluate_rank, compute_distance_matrix)


def fast_test_p_s(config, base, loaders, current_step, if_test_forget=True):
    # using Cython test during train
    # return mAP, Rank-1
    base.set_all_model_eval()
    print(f'****** start perform fast testing! ******')
    # meters

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
    results_dict = {}
    for dataset_name, temp_loaders in loaders.test_loader_dict.items():
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_metagraph_features_meter, query_metagraph_pids_meter, query_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_metagraph_features_meter, gallery_metagraph_pids_meter, gallery_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_fuse_features_meter, query_fuse_pids_meter, query_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_fuse_features_meter, gallery_fuse_pids_meter, gallery_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()

        print(time_now(), f' {dataset_name} feature start ')
        with torch.no_grad():
            for loader_id, loader in enumerate(temp_loaders):
                for data in loader:
                    # compute feautres
                    images, pids, cids = data[0:3]
                    images = images.to(base.device)
                    features, featuremaps = base.model_dict['tasknet'](images, current_step)
                    if config.if_test_metagraph:
                        features_metagraph, _ = base.model_dict['metagraph'](features)
                        features_fuse = features + features_metagraph
                        # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            query_fuse_features_meter.update(features_fuse.data)
                            query_metagraph_features_meter.update(features_metagraph.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            gallery_metagraph_features_meter.update(features_metagraph.data)
                            gallery_fuse_features_meter.update(features_fuse.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)

        # print(f'Save distance matrix to RegDB_three_stream_dist({current_step}).npy')
        #
        #
        #
        # np.save(os.path.join(config.feature_save_path, f'query_features_({dataset_name})_({current_step}).pth'),
        #         query_features_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'query_pids_({dataset_name})_({current_step}).pth'),
        #         query_pids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'query_cids_({dataset_name})_({current_step}).pth'),
        #         query_cids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_features_({dataset_name})_({current_step}).pth'),
        #         gallery_features_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_pids_({dataset_name})_({current_step}).pth'),
        #         gallery_pids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_cids_({dataset_name})_({current_step}).pth'),
        #         gallery_cids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'query_fuse_features_({dataset_name})_({current_step}).pth'),
        #         query_fuse_features_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_fuse_features_({dataset_name})_({current_step}).pth'),
        #         gallery_fuse_features_meter.get_val_numpy())

        print(time_now(), f' {dataset_name} feature done')
        rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
        results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
        if config.if_test_metagraph:
            # rank1, map = _cmc_map(query_metagraph_features_meter, gallery_metagraph_features_meter)
            # results_dict['metagraph_mAP'], results_dict['metagraph_Rank1'] = map, rank1
            rank1, map = _cmc_map(query_fuse_features_meter, gallery_fuse_features_meter)
            results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    return results_dict, results_str


def save_and_fast_test_p_s(config, base, loaders, current_step, current_epoch,if_test_forget=True):
    # using Cython test during train
    # return mAP, Rank-1
    base.set_all_model_eval()
    print(f'****** start perform fast testing! ******')

    # meters

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

    results_dict = {}
    for dataset_name, temp_loaders in loaders.test_loader_dict.items():
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_metagraph_features_meter, query_metagraph_pids_meter, query_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_metagraph_features_meter, gallery_metagraph_pids_meter, gallery_metagraph_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_fuse_features_meter, query_fuse_pids_meter, query_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_fuse_features_meter, gallery_fuse_pids_meter, gallery_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()

        print(time_now(), f' {dataset_name} feature start ')
        with torch.no_grad():
            for loader_id, loader in enumerate(temp_loaders):
                for data in loader:
                    # compute feautres
                    images, pids, cids = data[0:3]
                    images = images.to(base.device)
                    features, featuremaps = base.model_dict['tasknet'](images, current_step)
                    if config.if_test_metagraph:
                        features_metagraph, _ = base.model_dict['metagraph'](features)
                        features_fuse = features + features_metagraph
                        # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            query_fuse_features_meter.update(features_fuse.data)
                            query_metagraph_features_meter.update(features_metagraph.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        if config.if_test_metagraph:
                            gallery_metagraph_features_meter.update(features_metagraph.data)
                            gallery_fuse_features_meter.update(features_fuse.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)

        # print(f'Save ({dataset_name}) =>                                             feature ({current_step}).npy')
        #
        # np.save(os.path.join(config.feature_save_path, f'query_features_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         query_features_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'query_pids_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         query_pids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'query_cids_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         query_cids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_features_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         gallery_features_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_pids_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         gallery_pids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_cids_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         gallery_cids_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'query_fuse_features_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         query_fuse_features_meter.get_val_numpy())
        # np.save(os.path.join(config.feature_save_path, f'gallery_fuse_features_({dataset_name})_({current_step})_({current_epoch}).pth'),
        #         gallery_fuse_features_meter.get_val_numpy())

        print(time_now(), f' {dataset_name} feature done')
        rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
        results_dict[f'{dataset_name}_tasknet_mAP'], results_dict[f'{dataset_name}_tasknet_Rank1'] = map, rank1
        if config.if_test_metagraph:
            # rank1, map = _cmc_map(query_metagraph_features_meter, gallery_metagraph_features_meter)
            # results_dict['metagraph_mAP'], results_dict['metagraph_Rank1'] = map, rank1
            rank1, map = _cmc_map(query_fuse_features_meter, gallery_fuse_features_meter)
            results_dict[f'{dataset_name}_fuse_mAP'], results_dict[f'{dataset_name}_fuse_Rank1'] = map, rank1

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    return results_dict, results_str



