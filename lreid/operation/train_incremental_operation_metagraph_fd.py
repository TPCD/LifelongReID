import torch
from lreid.tools import MultiItemAverageMeter
from lreid.evaluation import accuracy

def train_incremental_metagraph_graphfd_no_detach_an_epoch(config, base, loader, current_step, old_model, old_graph_model, current_epoch=None, output_featuremaps=True):
    base.set_all_model_train()
    meter = MultiItemAverageMeter()
    print('****** train tasknet and metagraph ******\n')
    heatmaps_dict = {}
    ### we assume 200 iterations as an epoch
    for _ in range(config.steps):
        base.set_model_and_optimizer_zero_grad()
        ### load a batch data
        mini_batch = loader.continual_train_iter_dict[
            current_step].next_one()
        if mini_batch[0].size(0) != config.p * config.k:
            mini_batch = loader.continual_train_iter_dict[
                current_step].next_one()
        imgs, global_pids, global_cids, dataset_name, local_pids, image_paths = mini_batch

        if len(mini_batch) > 6:
            assert config.continual_step == 'task'
        imgs, local_pids, global_pids = imgs.to(base.device), local_pids.to(base.device), global_pids.to(base.device)
        # if we use low precision, input also need to be fp16
        if config.fp_16:
           imgs = imgs.half()
        loss = 0
        ### forward
        if old_model is None:
            features, cls_score, feature_maps = base.model_dict['tasknet'](imgs, current_step)
            protos, correlation = base.model_dict['metagraph'](features)

            feature_fuse = features + protos
            triplet_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse, local_pids, local_pids, local_pids)

            meter.update({
                'triplet_loss': triplet_loss.data,
                'show_correlation_meta': correlation[0].data,
                'show_correlation_transfered_meta': correlation[1].data,
                'show_correlation_transfered_proto': correlation[2].data
            })
            loss += triplet_loss
            del feature_maps, feature_fuse, features, protos
        else:
            old_current_step = list(range(current_step))
            new_current_step = list(range(current_step + 1))
            features, cls_score_list, feature_maps = base.model_dict['tasknet'](imgs, new_current_step)
            protos, correlation = base.model_dict['metagraph'](features)


            feature_fuse = features + protos
            triplet_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse,
                                                                    local_pids, local_pids, local_pids)

            cls_score = cls_score_list[-1]
            with torch.no_grad():
                old_features, old_cls_score_list, old_feature_maps = old_model(imgs, old_current_step)
                old_vertex = old_graph_model.meta_graph_vertex
            del old_features, old_feature_maps
            torch.cuda.empty_cache()
            new_logit = torch.cat(cls_score_list, dim=1)
            old_logit = torch.cat(old_cls_score_list, dim=1)

            knowladge_distilation_loss = config.weight_kd * base.loss_fn_kd(new_logit, old_logit, config.kd_T)

            fd_loss = config.weight_r * base.model_dict['metagraph'].MSE(old_vertex, base.model_dict['metagraph'].meta_graph_vertex)
            meter.update({
                'Kd_loss': knowladge_distilation_loss.data,
                'triplet_loss': triplet_loss.data,
                'fd_loss': fd_loss.data,
                'show_correlation_meta': correlation[0].data,
                'show_correlation_transfered_meta': correlation[1].data,
                'show_correlation_transfered_proto': correlation[2].data
            })
            loss += knowladge_distilation_loss + triplet_loss + fd_loss

        ### loss
        ide_loss = config.weight_x * base.ide_criterion(cls_score, local_pids)

        loss += ide_loss
        acc = accuracy(cls_score, local_pids, [1])[0]

        ### optimize
        base.optimizer_dict['tasknet'].zero_grad()
        base.optimizer_dict['metagraph'].zero_grad()
        if config.fp_16:  # we use optimier to backward loss
            with base.amp.scale_loss(loss, base.optimizer_list) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        base.optimizer_dict['tasknet'].step()
        base.optimizer_dict['metagraph'].step()
        ### recored
        meter.update({'ide_loss': ide_loss.data,
                      'acc': acc,
                      })
    if config.re_init_lr_scheduler_per_step:
        _lr_scheduler_step = current_epoch
    else:
        _lr_scheduler_step = current_step * config.total_train_epochs + current_epoch
    base.lr_scheduler_dict['tasknet'].step(_lr_scheduler_step)
    base.lr_scheduler_dict['metagraph'].step(_lr_scheduler_step)

    if output_featuremaps and not config.output_featuremaps_from_fixed:
        heatmaps_dict['feature_maps_true'] = base.featuremaps2heatmaps(imgs.detach().cpu(), feature_maps.detach().cpu(),
                                                                       image_paths,
                                                                       current_epoch,
                                                                       if_save=config.save_heatmaps,
                                                                       if_fixed=False,
                                                                       if_fake=False
                                                                       )
        return (meter.get_value_dict(), meter.get_str(), heatmaps_dict)
    else:
        return (meter.get_value_dict(), meter.get_str())


def train_incremental_metagraph_graphfd_an_epoch(config, base, loader, current_step, old_model, old_graph_model, current_epoch=None, output_featuremaps=True):
    base.set_all_model_train()
    meter = MultiItemAverageMeter()
    print('****** train tasknet and metagraph ******\n')
    heatmaps_dict = {}
    ### we assume 200 iterations as an epoch
    for _ in range(config.steps):
        base.set_model_and_optimizer_zero_grad()
        ### load a batch data
        mini_batch = loader.continual_train_iter_dict[
            current_step].next_one()
        if mini_batch[0].size(0) != config.p * config.k:
            mini_batch = loader.continual_train_iter_dict[
                current_step].next_one()
        imgs, global_pids, global_cids, dataset_name, local_pids, image_paths = mini_batch

        if len(mini_batch) > 6:
            assert config.continual_step == 'task'
        imgs, local_pids, global_pids = imgs.to(base.device), local_pids.to(base.device), global_pids.to(base.device)
        # if we use low precision, input also need to be fp16
        if config.fp_16:
           imgs = imgs.half()
        loss = 0
        ### forward
        if old_model is None:
            features, cls_score, feature_maps = base.model_dict['tasknet'](imgs, current_step)
            protos, correlation = base.model_dict['metagraph'](features.detach())

            feature_fuse = features + protos
            triplet_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse, local_pids, local_pids, local_pids)

            meter.update({
                'triplet_loss': triplet_loss.data,
                'show_correlation_meta': correlation[0].data,
                'show_correlation_transfered_meta': correlation[1].data,
                'show_correlation_transfered_proto': correlation[2].data
            })
            loss += triplet_loss
            del feature_maps, feature_fuse, features, protos
        else:
            old_current_step = list(range(current_step))
            new_current_step = list(range(current_step + 1))
            features, cls_score_list, feature_maps = base.model_dict['tasknet'](imgs, new_current_step)
            protos, correlation = base.model_dict['metagraph'](features.detach())


            feature_fuse = features + protos
            triplet_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse,
                                                                    local_pids, local_pids, local_pids)

            cls_score = cls_score_list[-1]
            with torch.no_grad():
                old_features, old_cls_score_list, old_feature_maps = old_model(imgs, old_current_step)
                old_vertex = old_graph_model.meta_graph_vertex
            del old_features, old_feature_maps
            torch.cuda.empty_cache()
            new_logit = torch.cat(cls_score_list, dim=1)
            old_logit = torch.cat(old_cls_score_list, dim=1)

            knowladge_distilation_loss = config.weight_kd * base.loss_fn_kd(new_logit, old_logit, config.kd_T)

            fd_loss = config.weight_r * base.model_dict['metagraph'].MSE(old_vertex, base.model_dict['metagraph'].meta_graph_vertex)
            meter.update({
                'Kd_loss': knowladge_distilation_loss.data,
                'triplet_loss': triplet_loss.data,
                'fd_loss': fd_loss.data,
                'show_correlation_meta': correlation[0].data,
                'show_correlation_transfered_meta': correlation[1].data,
                'show_correlation_transfered_proto': correlation[2].data
            })
            loss += knowladge_distilation_loss + triplet_loss + fd_loss

        ### loss
        ide_loss = config.weight_x * base.ide_criterion(cls_score, local_pids)

        loss += ide_loss
        acc = accuracy(cls_score, local_pids, [1])[0]

        ### optimize
        base.optimizer_dict['tasknet'].zero_grad()
        base.optimizer_dict['metagraph'].zero_grad()
        if config.fp_16:  # we use optimier to backward loss
            with base.amp.scale_loss(loss, base.optimizer_list) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        base.optimizer_dict['tasknet'].step()
        base.optimizer_dict['metagraph'].step()
        ### recored
        meter.update({'ide_loss': ide_loss.data,
                      'acc': acc,
                      })
    if config.re_init_lr_scheduler_per_step:
        _lr_scheduler_step = current_epoch
    else:
        _lr_scheduler_step = current_step * config.total_train_epochs + current_epoch
    base.lr_scheduler_dict['tasknet'].step(_lr_scheduler_step)
    base.lr_scheduler_dict['metagraph'].step(_lr_scheduler_step)

    if output_featuremaps and not config.output_featuremaps_from_fixed:
        heatmaps_dict['feature_maps_true'] = base.featuremaps2heatmaps(imgs.detach().cpu(), feature_maps.detach().cpu(),
                                                                       image_paths,
                                                                       current_epoch,
                                                                       if_save=config.save_heatmaps,
                                                                       if_fixed=False,
                                                                       if_fake=False
                                                                       )
        return (meter.get_value_dict(), meter.get_str(), heatmaps_dict)
    else:
        return (meter.get_value_dict(), meter.get_str())
