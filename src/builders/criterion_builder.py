import torch.nn as nn
from src.core.criterion import WeightedBCE, WeightedBCEWithLogitsLoss, MSE, ExpectedLandmarkMSE, HeatmapMSELoss, MAE
from copy import deepcopy


CRITERIA = {
    'mse': MSE,  # loss for coordinates
    'mae': MAE,  # loss for coordinates
    'bce': WeightedBCE,  # per-pixel loss for heatmaps
    'HeatmapMse': HeatmapMSELoss,  # per-pixel loss for heatmaps
    'WeightedBceWithLogits': WeightedBCEWithLogitsLoss,  # per-pixel loss for heatmaps
    'ExpectedLandmarkMse': ExpectedLandmarkMSE, # loss for coordinates using heatmaps
}


def build(config, logger):
    config = deepcopy(config)
    batch_size = config.pop('batch_size')
    frame_size = config.pop('frame_size')
    num_aux_graphs = config.pop('num_aux_graphs')
    use_main_graph_only = config.pop('use_main_graph_only')
    use_coordinate_graph = config.pop('use_coordinate_graph')
    num_output_channels = config.pop('num_output_channels')

    criteria = dict()
    for criterion_name, criterion_config in config.items():
        if criterion_name == 'ExpectedLandmarkMse':
            criteria[criterion_name] = CRITERIA[criterion_name](batch_size=batch_size,
                                                                frame_size=frame_size,
                                                                num_aux_graphs=num_aux_graphs,
                                                                use_main_graph_only=use_main_graph_only,
                                                                num_output_channels=num_output_channels,
                                                                **criterion_config)
        else:
            criteria[criterion_name] = CRITERIA[criterion_name](**criterion_config)

        logger.infov('{} criterion is built.'.format(criterion_name.upper()))

    # TODO: for now the coordinate loss doesn't need to be configurable
    if use_coordinate_graph:
        criteria['coordinate'] = CRITERIA['mae']()

    return criteria
