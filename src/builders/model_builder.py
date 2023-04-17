from src.core.models import HierarchicalPatchModel, CNN, CNNHierarchicalPatchModel,\
    UNETHierarchicalPatchModel, IdenticalModel, UNETIntermediateNoGnn, UNET
from copy import deepcopy


MODELS = {
    'cnn': CNN,
    'identical': IdenticalModel,
    'hierarchicalpatch': HierarchicalPatchModel,
    'cnn_hierarchical_patch': CNNHierarchicalPatchModel,
    'unet_hierarchical_patch': UNETHierarchicalPatchModel,
    'unet_noGNN': UNETIntermediateNoGnn,
    'unet': UNET
}


def build(model_config, logger):

    config = deepcopy(model_config)
    _ = config.pop('checkpoint_path')

    # Build a model
    model = dict()
    for model_key in config:
        model_name = config[model_key].pop('name')
        model[model_key] = MODELS[model_name](**config[model_key])

    logger.infov("Model is created.")

    return model
