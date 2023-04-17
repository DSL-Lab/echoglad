from torch import optim
from copy import deepcopy

OPTIMIZERS = {
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
}

def build(optim_config, models, logger):
    config = deepcopy(optim_config)
    optimizer_name = config.pop('name')

    config['params'] = list(models['embedder'].parameters()) + list(models['landmark'].parameters())
    optimizer = OPTIMIZERS[optimizer_name](**config)

    logger.infov('{} opimizer is built.'.format(optimizer_name.upper()))
    return optimizer
