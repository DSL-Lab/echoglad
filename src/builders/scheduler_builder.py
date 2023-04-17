from torch import optim
from torch.optim.optimizer import Optimizer
from src.core.schedulers import CustomScheduler
from copy import deepcopy


SCHEDULERS = {
    'multi': optim.lr_scheduler.MultiStepLR,
    'reduce_lr_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    'custom': CustomScheduler
}

def build(train_config, optimizer, logger):
    if 'lr_schedule' not in train_config:
        logger.warn('No scheduler is specified.')
        return None

    schedule_config = deepcopy(train_config['lr_schedule'])
    scheduler_name = schedule_config.pop('name', 'multi')
    schedule_config['optimizer'] = optimizer

    if scheduler_name in SCHEDULERS:
        scheduler = SCHEDULERS[scheduler_name](**schedule_config)
    else:
        logger.error(
            'Specify a valid scheduler name among {}.'.format(SCHEDULERS.keys())
        ); exit()

    logger.infov('{} scheduler is built.'.format(scheduler_name.upper()))
    return scheduler
