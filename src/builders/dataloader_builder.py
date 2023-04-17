from torch_geometric.loader import DataListLoader, DataLoader
from copy import deepcopy


def build(dataset, train_config, logger, use_data_parallel=False):
    # Get data parameters
    config = deepcopy(train_config)
    batch_size = config.pop('batch_size')
    num_workers = config.pop('num_workers')

    # Load datalodaers for each mode
    dataloaders = {}
    for mode in ['train', 'val', 'test']:
        shuffle = True if mode == 'train' else False
        drop_last = True if mode in ['train', 'val'] else False

        if use_data_parallel:
            dataloader = DataListLoader(dataset[mode],
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        drop_last=drop_last)
        else:
             dataloader = DataLoader(dataset[mode],
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     drop_last=drop_last)
        dataloaders[mode] = dataloader

    logger.infov("Dataloders are created.")

    return dataloaders

