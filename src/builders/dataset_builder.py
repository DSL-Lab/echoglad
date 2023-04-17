from torchvision import transforms
from src.core.datasets import LVLandmark, DummyDataset, UICLVLandmark, EchoNetLandmark
from src.utils.util import normalization_params
from copy import deepcopy


DATASETS = {
    'lvlandmark': LVLandmark,
    'uiclvlandmark': UICLVLandmark,
    'dummy': DummyDataset,
    'echonet': EchoNetLandmark
}


def build(data_config, logger):
    # Get data parameters
    config = deepcopy(data_config)
    data_name = config.pop('name')
    transform_config = config.pop('transform')

    if data_name not in DATASETS:
        logger.error('No data named {}'.format(data_name))

    # Create the datasets for each mode
    datasets = {}
    for mode in ['train', 'val', 'test']:
        transform = compose_transforms(transform_config, mode)
        datasets[mode] = DATASETS[data_name](**config,
                                             mode=mode,
                                             logger=logger,
                                             transform=transform,
                                             frame_size=transform_config['image_size'])

    return datasets


def compose_transforms(transform_config, mode):
    mean, std = normalization_params()
    image_size = transform_config['image_size']
    make_gray = transform_config.get('make_gray', False)

    transforms_list = [transforms.Resize((image_size, image_size))]

    if make_gray:
        transforms_list.append(transforms.Grayscale())

    transform = transforms.Compose(transforms_list)
    return transform
