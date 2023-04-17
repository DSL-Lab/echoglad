import os
from src.engine import Engine
from src.utils.util import updated_config, load_log, mkdir_p
import yaml


if __name__ == '__main__':
    # ############# handling the input arguments and yaml configuration file ###############
    config = updated_config()

    save_dir = os.path.join(config['save_dir'])
    mkdir_p(save_dir)
    logger = load_log(save_dir)

    with open(os.path.join(save_dir, "config.yml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    engine = Engine(config=config, logger=logger, save_dir=save_dir)

    if config['eval_only']:
        engine.evaluate(data_type=config['eval_data_type'])
    else:
        engine.run()
