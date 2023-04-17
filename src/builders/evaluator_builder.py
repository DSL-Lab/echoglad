from src.core.evaluators import BinaryAccuracyEvaluator, BalancedBinaryAccuracyEvaluator, LandmarkErrorEvaluator, \
                                LandmarkExpectedCoordiantesEvaluator, MSEEvaluator
from copy import deepcopy


EVALUATORS = {
    'accuracy': BinaryAccuracyEvaluator,
    'balancedaccuracy': BalancedBinaryAccuracyEvaluator,
    'landmarkcoorderror': LandmarkExpectedCoordiantesEvaluator,
    'landmarkerror': LandmarkErrorEvaluator,
    'mse': MSEEvaluator
}


def build(eval_config, logger):
    config = deepcopy(eval_config)
    batch_size = config.pop('batch_size')
    frame_size = config.pop('frame_size')
    use_coord_graph = config.pop('use_coordinate_graph')

    evaluators = dict()
    for standard in config['standards']:
        if standard in ['landmarkerror', 'landmarkcoorderror']:
            evaluators.update({standard: EVALUATORS[standard](logger=logger,
                                                              batch_size=batch_size,
                                                              frame_size=frame_size,
                                                              use_coord_graph=use_coord_graph)})
        else:
            evaluators.update({standard: EVALUATORS[standard](logger=logger)})

        logger.infov('{} evaluator is built.'.format(standard.upper()))

    return evaluators
