from src.core.meters import AverageEpochMeter

def build(logger):
    meter = AverageEpochMeter('loss meter', fmt=':f')

    logger.infov('Loss meter is built.')
    return meter
