import os

from tqdm import tqdm
import logging

def main_print(log) -> None:
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        print(log)

def main_log(log) -> None:
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        logger = logging.getLogger()
        logger.info(log)

def process_log(log) -> None:
    logger = logging.getLogger()
    logger.info(log)

def main_tqdm(inner):
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        return tqdm(inner)
    else:
        return inner

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)