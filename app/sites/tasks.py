from __future__ import unicode_literals, print_function

from celery import shared_task
import numpy as np
from .DQN.TrainDRLGYM import TrainDRLGYM


@shared_task
def add(x, y):
    return x + y


@shared_task
def trainDRLGYM(FromHour, ToHour, W, Desire):
    TrainDRLGYM(FromHour, ToHour, W, Desire)
    print("Finished training")
