from __future__ import unicode_literals, print_function

import os
from celery import shared_task
from celery.signals import task_success
import numpy as np
from .DQN.TrainDRLGYM import TrainDRLGYM
from core.models import TrainingResult
from django.utils.timezone import now
import datetime



@shared_task
def add(x, y):
    return x + y


@shared_task
def trainDRLGYM(FromHour, ToHour, W, Desire, device_id=None):
    res, created = TrainingResult.objects.get_or_create(
        device_id=device_id,
    )
    if created:
        res.save()
    dqn, model_json = TrainDRLGYM(FromHour, ToHour, W, Desire)
    weights_file = 'weights.h5'
    dqn.save_weights(weights_file, overwrite=True)
    Bytes = b''
    with open(weights_file, "rb") as f:
        while (byte := f.read(1)):
            Bytes += byte
    obj = TrainingResult.objects.get(device_id=device_id)
    print(obj)
    obj.weights_bin = Bytes
    obj.model = model_json
    obj.last_updated_at = datetime.datetime.now()
    obj.save()

    os.remove(weights_file)
    assert not os.path.exists(weights_file)
