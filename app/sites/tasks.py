from __future__ import unicode_literals, print_function

import os
from celery import shared_task
from celery.signals import task_success
import numpy as np
from .DQN.TrainDRLGYM import TrainDRLGYM
from core.models import TrainingResult, Device
from django.utils.timezone import now
import datetime


@shared_task
def trainDRLGYM(FromHour, ToHour, W, Desire):
    for device in Device.objects.all():
        obj, created = TrainingResult.objects.get_or_create(device=device)

        dqn, model_json = TrainDRLGYM(FromHour, ToHour, W, Desire)
        weights_file = 'weights.h5'
        dqn.save_weights(weights_file, overwrite=True)
        Bytes = b''
        with open(weights_file, "rb") as f:
            while (byte := f.read(1)):
                Bytes += byte
        obj.weights_bin = Bytes
        obj.model = model_json
        obj.last_updated_at = datetime.datetime.now()
        obj.save()

        os.remove(weights_file)
        assert not os.path.exists(weights_file)
