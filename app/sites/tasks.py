from __future__ import unicode_literals, print_function

from celery import shared_task
import numpy as np

@shared_task
def add(x, y):
    return x + y


@shared_task
def ForwardDRLGYM(dqn,W, Sample):
    action = dqn.forward(Sample)
    print('action is', action)
    t=Sample[5]
    z=np.exp(-300/130)
    OutdoorTemp_now=Sample[2]
    People_now=Sample[4]
    Desire_now=Sample[1]
    Prev_IndTemp=Sample[0]
    Price=Sample[3]
    airTemp=10+action
    if airTemp>Prev_IndTemp:
        IndoorTemp_new= Prev_IndTemp+(OutdoorTemp_now-Prev_IndTemp)*z+(airTemp-Prev_IndTemp)*z
        Tset= min(Prev_IndTemp+3,30)
    else:
        IndoorTemp_new= Prev_IndTemp+(OutdoorTemp_now-Prev_IndTemp)*z
        Tset= Prev_IndTemp/2


    # Calculate reward
    if airTemp>Prev_IndTemp:
        reward = -(Price*abs(airTemp)+People_now*W*abs(float(Desire_now)-IndoorTemp_new))
    else:
        reward = -(People_now*W*abs(Desire_now-IndoorTemp_new))

    Cost=-reward


    return airTemp,IndoorTemp_new,Tset, Cost