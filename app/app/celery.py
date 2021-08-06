from __future__ import absolute_import, unicode_literals

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')

app = Celery('app')


app.conf.beat_schedule = {
    "Train model": {
        "task": "sites.tasks.trainDRLGYM",
        "schedule": 7200,
        "args": (24, 72, 10, 20),
    }
}


# celery -A app beat -l INFO

app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
