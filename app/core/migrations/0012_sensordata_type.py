# Generated by Django 2.1.15 on 2021-05-13 22:01

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0011_auto_20210429_2051'),
    ]

    operations = [
        migrations.AddField(
            model_name='sensordata',
            name='type',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='core.SensorType'),
        ),
    ]