# Generated by Django 2.1.15 on 2021-04-25 23:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_devicedata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sensortype',
            name='name',
            field=models.CharField(choices=[('Temperatrue', 'Temperature'), ('Light', 'Light'), ('Motion', 'Motion'), ('Space', 'Space'), ('Humidty', 'Humidity')], max_length=255),
        ),
    ]