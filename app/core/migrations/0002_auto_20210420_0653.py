# Generated by Django 2.1.15 on 2021-04-20 06:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='device',
            name='state',
            field=models.CharField(choices=[('On', 'On'), ('Off', 'Off')], max_length=255),
        ),
    ]