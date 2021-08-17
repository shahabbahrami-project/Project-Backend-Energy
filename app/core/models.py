import uuid
import time
# from celery.result import AsyncResult
import os
from django.db import models
from django.db.models import F
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, \
    PermissionsMixin
from django.conf import settings
from django.utils.timezone import now
from datetime import datetime
from .celery import app
# class UserManager(BaseUserManager):
#
#     def create_user(self, email, password=None, **extra_fields):
#         """Creates and saves a new user"""
#         user = self.model(email=email, **extra_fields)
#         user.set_password(password)
#         user.save(using=self._db)
#
#         return user
#
#
# class User(AbstractBaseUser, PermissionsMixin):
#     """Custom user model that supports email instead of username"""
#     email = models.EmailField(max_length=255, unique=True)
#     name = models.CharField(max_length=255)
#     is_active = models.BooleanField(default=True)
#     is_staff = models.BooleanField(default=False)
#
#     objects = UserManager()
#
#     USERNAME_FIELD = 'email'


def user_image_file_path(instance, filename):
    """Generate file path for new user image"""
    ext = filename.split('.')[-1]
    filename = f'{uuid.uuid4()}.{ext}'

    return os.path.join('uploads/user/', filename)


def site_image_file_path(instance, filename):
    """Generate file path for new site image"""
    ext = filename.split('.')[-1]
    filename = f'{uuid.uuid4()}.{ext}'

    return os.path.join('uploads/site/', filename)


def recipe_image_file_path(instance, filename):
    """Generate file path for new recipe image"""
    ext = filename.split('.')[-1]
    filename = f'{uuid.uuid4()}.{ext}'

    return os.path.join('uploads/recipe/', filename)


class UserManager(BaseUserManager):

    def create_user(self, email, password=None, **extra_fields):
        """Creates and saves a new user"""
        if not email:
            raise ValueError('Users must have an email address')
        user = self.model(email=self.normalize_email(email), **extra_fields)
        user.set_password(password)
        user.save(using=self._db)

        return user

    def create_superuser(self, email, password, **extra_fields):
        """Creates and saves new superuser"""
        user = self.create_user(email, password, **extra_fields)
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)

        return user


class User(AbstractBaseUser, PermissionsMixin):
    """Custom user model that supports email instead of username"""
    email = models.EmailField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    observerInSitesIds = models.CharField(max_length=255, default="")
    operatorInSitesIds = models.CharField(max_length=255,  default="")
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    image = models.ImageField(null=True, upload_to=user_image_file_path)
    objects = UserManager()

    USERNAME_FIELD = 'email'


class Tag(models.Model):
    """Tag to be used for a recipe"""
    name = models.CharField(max_length=255)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    def __str__(self):
        return self.name


class Ingredient(models.Model):
    """Ingredient to be used for a recipe"""
    name = models.CharField(max_length=255)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    def __str__(self):
        return self.name


class Recipe(models.Model):
    """Recipe object"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    title = models.CharField(max_length=255)
    time_minutes = models.IntegerField()
    price = models.DecimalField(max_digits=5, decimal_places=2)
    link = models.CharField(max_length=255, blank=True)
    ingredients = models.ManyToManyField('Ingredient')
    tags = models.ManyToManyField('Tag')
    image = models.ImageField(null=True, upload_to=recipe_image_file_path)

    def __str__(self):
        return self.title


class SensorType(models.Model):
    """Sensor to be used for a site"""
    TYPES = (
        ('Temperature', 'Temperature'),
        ('Light', 'Light'),
        ('Motion', 'Motion'),
        ('Space', 'Space'),
        ('Humidity', 'Humidity'),
    )
    name = models.CharField(max_length=255, choices=TYPES)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    def __str__(self):
        return self.name


class Sensor(models.Model):
    """Sensor to be used for a site"""
    name = models.CharField(max_length=255)
    type = models.ForeignKey(
        SensorType,
        on_delete=models.CASCADE,
        null=True
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    lastRealValue = models.FloatField(default=0.0, null=False, blank=False)
    lastAlgoValue = models.FloatField(default=0.0, null=False, blank=False)
    lastTimeValue = models.DateTimeField(null=True, blank=False)
    created_at = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return self.name


class DeviceType(models.Model):
    """Device Type to be used for a site"""
    TYPES = (
        ('HVAC', 'HVAC'),
        ('Lighting', 'Lighting'),
        ('Pump', 'Pump'),
    )
    name = models.CharField(max_length=255, choices=TYPES)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    def __str__(self):
        return self.name


class Device(models.Model):
    """Device to be used for a site"""
    States = (
        ('On', 'On'),
        ('Off', 'Off'),
    )
    name = models.CharField(max_length=255)
    type = models.ForeignKey(
        DeviceType,
        on_delete=models.CASCADE,
        null=True
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    sensors = models.ManyToManyField('Sensor', blank=True)
    stateReal = models.CharField(max_length=255, null=True, choices=States)
    stateAlgo = models.CharField(max_length=255,  null=True, choices=States)
    lastRealPowerValue = models.FloatField(
        default=0.0, null=False, blank=False)
    lastAlgoPowerValue = models.FloatField(
        default=0.0, null=False, blank=False)
    costReal = models.FloatField(default=0.0, null=False, blank=False)
    costAlgo = models.FloatField(default=0.0, null=False, blank=False)
    lastTimeValue = models.DateTimeField(null=True, blank=False)
    created_at = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return self.name


class SensorData(models.Model):
    """Sensor to be used for a site"""
    name = models.ForeignKey(
        Sensor,
        on_delete=models.CASCADE,
        null=True
    )
    type = models.ForeignKey(
        SensorType,
        on_delete=models.CASCADE,
        null=True
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    datetime = models.DateTimeField(null=True, blank=False)
    realvalue = models.FloatField(default=0.0, null=False, blank=False)
    algovalue = models.FloatField(default=0.0, null=False, blank=False)

    def save(self, *args, **kwargs):
        if not self.pk:
            Sensor.objects.filter(pk=self.name_id).update(
                lastRealValue=self.realvalue)
            Sensor.objects.filter(pk=self.name_id).update(
                lastAlgoValue=self.algovalue)
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.name) + "\t \t \t" + str(self.datetime)


class DeviceData(models.Model):
    """Sensor to be used for a site"""
    States = (
        ('On', 'On'),
        ('Off', 'Off'),
    )
    name = models.ForeignKey(
        Device,
        on_delete=models.CASCADE,
        null=True
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    datetime = models.DateTimeField(null=True, blank=False)
    realpowervalue = models.FloatField(default=0.0, null=False, blank=False)
    algopowervalue = models.FloatField(default=0.0, null=False, blank=False)
    realstatevalue = models.CharField(
        max_length=255, null=True, choices=States)
    algostatevalue = models.CharField(
        max_length=255, null=True, choices=States)
    costreal = models.FloatField(default=0.0, null=False, blank=False)
    costalgo = models.FloatField(default=0.0, null=False, blank=False)

    def save(self, *args, **kwargs):
        if not self.pk:
            Device.objects.filter(pk=self.name_id).update(
                lastRealPowerValue=self.realpowervalue)
            Device.objects.filter(pk=self.name_id).update(
                lastAlgoPowerValue=self.algopowervalue)
            Device.objects.filter(pk=self.name_id).update(
                stateReal=self.realstatevalue)
            Device.objects.filter(pk=self.name_id).update(
                stateAlgo=self.algostatevalue)
            Device.objects.filter(pk=self.name_id).update(
                costReal=self.costreal)
            Device.objects.filter(pk=self.name_id).update(
                costAlgo=self.costalgo)
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.name) + "\t \t \t" + str(self.datetime)


class Site(models.Model):
    """Site object"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    name = models.CharField(max_length=255)
    locationX = models.FloatField(null=True, blank=True)
    locationY = models.FloatField(null=True, blank=True)
    link = models.CharField(max_length=255, blank=True)
    sensors = models.ManyToManyField('Sensor', blank=True)
    devices = models.ManyToManyField('Device', blank=True)
    timezone = models.CharField(max_length=255, blank=True)
    image = models.ImageField(null=True, upload_to=site_image_file_path)
    created_at = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return self.name


class CeleryTask(models.Model):
    device = models.OneToOneField(Device, on_delete=models.CASCADE, null=True)
    completed_at = models.DateTimeField(null=True)
    job_id = models.CharField(max_length=255, null=True)
    done = models.BooleanField(default=False)

    def __str__(self):
        return self.job_id

    def waitForCompletion(self):
        print(self.job_id)
        task = app.AsyncResult(self.job_id)
        while not task.ready():
            time.sleep(1)
        self.done = True
        self.completed_at = datetime.now()
        self.save()


class TrainingResult(models.Model):
    device = models.OneToOneField(Device, on_delete=models.CASCADE, null=True)
    model = models.JSONField(null=True)
    weights_bin = models.BinaryField(editable=True)
    last_updated_at = models.DateTimeField(default=now)

    def __str__(self):
        return f"{self.device}"
