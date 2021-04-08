from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import TestCase

from rest_framework import status
from rest_framework.test import APIClient

from core.models import Sesnor

from sites.serializers import SensorSerializer


SENSOR_URL = reverse('sites:sensor-list')


class PublicSensorsApiTests(TestCase):
    """Test the publicly available sensors API"""

    def setUp(self):
        self.client = APIClient()

    def test_login_required(self):
        """Test that login is required for retrieving sensors"""
        res = self.client.get(SENSOR_URL)

        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)


class PrivateSensorsApiTests(TestCase):
    """Test the authorized user sensors API"""

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            'shahab@yahoo.com',
            'password123'
        )
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def test_retrieve_sensors(self):
        """Test retrieving sensors"""
        Sensor.objects.create(user=self.user, name='Space')
        Sensor.objects.create(user=self.user, name='Temperature')

        res = self.client.get(SENSOR_URL)

        sensors = Sensor.objects.all().order_by('-name')
        serializer = SensorSerializer(sensors, many=True)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data, serializer.data)

    def test_sensor_limited_to_user(self):
        """Test that sensors are for authenticated user"""
        user2 = get_user_model().objects.create_user(
            'other@yahoo.com',
            'testpass'
        )
        Sensor.objects.create(user=user2, name='Space')
        sensor = Sensor.objects.create(user=self.user, name='Temperature')

        res = self.client.get(SENSOR_URL)

        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]['name'], sensor.name)

    def test_create_sensor_successful(self):
        """Test Creating a new sensor"""
        payload = {'name': 'Temperature'}
        self.client.post(SENSOR_URL, payload)

        exists = Sensor.objects.filter(
            user=self.user,
            name=payload['name']
        ).exists()
        self.assertTrue(exists)

    def test_create_sensor_invalid(self):
        """Test creating a new sensor with invalid payload"""
        payload = {'name': ''}
        res = self.client.post(SENSOR_URL, payload)

        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # def test_retrieve_sensors_assigned_to_recipes(self):
    #     """Test filtering sensors by those assigned to recipes"""
    #     sensor1 = Sensor.objects.create(user=self.user, name='Egg')
    #     sensor2 = Sensor.objects.create(user=self.user, name='Cheese')
    #     recipe = Recipe.objects.create(
    #         title='Coriander eggs on toast',
    #         time_minutes=10,
    #         price=5.00,
    #         user=self.user
    #     )
    #     recipe.sensors.add(sensor1)
    #
    #     res = self.client.get(SENSOR_URL, {'assigned_only': 1})
    #
    #     serializer1 = SensorSerializer(sensor1)
    #     serializer2 = SensorSerializer(sensor2)
    #     self.assertIn(serializer1.data, res.data)
    #     self.assertNotIn(serializer2.data, res.data)
    #
    # def test_retrieve_sensors_assigned_unique(self):
    #     """Test filtering sensors by assiging returns unique items"""
    #     sensor = Sensor.objects.create(user=self.user, name='Egg')
    #     Sensor.objects.create(user=self.user, name='Cheese')
    #     recipe1 = Recipe.objects.create(
    #         title='Coriander eggs on toast',
    #         time_minutes=10,
    #         price=5.00,
    #         user=self.user
    #     )
    #     recipe1.sensors.add(sensor)
    #     recipe2 = Recipe.objects.create(
    #         title='Steak',
    #         time_minutes=10,
    #         price=5.00,
    #         user=self.user
    #     )
    #     recipe2.sensors.add(sensor)
    #
    #     res = self.client.get(SENSOR_URL, {'assigned_only': 1})
    #
    #     self.assertEqual(len(res.data), 1)
