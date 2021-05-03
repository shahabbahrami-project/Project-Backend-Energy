from django.urls import path, include
from rest_framework.routers import DefaultRouter

from sites import views


router = DefaultRouter()
router.register('devicetypes', views.DeviceTypeViewSet)
router.register('devices', views.DeviceViewSet)
router.register('devicedata', views.DeviceDataViewSet)
router.register('sensortypes', views.SensorTypeViewSet)
router.register('sensors', views.SensorViewSet)
router.register('sites', views.SiteViewSet)
router.register('sensordata', views.SensorDataViewSet)

app_name = 'sites'

urlpatterns = [
    path('', include(router.urls))
]
