from django.urls import path, include
from rest_framework.routers import DefaultRouter

from sites import views


router = DefaultRouter()
router.register('sensors', views.SensorViewSet)
router.register('sites', views.SiteViewSet)

app_name = 'sites'

urlpatterns = [
    path('', include(router.urls))
]
