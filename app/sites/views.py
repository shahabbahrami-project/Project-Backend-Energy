from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import viewsets, mixins, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from core.models import Sensor, Site, SensorType, SensorData, Device, DeviceType, DeviceData

from sites import serializers


class BaseSiteAttrViewSet(viewsets.GenericViewSet,
                            mixins.ListModelMixin,
                            mixins.CreateModelMixin,
                            mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin):
    """Base viewset for user owned site attributes"""
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        """Return objects for the current authenticated user only"""
        assigned_only = bool(
            int(self.request.query_params.get('assigned_only', 0))
        )
        queryset = self.queryset
        if assigned_only:
            queryset = queryset.filter(site__isnull=False)

        return queryset.filter(
            user=self.request.user
        ).order_by('-name').distinct()

    def perform_create(self, serializer):
        """Create new object"""
        serializer.save(user=self.request.user)


class DeviceTypeViewSet(BaseSiteAttrViewSet):
    """Manage device type in the database"""
    queryset = DeviceType.objects.all()
    serializer_class = serializers.DeviceTypeSerializer


class DeviceViewSet(BaseSiteAttrViewSet):
    """Manage device in the database"""
    queryset = Device.objects.all()
    serializer_class = serializers.DeviceSerializer


class SensorTypeViewSet(BaseSiteAttrViewSet):
    """Manage sensor type in the database"""
    queryset = SensorType.objects.all()
    serializer_class = serializers.SensorTypeSerializer


class SensorViewSet(BaseSiteAttrViewSet):
    """Manage sensor in the database"""
    queryset = Sensor.objects.all()
    serializer_class = serializers.SensorSerializer


class SensorDataViewSet(BaseSiteAttrViewSet):
    """Manage sensor in the database"""
    queryset = SensorData.objects.all()
    serializer_class = serializers.SensorSerializer


class DeviceDataViewSet(BaseSiteAttrViewSet):
    """Manage sensor in the database"""
    queryset = DeviceData.objects.all()
    serializer_class = serializers.DeviceDataSerializer


class SiteViewSet(viewsets.ModelViewSet):
    """Manage sites in the database"""
    serializer_class = serializers.SiteSerializer
    queryset = Site.objects.all()
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def _params_to_ints(self, qs):
        """Convert a list of string IDs to a list of integers"""
        return [int(str_id) for str_id in qs.split(',')]

    def get_queryset(self):
        """Return objects for the current authenticated user only"""
        sensors = self.request.query_params.get('sensors')
        queryset = self.queryset
        if sensors:
            sensor_ids = self._params_to_ints(sensors)
            queryset = queryset.filter(sensors__id__in=sensor_ids)

        return queryset.filter(user=self.request.user)

    def get_serializer_class(self):
        """Return appropriate serializer class for our request"""
        if self.action == 'retrieve':
            return serializers.SiteDetailSerializer
        elif self.action == 'upload_image':
            return serializers.SiteImageSerializer

        return self.serializer_class

    def perform_create(self, serializer):
        """Create a new site"""
        serializer.save(user=self.request.user)

    @action(methods=['POST'], detail=True, url_path='upload-image')
    def upload_image(self, request, pk=None):
        """Upload an image to a site"""
        site = self.get_object()
        serializer = self.get_serializer(
            site,
            data=request.data
        )

        if serializer.is_valid():
            serializer.save()
            return Response(
                serializer.data,
                status=status.HTTP_200_OK
            )

        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
