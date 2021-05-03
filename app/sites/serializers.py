from rest_framework import serializers

from core.models import Sensor, Device, Site, SensorType, DeviceType, SensorData, DeviceData


class SensorTypeSerializer(serializers.ModelSerializer):
    """Serializer for Sensor Types objects"""

    class Meta:
        model = SensorType
        fields = ('id', 'name')
        read_only_fields = ('id',)


class DeviceTypeSerializer(serializers.ModelSerializer):
    """Serializer for Device Types objects"""

    class Meta:
        model = DeviceType
        fields = ('id', 'name')
        read_only_fields = ('id',)


class SensorSerializer(serializers.ModelSerializer):
    """Serializer for Sensor objects"""

    class Meta:
        model = Sensor
        fields = ('id', 'name', 'type', 'user', 'lastRealValue', 'lastAlgoValue', 'lastTimeValue', 'created_at')
        read_only_fields = ('id',)


class DeviceSerializer(serializers.ModelSerializer):
    """Serializer for Device objects"""

    class Meta:
        model = Device
        fields = ('id', 'name', 'type', 'user', 'sensors', 'stateReal', 'stateAlgo', 'lastRealPowerValue', 'lastAlgoPowerValue',  'costReal', 'costAlgo', 'lastTimeValue', 'created_at')
        read_only_fields = ('id',)

class SensorDataSerializer(serializers.ModelSerializer):
    """Serializer for Sensor Data objects"""

    class Meta:
        model = SensorData
        fields = ('id', 'name', 'user', 'realvalue', 'algovalue', 'datetime')
        read_only_fields = ('id',)


class DeviceDataSerializer(serializers.ModelSerializer):
    """Serializer for Sensor Data objects"""

    class Meta:
        model = DeviceData
        fields = ('id', 'name', 'user', 'realpowervalue', 'algopowervalue',  'realstatevalue', 'algostatevalue', 'costreal', 'costalgo', 'datetime')
        read_only_fields = ('id',)


class SiteSerializer(serializers.ModelSerializer):
    """Serializer for site objects"""
    sensors = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=Sensor.objects.all()
    )

    class Meta:
        model = Site
        fields = (
            'id', 'name', 'locationX', 'locationY', 'devices', 'sensors', 'timezone', 'link', 'image', 'created_at'
        )
        read_only_fields = ('id',)


class SiteDetailSerializer(SiteSerializer):
    """Serializer for site detail"""
    sensors = SensorSerializer(many=True, read_only=True)


class SiteImageSerializer(serializers.ModelSerializer):
    """Serializer for uploading images to sites"""

    class Meta:
        model = Site
        fields = ('id', 'image')
        read_only_fields = ('id', )
