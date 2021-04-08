from rest_framework import serializers

from core.models import Sensor, Site


class SensorSerializer(serializers.ModelSerializer):
    """Serializer for Sensor objects"""

    class Meta:
        model = Sensor
        fields = ('id', 'name')
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
            'id', 'name', 'locationX', 'locationY', 'sensors', 'link'
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
