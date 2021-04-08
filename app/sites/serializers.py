from rest_framework import serializers

from core.models import Sensor


class SensorSerializer(serializers.ModelSerializer):
    """Serializer for Sensor objects"""

    class Meta:
        model = Sensor
        fields = ('id', 'name')
        read_only_fields = ('id',)
