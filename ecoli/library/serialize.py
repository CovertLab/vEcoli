from unum import Unum
from vivarium.core.registry import Serializer


class UnumSerializer(Serializer):

    def can_serialize(self, data):
        return isinstance(data, Unum)

    def serialize_to_string(self, data):
        num = str(data.asNumber())
        assert ' ' not in num
        return f'{num} {data.strUnit()}'

    def deserialize_from_string(self, data):
        # WARNING: This deserialization is lossy and drops the unit
        # information since there is no easy way to parse Unum's unit
        # strings.
        return data.split(' ')[0]
