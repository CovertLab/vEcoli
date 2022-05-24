from unum import Unum
from vivarium.core.registry import Serializer
from vivarium.library.topology import convert_path_style, normalize_path

from ecoli.library.parameters import param_store, Parameter


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


class ParameterSerializer(Serializer):

    def can_serialize(self, _):
        return False

    def serialize_to_string(self, data):
        raise NotImplementedError(
            'The ParameterSerializer does not support serialization.')

    def deserialize_from_string(self, data):
        path = normalize_path(convert_path_style(data))
        return param_store.get(path)
