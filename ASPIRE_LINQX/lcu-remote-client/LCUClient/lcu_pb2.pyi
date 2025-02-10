from google.protobuf import wrappers_pb2 as _wrappers_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Failure(_message.Message):
    __slots__ = ["reason"]
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: _wrappers_pb2.StringValue
    def __init__(self, reason: _Optional[_Union[_wrappers_pb2.StringValue, _Mapping]] = ...) -> None: ...

class LcuAnnounce(_message.Message):
    __slots__ = ["description", "id"]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    description: _wrappers_pb2.StringValue
    id: _wrappers_pb2.StringValue
    def __init__(self, id: _Optional[_Union[_wrappers_pb2.StringValue, _Mapping]] = ..., description: _Optional[_Union[_wrappers_pb2.StringValue, _Mapping]] = ...) -> None: ...

class LcuClientMessage(_message.Message):
    __slots__ = ["lcu_announce", "lcu_status_req"]
    LCU_ANNOUNCE_FIELD_NUMBER: _ClassVar[int]
    LCU_STATUS_REQ_FIELD_NUMBER: _ClassVar[int]
    lcu_announce: LcuAnnounce
    lcu_status_req: LcuStatusRequest
    def __init__(self, lcu_announce: _Optional[_Union[LcuAnnounce, _Mapping]] = ..., lcu_status_req: _Optional[_Union[LcuStatusRequest, _Mapping]] = ...) -> None: ...

class LcuResponseMessage(_message.Message):
    __slots__ = ["failure", "lcu_status", "success"]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    LCU_STATUS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    failure: Failure
    lcu_status: LcuStatus
    success: Success
    def __init__(self, success: _Optional[_Union[Success, _Mapping]] = ..., failure: _Optional[_Union[Failure, _Mapping]] = ..., lcu_status: _Optional[_Union[LcuStatus, _Mapping]] = ...) -> None: ...

class LcuStatus(_message.Message):
    __slots__ = ["lcu_announces"]
    LCU_ANNOUNCES_FIELD_NUMBER: _ClassVar[int]
    lcu_announces: _containers.RepeatedCompositeFieldContainer[LcuAnnounce]
    def __init__(self, lcu_announces: _Optional[_Iterable[_Union[LcuAnnounce, _Mapping]]] = ...) -> None: ...

class LcuStatusRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class Success(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
