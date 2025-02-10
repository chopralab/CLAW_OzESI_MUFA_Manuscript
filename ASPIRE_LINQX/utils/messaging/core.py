from uuid import UUID
import json

from ASPIRE_LINQX.utils.messaging.base import BaseMessage
from ASPIRE_LINQX.core.command.base import BaseRunCommand
from ASPIRE_LINQX.core.library.base import BaseInfoMicroservice

class MessageRequest():
    '''
    Unimplemented class to denote a message request
    '''
    def __init__(self) -> None:
        pass

class MessageResponse():
    '''
    Unimplemented class to denote a message response
    '''
    def __init__(self) -> None:
        pass

class RunCommandRequest(BaseMessage):
    '''
    Message request from server to client on running a command.
    '''
    def __init__(self, run_command: BaseRunCommand):
        super().__init__(
            message_type='request',
            message_contents=run_command.model_dump_json(),
            message_schema=BaseRunCommand,
        )

class RunCommandResponse(BaseMessage):
    '''
    Message response from client to server on command output.
    '''
    def __init__(self, message_contents: str):
        super().__init__(
            message_category=BaseRunCommand.__name__,
            message_type='response',
            message_contents=message_contents,
        )

class MicroserviceRequest(BaseMessage):
    '''
    Message request from server to client to send over information on a specific microservice.
    '''
    def __init__(self, uuid: UUID):
        super().__init__(
            message_category=BaseInfoMicroservice.__name__,
            message_type='request',
            message_contents=json.dumps({'uuid': uuid}),
        )

class MicroserviceResponse(BaseMessage):
    '''
    Message response from client to server with information on a specific microservice.
    '''
    def __init__(self, microservice: BaseInfoMicroservice):
        super().__init__(
            message_type='response',
            message_contents=microservice.model_dump_json(),
            message_schema=BaseInfoMicroservice,
        )