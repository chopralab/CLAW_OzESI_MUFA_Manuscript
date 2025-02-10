from pydantic import (
    BaseModel,
    model_validator,
    Field,
)

from typing import Type, Literal

import json

class BaseMessage(BaseModel):
    '''
    Formatted message which is sent between endpoints.

    NOTE - It is recomended to use integrations in 'core' during implementation.

    Parameters
    ```
    message_category: str = Field(default="undefined")
    message_type: Literal['request', 'response']
    message_contents: str = ""  # The contents of a message
    message_schema: Type[BaseModel] | None # The schema of the message if provided
    ```

    The message content get validated against the schema if one is provided.
    
    Otherwise they are validated againts JSON format.
    '''
    message_category: str = Field(default='undefined')
    message_type: Literal['request', 'response', 'other'] = Field(default='other')
    message_contents: str = Field(default='')
    message_schema: Type[BaseModel] | None = Field(default=None, exclude=True)

    def _validate_schema(self) -> None:
        '''
        Validates the message content against the schema.
        '''
        try:
            self.message_schema(**json.loads(self.message_contents))
        except:
            raise ValueError("Incompatible message contents and schema")
        
    def _vaidate_json_format(self) -> None:
        '''
        If the response is not json formatted, an error is returned
        '''
        try: 
            json.loads(self.message_contents)
        except:
            return {'response': {'error': 'received invalid JSON formatted message'}}

    @model_validator(mode='after')
    def validate_message(self) -> 'BaseMessage':
        '''
        Assign the message category to the __name__ of the schema if provided.

        Validates the message content against the schema.
        '''
        if self.message_schema is not None:
            self.message_category = self.message_schema.__name__ 
            self._validate_schema()
        else: 
            self.model_validate_json()