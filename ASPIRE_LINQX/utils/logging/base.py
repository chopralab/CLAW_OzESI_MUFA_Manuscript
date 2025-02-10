from pydantic import BaseModel, Field
from typing import List, Any

class BaseLogger(BaseModel):
    '''
    Base class for a LINQX command logger. This class should be extended by 
    any implemented logger.

    A class which extends this class can be attached to any class which extends the
    BaseDriverCommand class to selectively log the inputs and outputs when the driver 
    command is called. If the logger is attached to a workflow, it generally is applied 
    to all commands in that workflow and logs all inputs and outputs.
    '''
    log: Any =  Field(description='The log object')
    
    def _validate_in_ex(
        inputs_in: List[str] | None = None,
        outputs_in: List[str] | None = None,
        inputs_ex: List[str] | None = None,
        outputs_ex: List[str] | None = None,
    ) -> None:
        '''
        Validates include/exclude varaibles for logging.

        Raises an error if there is a contradiction in the include/exclude setting
        '''
        raise NotImplementedError

    def _log_data(
        self,
        inputs_in: List[str] | None = None,
        outputs_in: List[str] | None = None,
        inputs_ex: List[str] | None = None,
        outputs_ex: List[str] | None = None,
        **kwargs
    ) -> None:
        '''
        Writes the provided data to the loggers internal log

        ### - Not implemented in the base class
        '''
        raise NotImplementedError
    
    def log_data(
        self,
        inputs_in: List[str] | None = None,
        outputs_in: List[str] | None = None,
        inputs_ex: List[str] | None = None,
        outputs_ex: List[str] | None = None,
        **kwargs
    ) -> None:
        '''
        Validates logging include/exclude and writes data to the 
        loggers internal log
        '''
        self._validate_in_ex()
        self._log_data()

    def get_log(self) -> Any:
        '''
        Default behavior returns the logger object's internal log
        '''
        return self.log