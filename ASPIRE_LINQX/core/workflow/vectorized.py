from pydantic import field_validator
from pydantic_core import PydanticCustomError
from typing import (
    Dict, 
    Any,
    List, 
    Union,
    Optional,
)
from ASPIRE_LINQX.core.workflow.base import BaseDriverWorkflow
from ASPIRE_LINQX.core.command.vectorized import (
    BaseDriverCommand,
    _CommandVectorBase
)

import torch
import pandas as pd
import numpy as np

class _WorkflowVectorBase():
    '''
    Description
    -----------
    Private class designed to provide based functionality to all vectorized
    workflows.
    '''

class VBaseDriverWorkflow(BaseDriverWorkflow, _WorkflowVectorBase):
    '''
    Description
    -----------
    Base model for workflows which contain only vectorized driver commands.
    This workflow will not allow initlization with any commands which are not
    vectorized (extend the _CommandVectorBase class)

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the workflow
    ```
    commands : List[BaseDriverCommand]
    ```
    A list of DriverCommands (in order) which define the workflow steps.
    ```
    wf_globals: Dict[str, Any]
    ```
    A dictionary of global varaibles that are shared between all workflow command during execution.
    Commands can read in from and write out to this list.

    Methods
    -------
    ```
    def clear_wf_globals()
    ```
    Resets `self.wf_globals` to an empty dictionary.
    ```
    def exec(self, list_kwargs: List[Union[Dict[str, Any], None]], list_save_vars: List[Union[Dict[str,str], None]]) -> List[Dict]
    ```
    Executes the DriverWorkflow (in order) with each command provided with workflow gloabs and its own set of kwargs.
    '''

    @field_validator("commands")
    @classmethod
    def validate_commands(cls, commands: List[BaseDriverCommand]):
        '''
        Decorators
        ----------
        ```
        @classmethod
        @validator("commands")
        ```
        Description
        -----------
        Additional validator to ensure that all commands of a vectorized driver workflow
        are vectorized commands (they should extend from BaseDriverCommand and _CommandVectorBase)

        Parameters
        ----------
        ```
        commands : List[BaseDriverCommand]
        ```
        A list of BaseDriverCommand objects to be called in order

        Raises
        ------
        ```
        raise TypeError
        ```
        Raises a TypeError if any command is not a subclass of _CommandVectorBase 

        Returns
        -------
        ```
        return commands : List[BaseDriverCommand]
        ```
        Returns the command list upon successful validation
        '''
        if not all(isinstance(command, _CommandVectorBase) for command in commands):
            raise PydanticCustomError(
                'non_vectorized_command_error',
                "Non-vectorize command present, all commands must be vectorized",
                {}
                )
        return commands