from pydantic import (
    model_validator,
    root_validator, 
    PrivateAttr
)
from typing import (
    Dict, 
    Any,
    List,
    Optional,
)
import ast
from ASPIRE_LINQX.core.command.base import BaseDriverCommand
from ASPIRE_LINQX.core.command.core import InteractiveResultDriverCommand
from ASPIRE_LINQX.core.parameter.vectorized import VParameter, VectorLike

# Vector imports
import numpy as np
import pandas as pd
import torch

class _CommandVectorBase():
    """
    Description
    -----------
    Private class designed to add common base functionality to vectorized commands.
    This class is extend by all vectorized command classes in addition to the
    corresponding non-vectorized command.

    NOTE
    ----
    I want this to be a base model to handle init of private vector attributes as well
    as common vector functionality but at this point in time it looks like pydantic
    does not allow for multiple inheritance from model classes with seperate private
    attributes. If this is fixed in the future, the implementation can be changed.
    """

    def _validate_param_len(
            self,
            parameters: Dict[str, VParameter]
    ) -> int:
        '''
        Description
        -----------
        Ensure that all vectorized parameters are the same size on their primary axis.
        Raises an error if they are not.
        
        Parameters
        ----------
        ```
        parameters : Dict[str, VParameter]
        ```
        A dictionary of vectorized parameters to evaluate.

        Raises
        ------
        ```
        raise AssertionError
        ```
        raises an AssertionError if not all vectorized parameters are the
        same length on thier primary axis
        
        Return
        ------
        ```
        return param_len -> int
        ```
        The length of all of the parameters (0 if there are no parameters)
        '''
        if not parameters:
            return 0
        shape_0_list = [elem.value.shape[0] for elem in parameters.values()]
        if shape_0_list.count(shape_0_list[0]) != len(shape_0_list):
            raise AssertionError(f"Parameter vectors must be the same size on axis 0, received: {shape_0_list}")
        return shape_0_list[0]
        
    def _validate_order(
            self,
            parameters: Dict[str, VParameter],
            order: List[str]
    ) -> None:
        '''
        Description
        -----------
        Ensure that all elemnts in an ordering list are valid
        parameters

        Parameters
        ---------
        ```
        parameters : Dict[str, VParameters]
        ```
        A dictionary of vectorized parameters to evaluate.
        ```
        order : List[str]
        ```
        A list of the parameter ordering

        Raises
        ------
        ```
        raise KeyError
        ```
        Raises a KeyError if not all of the elemnts of the ordering list
        are valid parameters
        '''
        if not all(elem in parameters.keys() for elem in order):
            raise KeyError

    def _squeeze_params_to_ndarray(
            self,
            parameters: Dict[str, VParameter],
            order: Optional[List[str]] = None
    ) -> np.ndarray:
        '''
        Description
        -----------
        Squeezes all parameters into a 2D numpy ndarray based on 
        the provided order. If no order is provided, parameters are
        squeezed in inherent dictionary order.

        Parameters
        ----------
        ```
        parameters : Dict[str, VParameter]
        ```
        The parameters to squeeze
        ```
        order : Optional[List[str]] = None
        ```
        The order in which to squeeze the parameters
        
        Returns
        -------
        ```
        return squeezed_array -> ndarray
        ```
        A numpy ndarray of the parameters squeezed in the specified order
        '''
        if order is not None:
            self._validate_order(parameters, order)
            return np.column_stack(tuple(parameters[elem].value.tolist() for elem in order))
        else:
            return np.column_stack(tuple(elem.value for elem in parameters.values()))
        
    def _squeeze_params_to_df(
            self,
            parameters: Dict[str, VParameter],
            order: Optional[List[str]] = None
    ) -> pd.DataFrame:
        '''
        Description
        -----------
        Squeezes all parameters into a 2D pandas DataFrame based on the
        provided order. If no order is provided, parameters are
        squeezed in inherent dictionary order.

        Parameters
        ----------
        ```
        parameters: Dict[str, VParameter]
        ```
        The parameters to squeeze
        ```
        order: Optional[List[str]] = None
        ```
        The order in which to squeeze the parameters

        Returns
        -------
        ```
        return df -> pd.DataFrame
        ```
        A pandas DataFrame of the parameters squeezed in a specifc order
        '''
        df = pd.DataFrame(self._squeeze_params_to_ndarray(parameters, order))
        if order is not None: df.columns = order
        else: df.columns = list(parameters.keys())
        return df

class VBaseDriverCommand(BaseDriverCommand, _CommandVectorBase):
    '''
    Description
    -----------
    Vectorized implementation of a BaseDriverCommand object.
    Command function will have a list of kwargs applied (or tensor) and the function 
    will be responsible for unpacking the arguments and applying them to a vectorizable 
    function. Parameters are not updated from their default value but instead are validated
    Workflow global output for this command will be a list of outputs corresponding to each input in order.

    Attributes
    ----------
    ```
    uuid : str
    ```
    UUID of system associated with the driver command
    ```
    fn : Union[Callable, str]
    ```
    Function or function name accessed via import of python callable object to be used during
    `__call__` override of driver command object
    ```
    module : Optional[str] = None
    ```
    Name of the module to import if accessing function from that module
    ```
    package : Optional[str] = None
    ```
    Name of the package to be used during module import if needed
    ```
    has_return : Optional[bool] = False
    ```
    True if function has a return value and it should be accessed, False otherwise

    Methods
    -------
    ```
    def __call__(
        self, 
        wf_globals: Dict[str, Any] = None, 
        save_vars: Dict[str, str] = None, 
        list_kwargs: List = None
    ) -> Dict:
    ```
    Call the driver command function with list_kwargs applied to the function.
    The funciton is responsible for parsing the input and performing operations in a vectorized manner.
    '''
    # VBaseDriverCommand public attributes
    parameters: Dict[str, VParameter] | None = {}
    # TODO implement functionality for squeezing parameters together
    # so that the function signature does not have to take in 
    squeeze: bool = False
    order: List | None = None

    # VBaseDriverCommand private attributes
    _vector_type: VectorLike = PrivateAttr(default=None)
    _vector_length: int = PrivateAttr(default=None) 

    # Validators
    @model_validator(mode='after')
    def validate(self) -> 'VBaseDriverCommand':
        '''
        Decorators
        ----------
        ```
        @classmethod
        @root_validator(skip_on_failure=True)
        ```
        Description
        -----------
        Root validator for VBaseDriverCommand. This calls BaseDriverCommand
        root validator and then checks to ensure that all parameter vectors 
        are of the same length on their primary axis. 

        Parameters
        ----------
        ```
        values : Dict[str, Any]
        ```
        The attributes of VBaseDriverCommand to validate
        
        Raises
        ------
        ```
        raise ValidationError
        ```
        Raises a validation error if any of the fields validate
        internal validation functions
        
        Return
        -------
        ```
        return values : Dict[str, Any]
        ```
        Upon successful validation, returns model values with any modifications
        '''
        super().validate()
        super()._validate_param_len(self.parameters)
        return self
    
    # Private attribute init
    def _init_vector_type(self):
        '''
        Description
        -----------
        Initalizes VBaseDriverCommand's private attributes for vector 
        typing and vector length. Called in class method `_init_private_attributes()`.
        '''
        if self.parameters is not None and len(self.parameters.keys()) > 0:
            first_v_type = type(list(self.parameters.values())[0].value)
            if not all(type(elem.value) is first_v_type for elem in self.parameters.values()):
                if self.squeeze: raise TypeError
            else:
                self._vector_type = first_v_type
        self._vector_length = self._validate_param_len()
    
    def _init_private_attributes(self):
        '''
        Description
        -----------
        Initalizes VBaseDriverCommand's private attributes via superclass
        initilization for BaseDriverCommand attributes and then vector
        specific attributes
        '''
        super()._init_private_attributes()
        self._init_vector_type()

    # Base vector override
    def _validate_param_len(self) -> int:
        return super()._validate_param_len(self.parameters)
    
    def _validate_order(self) -> None:
        return super()._validate_order(self.parameters, self.order)
    
    def _squeeze_params_to_ndarray(self) -> np.ndarray:
        return super()._squeeze_params_to_ndarray(self.parameters, self.order)
    
    def _squeeze_params_to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self._squeeze_params_to_ndarray())
        if self.order is not None: df.columns = self.order
        else: df.columns = list(self.parameters.keys())
        return df
        
class VInteractiveResultDriverCommand(InteractiveResultDriverCommand, _CommandVectorBase):
    '''
    Description
    -----------
    Driver command for interactive user input of command results. When calling 
    this command via `__call__()` override, the user will be shown command 
    parameters in the form of a pandas DataFrame and then prompted to input
    vectorized results for that command.

    For example, you can run VInteractiveResultDriverCommand's in the following manner:

    ```python
    >>> run_interactive_lc = VInteractiveResultDriverCommand(
            name="Interactive_LC_Run",
            parameters={
                "solvent_gradient": SolventGradient(),  # Assume it is set to [4,5,6,7,8]
                "flow_rate": FlowRate(),                # Assume it is set to [100,200,300,400,500]
                "injection_volume": InjectionVolume()   # Assume it is set to [1,1,1,1,1]
            },
            uuid="LCMS",
            result_vars=["resolution"],
        )
    >>> run_interactive_lc()
    ```
    ```text
    >>>     solvent_gradient    flow_rate   injection_volume
        0   4.0                 100.0       1.0
        1   5.0                 200.0       1.0
        2   6.0                 300.0       1.0
        3   7.0                 400.0       1.0
        4   8.0                 500.0       1.0
    >>> Input value(s) for 'resoluion': 1.0,1.0,1.0,1.0,1.0
    ```
    
    Note
    ----
    It is important to note that all parameters must be the same length on their primary
    access or else an AssertionError will be thrown.

    It is important to note that the resulting vector input by the user must 
    be the same length on their primary access as the parameters 
    or else an Assertion Error will be thrown.

    Attributes
    ----------
    ```
    uuid : str
    ```
    UUID of system associated with the driver command
    ```
    fn : Optional[Callable] = None
    ```
    Optional helper function to be run during interactive input function call
    ```
    has_return : Optional[bool] = False
    ```
    True if function has a return value and it should be accessed, False otherwise

    Methods
    -------
    ```
    def __call__(
        self, 
        wf_globals: Dict[str, Any] = None, 
        save_vars: Dict[str, str] = None, 
        list_kwargs: List = None
    ) -> Dict:
    ```
    Calls the internal function for printing out vectorized parameters
    as a pandas DataFrame and then prompts the user for vectorized input
    correspond to result varaibles.
    '''
    parameters: Dict[str, VParameter] | None = {}
    # TODO implement functionality for squeezing parameters together
    # so that the function signature does not have to take in 
    squeeze: bool = False
    order: List | None = None

    _vector_type: VectorLike = PrivateAttr(default=None)
    _vector_length: int = PrivateAttr(default=None) 

    # Validators
    @model_validator(mode='after')
    def validate(self) -> 'VInteractiveResultDriverCommand':
        '''
        Decorators
        ----------
        ```
        @classmethod
        @root_validator(skip_on_failure=True)
        ```
        Description
        -----------
        Root validator for VBaseDriverCommand. This calls BaseDriverCommand
        root validator and then checks to ensure that all parameter vectors 
        are of the same length on their primary axis. 

        Parameters
        ----------
        ```
        values : Dict[str, Any]
        ```
        The attributes of VBaseDriverCommand to validate
        
        Raises
        ------
        ```
        raise ValidationError
        ```
        Raises a validation error if any of the fields validate
        internal validation functions
        
        Return
        -------
        ```
        return values : Dict[str, Any]
        ```
        Upon successful validation, returns model values with any modifications
        '''
        super().validate()
        self._init_private_attributes()
        super()._validate_param_len(self.parameters)
        return self

    # Private attribute init
    def _init_vector_type(self):
        '''
        Description
        -----------
        Initalizes VInteractiveResultDriverCommand's private attributes for vector 
        typing and vector length. Called in class method `_init_private_attributes()`.
        '''
        if self.parameters is not None and len(self.parameters.keys()) > 0:
            first_v_type = type(list(self.parameters.values())[0].value)
            if not all(type(elem.value) is first_v_type for elem in self.parameters.values()):
                if self.squeeze: raise TypeError
            else:
                self._vector_type = first_v_type
        self._vector_length = self._validate_param_len()

    def _init_private_attributes(self):
        def obtain_v_results(**kwargs) -> Dict[str, torch.Tensor]:
            '''
            Description
            -----------
            Function using during __call__() override to receive batch
            results interactively. Results must be provided as a list
            '''
            # Print dataframe of current parameters
            print(self._squeeze_params_to_df(), flush=True)
            
            # Print helper function if needed
            if self.fn is not None:
                print(f"Helper Function {self.fn(**kwargs)}")

            # Save off results as specified in result_vars
            results = {}
            for elem in self.result_vars:
                # Eval input using ast
                result = ast.literal_eval(
                    input(f"Input value for result {elem}")
                )

                # Ensure we are getting list
                try: result = list(result)
                except: 
                    raise TypeError(f"Result {result} cannot be casted to list")
                
                # Update vector length if needed
                self._vector_length = self._validate_param_len()

                if len(result) != self._vector_length:
                    raise AssertionError(f"Result length {len(result)} does not match vector length {self._vector_length}")
                
                results[elem] = torch.Tensor(result).to(float)
            return results
        
        # Assign function and vector type
        self._function = obtain_v_results  
        self._init_vector_type()

    # Base vector override
    def _validate_param_len(self) -> int:
        return super()._validate_param_len(self.parameters)
    
    def _validate_order(self) -> None:
        return super()._validate_order(self.parameters, self.order)
    
    def _squeeze_params_to_ndarray(self) -> np.ndarray:
        return super()._squeeze_params_to_ndarray(self.parameters, self.order)
    
    def _squeeze_params_to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self._squeeze_params_to_ndarray())
        if self.order is not None: df.columns = self.order
        else: df.columns = list(self.parameters.keys())
        return df