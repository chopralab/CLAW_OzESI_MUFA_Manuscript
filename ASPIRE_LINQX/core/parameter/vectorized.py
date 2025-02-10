from pydantic import (
    ConfigDict, 
    field_validator, 
    PrivateAttr, 
)
from typing import (
    Callable,
    List, 
    Dict,
    Tuple, 
    Union, 
    Any, 
    Literal, 
    Optional, 
    Type,
    Type
)
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from ASPIRE_LINQX.core.parameter.base import ParameterModel, Parameter

# Define VectorLike type, union of ndarray, Series, DataFrame, and Tensor
VectorLike = Union[ndarray, Series, DataFrame, Tensor]

class VParameter(Parameter):
    '''
    Description
    -----------
    This is a shell parameter class that extends Parameter

    It is used to provide a uniform superclass when dynamically creating 
    vectorized parameters using the create_model() method

    Attributes
    ----------
    ```
    value : VectorLike
    ```
    The value of the parameter, must be a vector type
    ```
    desc : Optional[str] = ""
    ```
    A description of the parameter
    ```
    from_var: Optional[bool] = False
    ```
    A flag to denote wheter the parameter should be read from a workflow global
    varaible at runtime during workflow execution. See `commands.py` and `workflow.py`
    for more details.
    ```
    var_name : Optional[str] = ""
    ```
    The name of the varaible which will have its value assigned to the parameter
    upon workflow execution.
    ```
    upper_limit : Optional[Union[int, str, float]] = None
    ```
    The upper limit of the parameter, must match the data type.
    Cannot be less than the lower limit.
    ```
    lower_limit : Optional[Union[int, str, float]] = None
    ```
    The lower limit of the parameter, must match the data type.
    Cannot be greater than the upper limit.
    '''
    # Model config
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Public attributes
    value: VectorLike

class VParameterModel(ParameterModel):
    '''
    Description
    -----------
    This class is used to define a custom model for a vectorized parameter to be use in LINQX
    infrastructure components. After initilizing the model, use the to_param()
    method to create a class which represents the vectorized parameter that was defined.
    From there, individual objects of that parameter can be created and reference
    in other LINQX infrastrucutre components.

    For example, to create a vectorized parameter called voltage which ranges between 60.0v and 240.0v
    and has a default value of [60, 100, 140, 180, 220]:

    ```python
    # Define the voltage model
    voltage_model = ParameterModel(
        name="Voltage",
        data_type="float",
        vector_type="ndarray"
        upper_limit=240.0,
        lower_limit=60.0,
        default=[60,100,140,180,220],
        description="Voltage in volts"
    )

    # Build the Voltage class
    Voltage = voltage_model.to_param() # Subclass of VParameter
    
    # Build voltage objects
    v1 = Voltage() # -> has value of [60,100,140,180,220]
    v2 = Voltage(value=[100,100,100,100,100]) # -> has value of [100,100,100,100,100]
    v3 = Voltage(value=[200,200,200,200,200]) # -> has value of [200,200,200,200,200]

    # It will prevent invalid objects from being build
    v_invalid = Voltage(value=[300,300,300,300,300]) # This will raise an error
    
    # Operators are supported
    v1 = v1 + 20
    v1.value # -> should be [80,120,160,200,240]
    v1 == [80,80,80,80,80]# -> should be [True,False,False,False,False]
    ```

    It is also possible to define parameters which can have dynamically assigned 
    values during workflow execution. This is done by defining the parameter to 
    read from a specific varaible. See the example below:

    ```python
    # Assign v1 to read from 'voltage_1' dynamically
    v1.from_var = True
    v1.var_name = "voltage_1"

    # Assign v2 to read from 'voltage_2' dynamically
    v2.from_var = True
    v2.var_name = "voltage_2"
    ```

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the parameter model
    ```
    data_type : Literal["str", "int", "float"]
    ```
    The data type of the parameter model. Supports only primitives.
    Must be a string.
    ```
    vector_type : Literal["ndarray", "series", "dataframe", "tensor"]
    ``` 
    The vector type of the parameter model. Must be a string.
    ```
    precision : Optional[Union[Literal[-1], PositiveInt]] = -1
    ```
    Precision to round floats to if applicable (-1 is infinite precision)
    ```
    upper_limit : Optional[Union[int, str, float]] = None
    ```
    The upper limit of the parameter, must match the data type.
    Cannot be less than the lower limit.
    ```
    lower_limit : Optional[Union[int, str, float]] = None
    ```
    The lower limit of the parameter, must match the data type.
    Cannot be greater than the upper limit.
    ```
    allowed_value : Optional[List[Union[int, str, float]]] = []
    ```
    A list of allowed values to restrict the parameter.
    Must match the data type of the parameter
    ```
    is_optional : Optional[bool] = False
    ```
    A flag to denote whether the parameter is required or optional.
    ```
    is_list : Optional[bool] = False
    ``` 
    A flag to denote whether the parameter is a value or a list.
    ```
    default : Optional[Union[List, VectorLike]] = None
    ```
    A default value for the parameter, must match type and obey limits.
    ```
    from_var : Optional[bool] = False
    ```
    A flag to denote wheter the parameter should be read from a workflow global
    varaible at runtime during workflow execution. See `commands.py` and `workflow.py`
    for more details.
    ```
    var_name : Optional[str] = ""
    ```
    The name of the varaible which will have its value assigned to the parameter
    upon workflow execution.
    ```
    desc : Optional[str] = ""
    ```
    A description of the parameter

    Methods
    -------
    ```
    def to_param() -> type[VParameter]
    ```
    Builds a VParameter subclass based on the VParamModel's
    specifications of the vectorized parameter
    '''
    # Model Config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Vector specific attributes
    vector_type: Literal["ndarray", "series", "dataframe", "tensor"]
    default: List | VectorLike | None = None

    # Private vector specific attributes
    _vector_type: Type[VectorLike] | None = PrivateAttr(default=None)

    def _cast_default(self) -> None:
        '''
        '''
        # Cast for list
        if isinstance(self.default, list):
            if self._vector_type is ndarray:
                self.default = np.array(self.default)
            if self._vector_type is Series:
                self.default = pd.Series(self.default)
            if self._vector_type is DataFrame:
                self.default = pd.DataFrame(self.default)
            if self._vector_type is Tensor:
                self.default = torch.Tensor(self.default)

        # Cast for data type
        if isinstance(self.default, ndarray):
            self.default = self.default.astype(self._data_type)
        if isinstance(self.default, Series):
            self.default = self.default.astype(self._data_type)
        if isinstance(self.default, DataFrame):
            self.default = self.default.astype(self._data_type)
        if isinstance(self.default, Tensor):
            self.default = self.default.to(self._data_type)

    def _validate_default(self) -> None:
        '''
        '''
        # Ensure the default value is of the correct vector type
        if not isinstance(self.default, self._vector_type):
            raise TypeError
        
        # Ensure all values of the default vector are valid
        if self._vector_type is DataFrame:
            if self.upper_limit is not None and (self.default > self.upper_limit).any().any(): raise ValueError
            if self.lower_limit is not None and (self.default < self.lower_limit).any().any(): raise ValueError
            if len(self.allowed_values) > 0 and not all(elem in self.allowed_values for elem in np.stack(self.default.values).flatten()):
                    raise ValueError(f"Pandas Dataframe: {self.default} has values not in allowed values: {self.allowed_values}")
        else:
            if self.upper_limit is not None and bool((self.default > self.upper_limit).any()): raise ValueError
            if self.lower_limit is not None and bool((self.default < self.lower_limit).any()): raise ValueError
            if self._vector_type is ndarray:
                if len(self.allowed_values) > 0 and not all(elem in self.allowed_values for elem in self.default.flatten()):
                    raise ValueError(f"Ndarray: {self.default} has values not in allowed values: {self.allowed_values}")
            if self._vector_type is Series:
                if len(self.allowed_values) > 0 and not all(elem in self.allowed_values for elem in np.stack(self.default.values).flatten()):
                    raise ValueError(f"Pandas Dataframe: {self.default} has values not in allowed values: {self.allowed_values}")
            if self._vector_type is Tensor:
                if len(self.allowed_values) > 0 and not all(elem in self.allowed_values for elem in self.default.detach().cpu().numpy().flatten()):
                    raise ValueError(f"Torch tensor: {self.default} has values not in allowed values: {self.allowed_values}")

    def _init_private_attributes(self) -> None:
        '''
        Description
        -----------
        Initalize the data type of the class via super class call. 
        Initalize the vector type to class based on vector type provided during init.
        Initalize the object of the parameter class to VParameter instead of parameter.
        '''
        # Initalize data type
        super()._init_private_attributes()

        # Initalize vector type
        vector_typing = {
            "ndarray": ndarray,
            "series": Series, 
            "dataframe": DataFrame,
            "tensor": Tensor
        }
        self._vector_type = vector_typing[self.vector_type]

        # Initalize base object class
        self._obj_base_class = VParameter

    def _assign_value(self) -> Tuple:
        '''
        Description
        -----------
        Assigns default value and data type to constructed VParameter class.
        This will be used in the create model function in the super class call
        for dynamic model creation.


        Return
        ------
        ```
        return (self._vector_type, self.default) -> (type(VectorLike), VectorLike)
        ```
        Tuple assignment for the value field of the create model function
        '''
        return (self._vector_type, self.default)

    def _assign_numpy_validators(self, validator_dict: Dict[str, Callable[..., Any]]) -> None:
        '''
        Description
        -----------
        Assign additional validators for numpy ndarray data type.

        - Adds validator for casting from list, Series, and Tensor data types 
        (list, Pandas Series, and Torch Tensor) to a numpy array. 
        - Adds validator for ensuring that all values are in the allowed value list

        Parameters
        ----------
        ```
        validator_dict: Dict[str, Callable]
        ```
        The validator dictionary used for the VParameter class 
        which contains all validators
        '''

        # Validator for casting to numpy ndarray
        @field_validator('value', mode='before')
        @classmethod
        def cast_pre_validate(cls, v: Any):
            if isinstance(v, list): 
                return np.array(v).astype(self._data_type)
            if isinstance(v, Series) or isinstance(v, DataFrame):
                return v.to_numpy().astype(self._data_type) 
            if isinstance(v, Tensor):
                return v.cpu().detach().numpy().astype(self._data_type)
            return v
        validator_dict["cast_pre_validate"] = cast_pre_validate

        # Validator for checking allowed values of numpy ndarray
        if len(self.allowed_values) > 0:
            @field_validator('value')
            @classmethod
            def validate_allowed_values(cls, v: ndarray) -> ndarray:
                if not all(elem in self.allowed_values for elem in v.flatten()):
                    raise ValueError(f"Ndarray: {v} has values not in allowed values: {self.allowed_values}")
                return v
            validator_dict["validate_allowed_values"] = validate_allowed_values
        
    def _assign_pandas_validators(self, validator_dict: Dict[str, Callable[..., Any]]) -> Dict[str, Callable]:
        '''
        Description
        -----------
        Assign additional validators for pandas DataFrame and Series data type

        - Adds validators for assuring that values of the DataFrame or Series
        are in the allowed value list

        Parameters
        ----------
        ```
        validator_dict : Dict[str, Callable]
        ```
        The validator dictionary used for the VParameter class 
        which contains all validators
        '''

        if len(self.allowed_values) > 0:
            @field_validator('value')
            @classmethod
            def validate_allowed_values(cls, v: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
                if not all(elem in self.allowed_values for elem in np.stack(v.values).flatten()):
                    raise ValueError(f"Pandas Series/Dataframe: {v} has values not in allowed values: {self.allowed_values}")
                return v
            validator_dict["validate_allowed_values"] = validate_allowed_values

    def _assign_series_validators(self, validator_dict: Dict[str, Callable[..., Any]]) -> Dict[str, Callable[..., Any]]:
        '''
        Description
        -----------
        Assign additional validators for pandas Series data type

        - Adds validators for casting from list and ndarray to Series data type

        Parameters
        ----------
        ```
        validator_dict : Dict[str, Callable]
        ```
        The validator dictionary used for the VParameter class 
        which contains all validators
        '''
        @field_validator('value', mode='before')
        @classmethod
        def cast_pre_validate(cls, v: Any):
            if isinstance(v, list):
                return pd.Series(v).astype(self._data_type)
            if isinstance(v, ndarray) or isinstance(v, Tensor):
                return pd.Series(v.tolist()).astype(self._data_type)
            return v
        validator_dict["cast_pre_validate"] = cast_pre_validate

    def _assign_dataframe_validators(self, validator_dict: Dict[str, Callable[..., Any]]) -> Dict[str, Callable[..., Any]]:
        '''
        Description
        -----------
        Assign additional validators for pandas DataFrame data type

        - Adds validators for casting from list, ndarray, and Series 
        to DataFrame data type
        - Adds validators for upper and lower limit validation of 
        DataFrame (as general validators for other vector types 
        do not work)

        Parameters
        ----------
        ```
        validator_dict : Dict[str, Callable]
        ```
        The validator dictionary used for the VParameter class 
        which contains all validators
        '''

        # Validator for casting to DataFrame
        @field_validator('value', mode='before')
        @classmethod
        def cast_pre_validate(cls, v: Any):
            if isinstance(v, list):
                return pd.DataFrame(v).astype(self._data_type)
            if isinstance(v, ndarray) or isinstance(v, Tensor):
                return pd.DataFrame(v.tolist()).astype(self._data_type)
            if isinstance(v, Series):
                return v.to_frame().astype(self._data_type)
            return v
        validator_dict["cast_pre_validate"] = cast_pre_validate

        # Validator for validating DataFrame upper limit
        @field_validator('value')
        @classmethod
        def validate_upper_limit(cls, v: DataFrame) -> DataFrame:
            if (v > self.upper_limit).any().any():
                raise ValueError
            return v

        # Validator for validating DataFrame lower limit
        @field_validator('value')
        @classmethod
        def validate_lower_limit(cls, v: DataFrame) -> DataFrame:
            if (v < self.lower_limit).any().any():
                raise ValueError
            return v

        validator_dict["validate_upper_limit"] = validate_upper_limit
        validator_dict["validate_lower_limit"] = validate_lower_limit

    def _assign_tensor_validators(self, validator_dict: Dict[str, Callable[..., Any]]) -> Dict[str, Callable[..., Any]]:
        '''
        Description
        -----------
        Assign additional validators for torch Tensor data type

        - Adds validators for casting from list and ndarray
        to Tensor data type
        - Adds validators for ensure Tensor values are in allowed
        value list if set

        Parameters
        ----------
        ```
        validator_dict : Dict[str, Callable]
        ```
        The validator dictionary used for the VParameter class 
        which contains all validators
        '''

        # Validator for casting to Tensor data type
        @field_validator('value', mode='before')
        @classmethod
        def cast_pre_validate(cls, v: Any):
            if isinstance(v, list):
                return torch.Tensor(np.array(v).astype(self._data_type)).to(self._data_type)
            if isinstance(v, ndarray) or isinstance(v, Series):
                return torch.Tensor(v.astype(self._data_type)).to(self._data_type)
            return v
        validator_dict["cast_pre_validate"] = cast_pre_validate

        # Validator for ensuring all Tensor values are in allowed values
        if len(self.allowed_values) > 0:
            @field_validator('value')
            @classmethod
            def validate_allowed_values(cls, v: Tensor) -> Tensor:
                if not all(elem in self.allowed_values for elem in v.detach().cpu().numpy().flatten()):
                    raise ValueError(f"Torch tensor: {v} has values not in allowed values: {self.allowed_values}")
                return v
            validator_dict["validate_allowed_values"] = validate_allowed_values

    def _assign_model_validators(self) -> Dict[str, Callable[..., Any]]:
        '''
        Description
        -----------
        Private class method for assigning model validators. This method is overridden 
        from the base class to account for custom validators for vectorized data types.
        Generic upper and lower limit validators are the same for all data types. 
        More specific validators for specific data types are added later.

        Returns
        -------
        ```
        return validator_dict : Dict[str, Callable]
        ```
        The validator dictionary used for the VParameter class 
        which contains all validators
        '''
        # Create the dictionary of custom validators
        validator_dict = {}

        # Assign generic upper and lower limit validators (works for all current data types)
        if self.upper_limit is not None:
            @field_validator('value')
            @classmethod
            def validate_upper_limit(cls, v: Union[ndarray, DataFrame, Tensor]) -> Union[ndarray, DataFrame, Tensor]:
                if bool((v > self.upper_limit).any()):
                    raise ValueError
                return v
            validator_dict["validate_upper_limit"] = validate_upper_limit
        if self.lower_limit is not None:
            @field_validator('value')
            @classmethod
            def validate_lower_limit(cls, v: Union[ndarray, DataFrame, Tensor]) -> Union[ndarray, DataFrame, Tensor]:
                if bool((v < self.lower_limit).any()):
                    raise ValueError
                return v
            validator_dict["validate_lower_limit"] = validate_lower_limit

        # Add custom validators for specific data types
        if self._vector_type is ndarray:
            self._assign_numpy_validators(validator_dict)
        if self._vector_type is Series:
            self._assign_series_validators(validator_dict)
            self._assign_pandas_validators(validator_dict)
        if self._vector_type is DataFrame:
            self._assign_pandas_validators(validator_dict)
            self._assign_dataframe_validators(validator_dict)
        if self._vector_type is Tensor:
            self._assign_tensor_validators(validator_dict)

        return validator_dict