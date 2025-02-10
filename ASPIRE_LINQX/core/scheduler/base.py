import json, copy
from pydantic import (
    BaseModel,
    ConfigDict, 
    BaseConfig, 
    model_validator,
    model_serializer,
)
from pydantic_core import PydanticCustomError
from typing import (
    List, 
    Dict, 
    Union, 
    Tuple,
    Any,
)
from ASPIRE_LINQX.models.parameter.base import (
    ParameterModel, 
    Parameter
)

class _BaseScheduleModel(BaseModel):
    '''
    Description
    -----------
    Base model for scheduling classes. This class is not designed 
    to be used for scheduling but instead is designed to implement
    core functionality of other scheduling classes which extend
    this class.
    '''
    # Model configuration
    model_config = ConfigDict(
        validate_assignment=True, 
        arbitrary_types_allowed=True,
    )

    # _BaseScheduleModel public attributes
    name: str
    template:  Dict[str, Any]

    def _validate_template(self) -> None:
        '''
        Not implemented in the base class.
        '''
        raise NotImplementedError
    
    def _load_from_mapping(self) -> None:
        '''
        Not implemented in the base class.
        '''
        raise NotImplementedError
    
    @model_validator(mode='after')
    def validate(self) -> '_BaseScheduleModel':
        '''
        Decorators
        ----------
        ```
        @classmethod
        @root_validator
        ```

        Description
        -----------
        Validation function for BastScheduleTemplate class.

        Raises
        ------
        ```
        raise ValidationError
        ```
        Raises a validation error if any invalid fields are present.

        Returns
        -------
        ```
        return values : BaseSchedulerTemplate | Dict[str, Any]
        ```
        The object or validate values of the object.
        '''
        
        # Ensure that the template is valid
        self._validate_template()

        # Load from mapping if needed
        self._load_from_mapping()

        return self
    
    def keys(self) -> List[str]:
        '''
        Description
        -----------
        Returns the keys of the top level of the scheduling template
        
        Returns
        -------
        ```
        return keys : List[str]
        ```
        The keys of the top level of the template
        '''
        return self.template.keys()
    
    def values(self) -> List[Union['BaseScheduleTemplate', ParameterModel]]:
        '''
        Description
        -----------
        Returns the values of the top level of the scheduling template
        
        Returns
        -------
        ```
        return values : List[BaseScheduleTemplate | ParameterModel]
        ```
        The values of the top level of the template
        '''
        return self.template.values()
    
    def items(self) -> Tuple[List[str], List[Union['BaseScheduleTemplate', ParameterModel]]]:
        '''
        Description
        -----------
        Returns the scheduling templates key-value parings as a iterable tuple

        Returns
        -------
        ```
        return (keys, values) : Tuple[List[str], List[BaseScheduleTemplate | ParameterModel]]
        ```
        The keys and values of the tempalate
        '''
        return (self.keys(), self.values())

    def __setitem__(self, key: str, value: Union['BaseScheduleTemplate', ParameterModel]) -> None:
        '''
        Description
        -----------
        Override of __setitem__(). Sets the provided key in the template to the 
        provided value. Pydantic validation upon assignment is automatically
        performed to ensure the template is still validate after assignment.

        Parameters
        ----------
        ```
        key : str
        ```
        The key of the template to assign
        ```
        value : BaseScheduleTemplate | ParameterModel
        ```
        The value to assign to the template under the provided key

        Raises
        ------
        ```
        raise ValidationError
        ```
        If the provided value is invalid, the root validator will raise 
        a ValidationError prior to assignment.
        '''
        self.template[key] = value

    def __getitem__(self, key: str) -> Union['BaseScheduleTemplate', ParameterModel]:
        '''
        Description
        -----------
        Override of __getitem__(). Returns the template value associated
        with the provided key. If the provided key is not present in the template, 
        an error is raised.

        Parameters
        ----------
        ```
        key : str
        ```
        The key to access in the template

        Raises
        ------
        ```
        raise KeyError
        ```
        Raises a key error if the key is not present in the template

        Returns
        -------
        ```
        return template[key] : BaseScheduleTemplate | ParameterModel
        ```
        The value associated with the template key
        '''
        if key not in self.template.keys():
            raise KeyError(f"Key {key} not found, expected one of {self.template.keys()}")
        return self.template[key]

class BaseScheduleObject(_BaseScheduleModel):
    '''
    Description
    -----------
    Base model for requests used in LINQX endpoint scheduling. This class
    is a recursively defined model which allows for users to create a structured 
    request for scheduling on specific endpoints. 
    
    Users can interact with Parameter objects to modfiy values within allow ranges
    set by the corresponding template. The request can then be written to JSON format
    as needed. See below for an example:

    Attributes
    ----------
    ```
    name : str
    ```
    The name of the schedule object
    ```
    template : Dict[str, BaseScheduleObject | Parameter]
    ```
    The template of the schedule object

    Methods
    -------
    ```
    def keys() -> List[str]:
    ```
    Returns the keys of the template
    ```
    def values() -> List[BaseScheduleTemplate | ParameterModel]:
    ```
    Returns the values of the template
    ```
    def __setitem__(key: str, value: Union[BaseScheduleTemplate, ParameterModel]) -> None:
    ```
    Sets the provided key of the template to the provided value
    ```
    def __getitem__(key: str) -> Union[BaseScheduleTemplate, ParameterModel]:
    ```
    Gets the value of the template at the provided key
    '''

    @model_serializer(when_used='json')
    def serialize_base_schedule_object(self) -> dict:
        '''
        Decorators
        ----------
        ```
        @model_serializer(when_used='json')
        ```
        
        Description
        -----------
        Override standard JSON serializer to return the 
        object template

        Returns
        -------
        ```
        return self.template
        ```
        The template of the object
        '''
        return self.template

    def _validate_template(self) -> None:
        '''
        Decorators
        ----------
        ```
        @staticmethod
        ```

        Description
        -----------
        Validation function for the template field. This function ensures
        that all values of the template are either BaseScheduleObjects
        or Parameters

        Parameters
        ----------
        ```
        template : Dict[str, Any]
        ```
        The template to validate

        Raises
        ------
        ```
        raise TypeError
        ```
        Raises a type error if the template contains fields of 
        incompatible types.
        '''
        if not all(
            isinstance(elem, BaseScheduleObject) 
            or isinstance(elem, Parameter) 
            for elem in self.template.values()
        ):
            raise PydanticCustomError(
                'template_validation_error',
                """
                Expected all template values to be of type 'BaseScheduleObject' or 'Parameter',
                received {template}
                """,
                {'template': [type(elem) for elem in self.template.values()]}
            )
    
    def _load_from_mapping(self) -> None:
        '''
        Mapping loading not currently support for this class.
        ```
        return template
        ```
        '''
        pass

class BaseScheduleTemplate(_BaseScheduleModel):
    '''
    Description
    -----------
    Base model for templates used in LINQX endpoint scheduling. This class
    is a recursively defined model which allows for users to create a structured 
    template for scheduling on specifc endpoints. 
    
    Users can define template parameters data types, ranges, allowed values, etc. 
    based on the ParameterModel class. Users can also recusively define subtemplates
    as needed for greater specificity. See below for an example:


    Attributes
    ----------
    ```
    name : str
    ```
    The name of the schedule template
    ```
    template : Dict[str, BaseScheduleTemplate | ParameterModel]
    ```
    The template of the schedule template

    Methods
    -------
    ```
    def keys() -> List[str]:
    ```
    Returns the keys of the template
    ```
    def values() -> List[BaseScheduleTemplate | ParameterModel]:
    ```
    Returns the values of the template
    ```
    def __setitem__(key: str, value: Union[BaseScheduleTemplate, ParameterModel]) -> None:
    ```
    Sets the provided key of the template to the provided value
    ```
    def __getitem__(key: str) -> Union[BaseScheduleTemplate, ParameterModel]:
    ```
    Gets the value of the template at the provided key
    ```
    def to_obj() -> BaseScheduleObject
    ```
    Builds a BaseScheduleObjects based on the current BaseScheduleTemplate
    '''

    def _validate_template(self) -> None:
        '''
        Decorators
        ----------
        ```
        @staticmethod
        ```

        Description
        -----------
        Validation function for the template field. This function ensures
        that all values of the template are either BaseScheduleTemplates
        or Parameter models

        Parameters
        ----------
        ```
        template : Dict[str, Any]
        ```
        The template to validate

        Raises
        ------
        ```
        raise TypeError
        ```
        Raises a type error if the template contains fields of 
        incompatible types.
        '''
        if not all(
            isinstance(elem, BaseScheduleTemplate) 
            or isinstance(elem, ParameterModel) 
            or isinstance(elem, dict)
            for elem in self.template.values()
        ):
            raise PydanticCustomError(
                'template_validation_error',
                """
                Expected all template values to be of type 'BaseScheduleTemplate' or 'ParameterModel',
                received {template}
                """,
                {'template': [type(elem) for elem in self.template.values()]}
            )
    
    def _load_from_mapping(self) -> None:
        '''
        Decorators
        ----------
        ```
        @staticmethod
        ```
        Description
        -----------
        Helper function to load BaseScheduleTemplate object from args passed
        in as a mapping. This funciton will recursively create mappings down 
        the template until all mappings are converted to corresponding objects.
        If an invalid mapping is found, the execption is caught and the invalid
        mapping is shown in the error.

        Parameters
        ----------
        ```
        template : Dict[str, Any]
        ```
        The template of the base schedule object which will
        be loaded from a mapping.

        Raises
        ------
        ```
        raise ValueError
        ```
        Raises a ValueError if an invalid mapping is encountered

        Returns
        -------
        ```
        template : Dict[str, ParameterModel | BaseScheduleTemplate]
        ```
        The template with all mappings convereted to their corresponding objects
        '''
        for key, value in self.template.items():
            # If any of the template values are dictionaries
            if isinstance(value, dict):
                mapped_value = None 
                # Try to pass mapping to ParameterModel
                try: 
                    mapped_value = ParameterModel(**value)
                except: pass
                # Try to pass mapping to BaseScheduleTemplate
                try: 
                    mapped_value = BaseScheduleTemplate(**value)
                except: pass
                # If neither object construction was successful, raise an error
                if mapped_value is None:
                    raise ValueError(f"Invalid dictionary: {value}")
                # Reassing the key to the object
                else:
                    self.template[key] = mapped_value
        
    @staticmethod
    def _to_obj(name: str, template: Dict[str, Any]) -> BaseScheduleObject:
        '''
        Description
        -----------
        Recursive helper function for converting a BaseSchedulerTemplate's
        template to a BaseSchedule object. 

        - Base case: All values of the template are ParameterModels, 
        convert to Parameters

        - Recursive case: Some value(s) of the template is/are BaseScheduleTemplate's,
        recursive call on the template field of the object.

        Parameters
        ----------
        ```
        name: str
        ```
        The name of the BaseScheduleObject at the current recursive level
        ```
        template : Dict[str, Any]
        ```
        The template that will be converted to a BaseScheduleObject template
        at the current recursive level

        Return
        ------
        ```
        return BaseScheduleObject() : BaseScheduleObject
        ```
        A new BaseScheduleObject at the current recursive level
        '''
        for key, elem in template.items():
            if isinstance(elem, ParameterModel):
                template[key] = elem.to_param()()
            elif isinstance(elem, BaseScheduleTemplate):
                template[key] = BaseScheduleTemplate._to_obj(elem.name, elem.template)
            else: 
                raise TypeError(f"Template value has type of {type(elem)}, expected {BaseScheduleTemplate} or {ParameterModel}")
        return BaseScheduleObject(
            name=name,
            template=template
        )

    def to_obj(self) -> BaseScheduleObject:
        '''
        Description
        -----------
        Creates a BaseScheduleObject based on the current template of
        the BaseScheduleTemplate. See BaseScheduleObject doccumentation
        for details.

        Returns
        -------
        ```
        return BaseScheduleObject() : BaseScheduleObject
        ```
        A new BaseScheduleObject based on the current template.
        '''
        return self._to_obj(self.name, copy.deepcopy(self.template))