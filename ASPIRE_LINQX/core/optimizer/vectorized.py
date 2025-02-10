from typing import (
    Dict, 
    Any,
    List,
    Tuple, 
)
from ASPIRE_LINQX.core.workflow.vectorized import VBaseDriverWorkflow
from ASPIRE_LINQX.core.optimizer.base import BaseObjectiveFunction
import torch

class VBaseObjectiveFunction(BaseObjectiveFunction):
    '''
    Description
    -----------
    Objective function class designed to wrap vectorized workflows for 
    running closed-loop optimization. This class has its `__call__()`
    method overridden to take in a tensor (workflow parameters), run
    the vectorized workflow, and then return a tensor (fitness values).

    Attributes
    ----------
    ```
    workflow : VBaseDriverWorkflow
    ```
    A workflow of executable vectorized driver commands to be run as a fitness fuction
    ```
    order_kwargs: Iterable[Dict[str, Tuple[int, int]]]
    ```
    A list of which positions of the fitness function input tensor to extract for 
    specific kwargs of specific commands. 
        
    For example:
    ```python
    #               0  1  2  3  4  5  6  7  8  9
    input_tensor = [[10,20,30,40,50,60,70,80,90,100],
                    [10,20,30,40,50,60,70,80,80,100],
                    [10,20,30,40,50,60,70,80,80,100]]
    order_kwargs = [
        {"arg_1_var_1": (0,5), "arg_1_var_2": (6,8)},
        {"arg_2_var_1": (8,8), "arg_2_var_2": (9,9)},
    ]
    # After kwarg assignment
    list_kwargs = [
        {
            "arg_1_var_1": [[10,20,30,40,50],
                            [10,20,30,40,50],
                            [10,20,30,40,50]]
            "arg_1_var_2": [[70,80],
                            [70,80],
                            [70,80]]
        },
        {
            "arg_2_var_1": [[90],
                            [90],
                            [90]]
            "arg_2_var_2": [[100],
                            [100],
                            [100]]
        }
    ]
    workflow.exec(list_kwargs,*)
    ```
    ```
    list_save_vars: List[Union[Dict[str,str], None]]
    ```
    A list of command outputs to save off to the global dictionary during workflow execution
    ```
    fitness_criteria: List[str]
    ```
    A list of values to extract from the global dictionary to return as fitness values for the objective function.

    Note: This must match with the number of objectives in the problem

    Methods
    -------
    ```
    def __call__(self, x: torch.Tensor, print_log: bool=False) -> torch.Tensor
    ```
    Calls the assigned workflow with kwargs taken from the input tensor x. Kwargs
    are taken from the tensor based on `kwargs_order` attribute. Call returns a
    tensor of fitness values based on keys in the `fitness_criteria` attribute and
    values in the workflow globals.
    '''
    workflow: VBaseDriverWorkflow

    @staticmethod
    def _assign_kwarg_list(
        order_kwargs: List[Dict[str, Tuple[int, int]]],
        x: torch.Tensor
    ) -> List[Dict[str, Any]]:
        '''
        Decorators
        ----------
        ```
        @staticmethod
        ```

        Description
        -----------
        Assigns input tensor to workflow kwargs as specified by 
        the input order. See the definition of order_kwargs in the
        class description for a detailed example.

        Parameters
        ----------
        ```
        order_kwargs : List[Dict[str, Tuple[int, int]]]
        ```
        A list of dictionaries which contain the varaible name 
        and the position of the input tensor to assign to the varaible
        ```
        x : torch.Tensor
        ```
        The tensor input from the optimization algorithm

        Returns
        -------
        ```
        return list_kwargs : List[Dict[str, Any]]
        ```
        A list of kwargs to be applied to workflow commands
        '''
        list_kwargs = []
        for elem in order_kwargs: 
            # For each kwarg ordering make a new set of arguments
            kwargs = {}
            for key, value in elem.items():
                # Assign each key to a section of the tensor
                kwargs[key] = torch.flatten(x[:,value[0]:value[1]])
            list_kwargs.append(kwargs)
        return list_kwargs
    
    def __call__(self, x: torch.Tensor, print_log: bool = False) -> torch.Tensor:
        '''
        Description
        -----------
        Calls the assigned workflow with kwargs taken from the input tensor x. Kwargs
        are taken from the tensor based on `kwargs_order` attribute. Call returns a
        tensor of fitness values based on keys in the `fitness_criteria` attribute and
        values in the workflow globals.

        Attributes
        ----------
        ```
        x : torch.Tensor
        ```
        The input tensor to the workflow
        ```
        print_log : bool = False
        ```
        Set to true to print a log of the workflow execution

        Return
        ------
        ```
        return fitness_values : Tensor
        ```
        The fitness values of the workflow correspond to the input tensor
        '''
        # Clear globals from a previous run
        self.workflow.clear_wf_globals()

        # Assign kwargs for workflow exec
        list_kwargs = type(self)._assign_kwarg_list(order_kwargs=self.order_kwargs, x=x)
        # list_kwargs = BaseObjectiveFunction._assign_kwarg_list(order_kwargs=self.order_kwargs, x=x)

        # Execute workflow
        log = self.workflow.exec(
            list_kwargs=list_kwargs,
            list_save_vars=self.list_save_vars
        )
        if print_log: print(log)
        
        # Get fitness values based on criteria
        fitness_values = [self.workflow.wf_globals[key] for key in self.fitness_criteria]

        # Ensure all fitness values are tensor
        if not all(isinstance(elem, torch.Tensor) for elem in fitness_values):
            raise TypeError
        
        # Return fitness values
        return torch.transpose(torch.vstack(tuple(elem for elem in fitness_values)), 0, 1)