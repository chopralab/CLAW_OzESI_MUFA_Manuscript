from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Tuple

class ObjectiveFunctionSchemaV1(BaseModel):
    name: str
    order_kwargs: List[Dict[str, Tuple[int, int]]] = Field(
        description="""A list of which positions of the fitness function input Pytorch tensor to extract for 
        specific kwargs of specific commands. 
            
        For example see the following python code block:
        ```
        input_tensor = [10,20,30,40,50,60,70,80,90,100]
        order_kwargs = [
            {"arg_1_var_1": (0,5), "arg_1_var_2": (6,8)},
            {"arg_2_var_1": (8,8), "arg_2_var_2": (9,9)},
        ]
        # After kwarg assignment
        list_kwargs = [
            {"arg_1_var_1": [10,20,30,40,50], "arg_1_var_2": [70,80]}
            {"arg_2_var_1": 90, "arg_2_var_2": 100}
        ]
        """
    )
    fitness_criteria: List[str] = Field(
        description="""A list of varibles to take from the workflow global which will
        determine the fitness of the objective function.
        """
    )