import unittest, sys, inspect, json
import warnings
warnings.filterwarnings('ignore')
from pydantic import BaseModel
from typing import (
    Callable, 
    List, 
    Dict
)
sys.path.append("../../JSON")
from ASPIRE_LINQX.core.parameter.base import ParameterModel
from ASPIRE_LINQX.core.command.base import BaseDriverCommand
from ASPIRE_LINQX.core.workflow.base import BaseDriverWorkflow

from ASPIRE_LINQX.core.optimizer.base import BaseObjectiveFunction
'''
from models.optimizer.evotorch import (
    EvotorchProblemModel,
    EvotorchGeneticAlgorithmModel,
    EvotorchGuassianMutationModel,
    EvotorchMultiPointCrossOverModel
)
'''
import torch
from evotorch import Problem
from evotorch.algorithms import SNES, SteadyStateGA
from evotorch.operators import (
    SimulatedBinaryCrossOver,
    GaussianMutation
)
from evotorch.logging import StdOutLogger

class TestBaseObjectiveFunction(unittest.TestCase):
    def setUp(self) -> None:
        # Set up models for basic float and float list for summation
        self.base_float_model = ParameterModel(
            name="Base Float",
            data_type="float",
            default=0.0
        )
        self.value_model = ParameterModel(
            name="Value",
            data_type="float",
            upper_limit=1.0,
            lower_limit=-1.0,
            is_list=True,
            default=[0.0, 0.0, 0.0, 0.0]
        )

        # Build objects from parameter models
        Value = self.value_model.to_param()
        BaseFloat = self.base_float_model.to_param()

        self.Value = Value
        
        f1 = BaseFloat()
        f2 = BaseFloat()

        # Update certian parameters to read from global vars
        f1.from_var = True
        f1.var_name = "pos_output"
        f2.from_var = True
        f2.var_name = "neg_output"

        # Define some dummy microservice functions
        def sum_pos(values): return {"output": torch.sum(torch.Tensor(values))}
        def sum_neg(values): return {"output": -1*torch.sum(torch.Tensor(values))}
        def sum_two(num1, num2): return {"sum": num1+num2}

        # Put those functions into driver command objects
        self.sum_pos_command = BaseDriverCommand(
            name="sum_positive",
            microservice='calculator',
            parameters={"values": self.value_model},
            uuid="slurm",
            has_return=True,
            fn=sum_pos,
        )
        self.sum_neg_command = BaseDriverCommand(
            name="sum_negative",
            microservice='calculator',
            parameters={"values": self.value_model},
            uuid="slurm",
            has_return=True,
            fn=sum_neg,
        )
        self.sum_two_command = BaseDriverCommand(
            name="sum_two",
            microservice='calculator',
            parameters={"num1": self.base_float_model, "num2": self.base_float_model},
            uuid="slurm",
            has_return=True,
            fn=sum_two,
        )

        self.sum_two_command.set_var_name('num1', 'pos_output')
        self.sum_two_command.set_var_name('num2', 'neg_output')

        # print(self.sum_two_command._parameters)

        # Build a workflow from our commands
        self.workflow = BaseDriverWorkflow(
            name="sum_workflow",
            commands=[
                self.sum_pos_command,
                self.sum_neg_command,
                self.sum_two_command
            ]
        )

        # This is how we will save command output to globals during workflow execution
        self.list_save_vars = [{"output": "pos_output"}, {"output": "neg_output"}, {"sum": "sum_output"}]

    def test_workflow_init(self):
        # This list provides kwargs to commands in order
        list_kwargs = [
            {"values": [0.5,0.5,0.5,0.5]},
            {"values": [-0.5,-0.5,-0.5,-0.5]},
            {}
        ]
        # Execute the workflow (if there is an error this is bad)
        log = self.workflow.exec(
            list_kwargs=list_kwargs,
            list_save_vars=self.list_save_vars
        )

    def test_obj_kwarg_conversion(self):
        # Define our tensor here
        x = torch.Tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

        # Order of vector positions to kwargs components and get them from the tensor
        order_kwargs = [
            {"values": (0,4)}, # Should get [0.1, 0.2, 0.3, 0.4]
            {"values": (4,8)}, # Should get [0.5, 0.6, 0.7, 0.8]
            {}
        ]

        # Get the list of the arguments
        list_kwargs = BaseObjectiveFunction._assign_kwarg_list(
            order_kwargs=order_kwargs,
            x=x
        )

        # Get the sets of kwargs as a list of floats in the most complex way possible (round for comparison)
        kwargs_1 = list(map(lambda x: round(x, ndigits=4), list(map(float, list_kwargs[0]['values']))))
        kwargs_2 = list(map(lambda x: round(x, ndigits=4), list(map(float, list_kwargs[1]['values']))))

        self.assertListEqual(kwargs_1, [0.1, 0.2, 0.3, 0.4])
        self.assertListEqual(kwargs_2, [0.5, 0.6, 0.7, 0.8])

    def test_obj_fn_init(self):
        # Define our tensor
        x = torch.Tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

        # Determine the order of the arugments from the input tensor
        order_kwargs = [
            {"values": (0,4)}, # Should get [0.1, 0.2, 0.3, 0.4]
            {"values": (4,8)}, # Should get [0.5, 0.6, 0.7, 0.8]
            {}
        ]
        
        # Our fitness value is the final sum
        fitness_criteria = ["sum_output"]

        # Build our objective function
        obj_fn = BaseObjectiveFunction(
            name="sum_fn",
            workflow=self.workflow,
            order_kwargs=order_kwargs,
            list_save_vars=self.list_save_vars,
            fitness_criteria=fitness_criteria
        )

        # Sum: 0.1, 0.2, 0.3, 0.4 = 1
        # Negitive Sum: 0.5, 0.6, 0.7, 0.8 = -2.6
        # Fitness value should be -1.6

        # Call it on our tensor, use almost equal because of tensor rounding differeneces
        fitness_value = obj_fn(x=x)
        self.assertAlmostEqual(float(fitness_value), -1.6, delta=0.0001)

    def test_obj_fn_ga_optimize(self):
        # Determine the order of the arugments from the input tensor
        order_kwargs = [
            {"values": (0,4)},
            {"values": (4,8)},
            {}
        ]
        
        # Our fitness value is the final sum
        fitness_criteria = ["sum_output"]

        # Build our objective function
        obj_fn = BaseObjectiveFunction(
            name="sum_fn",
            workflow=self.workflow,
            order_kwargs=order_kwargs,
            list_save_vars=self.list_save_vars,
            fitness_criteria=fitness_criteria
        )

        # Define our problem:
        #   - Maximize objective function output ("max")
        #   - Bounds are value lower limit and upper limit
        #   - Solution length is a vector of length 8
        problem = Problem(
            objective_sense="max",
            objective_func=obj_fn,
            bounds=(self.value_model.lower_limit, self.value_model.upper_limit),
            solution_length=8
        )

        # Create a genetic algorithm, add guassian mutation operator
        searcher = SteadyStateGA(problem=problem, popsize=1)
        searcher.use(GaussianMutation(problem=problem, stdev=0.1))

        # Log the output of the searcher and run
        num_generations = 1000
        searcher.run(num_generations=num_generations)

        # See if we found the global optimum, global opt is 8 -> (1,1,1,1,-1,-1,-1,-1)
        # Note that if the number of generations is too low, this may fail based on algorithm used
        self.assertListEqual(
            list(map(float, searcher.get_status_value('pop_best'))), 
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        )

if __name__ == "__main__":
    unittest.main()