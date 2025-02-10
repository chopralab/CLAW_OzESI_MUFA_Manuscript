from pydantic import (
    BaseModel, 
    root_validator,
    PrivateAttr, 
    Field,
)
from typing import (
    Dict, 
    Any,
    List,
    Tuple, 
    Optional,
    Union,
    Callable,
    Literal
)
from ASPIRE_LINQX.core.workflow.base import BaseDriverWorkflow
from ASPIRE_LINQX.core.parameter.base import Parameter
from ASPIRE_LINQX.core.optimizer.base import BaseProblemModel, BaseSearcherModel

import torch
from evotorch import Problem
from evotorch.algorithms import (
    CMAES,
    PyCMAES,
    CEM,
    Cosyne,
    MAPElites,
    PGPE,
    SNES,
    XNES,
    GeneticAlgorithm,
    SteadyStateGA,
    SearchAlgorithm
)
from evotorch.operators import (
    Operator,
    GaussianMutation,
    MultiPointCrossOver
)
from evotorch.core import ObjectiveSense, BoundsPairLike

class EvotorchProblemModel(BaseProblemModel):
    '''
    Not implemented
    ---------------
    Base model for an EvoTorch optimizaiton problem using LINQX workflows
    '''
    objective_goals: ObjectiveSense
    
    def _get_bounds(self) -> BoundsPairLike:
        '''
        Get EvoTorch support bounds for all parameters
        '''
        bounds = [[],[]]
        for elem in self.input_params:
            if isinstance(list, elem.value):
                # If the parameter is a list (vector), we need to enumerate the 
                # bounds for each position 
                bounds[0].append(len(elem.value)*[elem.lower_limit])
                bounds[1].append(len(elem.value)*[elem.upper_limit])
            else:
                # If is is not a vector, append limits for single value
                bounds[0].append(elem.lower_limit)
                bounds[1].append(elem.upper_limit)
        
        # Convert bounds to tuples and return
        bounds[0] = tuple(bounds[0])
        bounds[1] = tuple(bounds[1])
        return tuple(bounds)

    def to_obj(self) -> Problem:
        # Get EvoTorch formatted bounds based on input parameters
        bounds = self._get_bounds()
        # Create an EvoTorch formatted problem
        problem = Problem(
            objective_sense=self.objective_goals,
            objective_func=self.objective_function,
            bounds=bounds,
            solution_length=self.solution_length
        )
        return problem

class EvotorchOperatorModel(BaseModel):
    '''
    Not implemented
    ---------------
    Base model for all EvoTorch operators
    '''
    problem: Problem

class EvotorchGuassianMutationModel(EvotorchOperatorModel):
    '''
    Not implemented
    ---------------
    Custom model for EvoTorch guassian mutation operator
    '''
    stdev: float
    mutation_probability: Optional[float] = None

    def to_obj(self) -> GaussianMutation:
        return GaussianMutation(
            problem=self.problem,
            stdev=self.stdev,
            mutation_probability=self.mutation_probability,
        )

class EvotorchMultiPointCrossOverModel(EvotorchOperatorModel):
    '''
    Not implemented
    ---------------
    Custom model for EvoTorch one point cross over operator
    '''
    tournament_size: int
    obj_index: Optional[int] = None
    num_points: int = Field(1, ge=1)
    num_children: Optional[int] = None
    cross_over_rate: Optional[float] = None

    def to_obj(self) -> MultiPointCrossOver:
        return MultiPointCrossOver(
            problem=self.problem,
            tournament_size=self.tournament_size,
            obj_index=self.obj_index,
            num_points=self.num_points,
            num_children=self.num_children,
            cross_over_rate=self.cross_over_rate,
        )

class EvotorchSearcherModel(BaseSearcherModel):
    '''
    Not implemented
    ---------------
    Base model for an EvoTorch implemented search algorithm

    problem must be an instance of EvoTorch `Problem` class
    '''
    problem: Problem

class EvotorchGeneticAlgorithmModel(EvotorchSearcherModel):
    '''
    Not implemented
    ---------------
    Base model for EvoTorch genetic algorithm
    '''
    operators: Optional[List[Operator]]
    popsize: int
    elitist: bool = True
    re_evaluate: bool = True
    re_evaluate_parents_first: Optional[bool] = None

    def to_obj(self) -> GeneticAlgorithm:
        return GeneticAlgorithm(
            problem=self.problem,
            operators=self.operators,
            popsize=self.popsize,
            elitist=self.elitist,
            re_evaluate=self.re_evaluate,
            re_evaluate_parents_first=self.re_evaluate_parents_first,
        )