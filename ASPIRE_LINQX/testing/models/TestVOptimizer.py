import unittest, sys
sys.path.append("../../JSON")

from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
from typing import Dict

from models.command.base import BaseDriverCommand
from models.command.vectorized import VBaseDriverCommand, _CommandVectorBase
from models.workflow.vectorized import VBaseDriverWorkflow
from models.workflow.base import BaseDriverWorkflow

import numpy as np
import pandas as pd
import torch

class TestVBaseObjectiveFunction(unittest.TestCase):

    def setUp(self) -> None:
        pass