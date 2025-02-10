import unittest, sys
sys.path.append("../../JSON")

from pydantic import BaseModel, ValidationError
from typing import Dict

from models.command.base import BaseDriverCommand
from models.command.vectorized import VBaseDriverCommand, _CommandVectorBase
from models.workflow.vectorized import VBaseDriverWorkflow
from models.workflow.base import BaseDriverWorkflow

import numpy as np
import pandas as pd
import torch

class TestVBaseDriverWorkflow(unittest.TestCase):

    def test_v_driver_workflow_init(self):
        def test():
            pass

        command = BaseDriverCommand(
            name="base_command",
            uuid="local",
            fn=test
        )

        v_command = VBaseDriverCommand(
            name="v_base_command",
            uuid="local",
            fn=test
        )

        BaseDriverWorkflow(
            name="base_workflow",
            commands=[v_command, command]
        )

        VBaseDriverWorkflow(
            name="v_base_workflow",
            commands=[v_command]
        )

        with self.assertRaises(ValidationError):
            VBaseDriverWorkflow(
                name="v_base_workflow",
                commands=[v_command, command]
            )

if __name__ == "__main__":
    unittest.main()