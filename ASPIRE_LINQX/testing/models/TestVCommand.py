import unittest, sys
sys.path.append("../../JSON")

from pydantic import BaseModel, ValidationError
from typing import Dict

from models.parameter.base import ParameterModel
from models.parameter.vectorized import VParameterModel
from models.command.base import BaseCommand, BaseDriverCommand
from models.command.vectorized import VBaseDriverCommand, _CommandVectorBase

import numpy as np
import pandas as pd
import torch

class TestCommandVectorBase(unittest.TestCase):

    def setUp(self) -> None:
        self.ndarray_model = VParameterModel(
            name="Ndarray",
            data_type="float",
            vector_type="ndarray",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )
        self.series_model = VParameterModel(
            name="Series",
            data_type="float",
            vector_type="series",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )
        self.tensor_model = VParameterModel(
            name="Tensor",
            data_type="float",
            vector_type="tensor",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )

        self.Ndarray = self.ndarray_model.to_param()
        self.Series = self.series_model.to_param()
        self.Tensor = self.tensor_model.to_param()

        self.arr1 = self.Ndarray(value=[1,1,1,1])
        self.series2 = self.Series(value=[2,2,2,2])
        self.tensor3 = self.Tensor(value=[3,3,3,3])

        self.mixed_parameters = {
            "array_1": self.arr1,
            "array_2": self.series2,
            "array_3": self.tensor3
        }

        self.arr_error = self.Ndarray(value=[4,4,4,4,4])

        self.error_parameters = {
            "array_1": self.arr1,
            "array_2": self.series2,
            "array_3": self.arr_error
        }

    def test_command_vector_base_validate_len(self):
        # Test no error is thrown with valid param lengths
        _CommandVectorBase()._validate_param_len(self.mixed_parameters)

        # Test error is thrown with invalid param lengths
        with self.assertRaises(AssertionError):
            _CommandVectorBase()._validate_param_len(self.error_parameters)

    def test_command_vector_base_squeeze_to_ndarray(self):
        # Call without order argument, should default to 1,2,3
        arr = _CommandVectorBase()._squeeze_params_to_ndarray(
            self.mixed_parameters,
        )
        self.assertListEqual(
            arr.tolist(),
            [
                [1,2,3],
                [1,2,3],
                [1,2,3],
                [1,2,3]
            ]
        )

        # Call with different order than default: 2, 3, 1
        arr = _CommandVectorBase()._squeeze_params_to_ndarray(
            self.mixed_parameters,
            order=["array_2", "array_3", "array_1"]
        )
        self.assertListEqual(
            arr.tolist(),
            [
                [2,3,1],
                [2,3,1],
                [2,3,1],
                [2,3,1]
            ]
        )

    def test_command_vector_base_squeeze_to_df(self):
        df = _CommandVectorBase()._squeeze_params_to_df(
            self.mixed_parameters,
            order=["array_1", "array_3", "array_2"]
        )
        # Test to see if the columns work
        self.assertListEqual(
            df.columns.to_list(),
            ["array_1", "array_3", "array_2"]
        )

        # Test row access
        self.assertListEqual(
            df.iloc[0].to_list(),
            [1,3,2]
        )
        self.assertListEqual(
            df.iloc[1].to_list(),
            [1,3,2]
        )
        self.assertListEqual(
            df.iloc[2].to_list(),
            [1,3,2]
        )
        self.assertListEqual(
            df.iloc[3].to_list(),
            [1,3,2]
        )

        # Test column access
        self.assertListEqual(
            df["array_1"].to_list(),
            [1,1,1,1]
        )
        self.assertListEqual(
            df["array_2"].to_list(),
            [2,2,2,2]
        )
        self.assertListEqual(
            df["array_3"].to_list(),
            [3,3,3,3]
        )

class TestVBaseDriverCommand(unittest.TestCase):
    
    def setUp(self) -> None:
        # Define vectorized paramenters
        self.ndarray_model = VParameterModel(
            name="Ndarray",
            data_type="float",
            vector_type="ndarray",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )
        self.series_model = VParameterModel(
            name="Ndarray",
            data_type="float",
            vector_type="series",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )

        self.base_param_model = ParameterModel(
            name="base_parameter",
            data_type="float"
        )

        self.Ndarray = self.ndarray_model.to_param()
        self.Series = self.series_model.to_param()
        self.BaseParam = self.base_param_model.to_param()
        
        self.x_param = self.Ndarray(value=[1,1,1,1])
        self.y_param = self.Ndarray(value=[2,2,2,2])
        self.z_param = self.Series(value=[5,6,7,8])
        self.non_vector_param = self.BaseParam(value=5)

        self.error_param = self.Series(value=[1,3,5,7,9])

        # Define vectorized sum
        def sum(
                x: np.ndarray,
                y: np.ndarray
            ) -> Dict:
            return {"sum": np.add(x, y)}

        self.sum = sum

        self.ndarray_sum = VBaseDriverCommand(
            name="vector_sum",
            parameters={
                "x": self.x_param,
                "y": self.y_param,
            },
            uuid="local",
            fn=self.sum,
            has_return=True
        )

    def test_v_command_init(self):
        # Test class inheritance
        self.assertIsInstance(self.ndarray_sum, BaseModel)
        self.assertIsInstance(self.ndarray_sum, BaseCommand)
        self.assertIsInstance(self.ndarray_sum, BaseDriverCommand)
        self.assertIsInstance(self.ndarray_sum, VBaseDriverCommand)
        self.assertIsInstance(self.ndarray_sum, _CommandVectorBase)

        # Test private attribute assignment
        self.assertIs(self.ndarray_sum._vector_type, np.ndarray)
        self.assertEqual(self.ndarray_sum._vector_length, 4)

    def test_v_command_invalid_init(self):
        # This method is invalid because we need the function arguments
        # to match the command parameters
        with self.assertRaises(KeyError):
            VBaseDriverCommand(
                name="invalid_sum",
                parameters={
                    "x": self.x_param,
                    "z": self.y_param
                },
                uuid="local",
                fn=self.sum,
                has_return=True
            )

        # We should see a validation error when we assign a non_vectorized command
        with self.assertRaises(ValidationError):
            VBaseDriverCommand(
                name="invalid_param",
                parameters={
                    "x": self.x_param,
                    "y": self.non_vector_param
                },
                uuid="local",
                fn=self.sum,
                has_return=True
        )
            
        with self.assertRaises(ValidationError):
            VBaseDriverCommand(
                name="invalid_param_length",
                parameters={
                    "x": self.x_param,
                    "y": self.error_param
                },
                uuid="local",
                fn=self.sum,
                has_return=True
            )

    def test_v_command_call(self):
        # Call with default args
        self.assertListEqual(
            list(self.ndarray_sum()["sum"]),
            [3,3,3,3]
        )
        
        # Call with valid args
        result = self.ndarray_sum(
            x=[1,3,5,9],
            y=[2,4,6,8]
        )
        self.assertListEqual(
            list(result["sum"]),
            [3,7,11,17]
        )

        # Check parameter update
        self.assertListEqual(
            list(self.ndarray_sum["x"]),
            [1,3,5,9]
        )
        self.assertListEqual(
            list(self.ndarray_sum["y"]),
            [2,4,6,8]
        )

        # Call with invalid args
        with self.assertRaises(ValidationError):
            result = self.ndarray_sum(
                x=[1,3,5,20],
                y=[2,4,6,8]
            )

        # Check parameters did not update
        self.assertListEqual(
            list(self.ndarray_sum["x"]),
            [1,3,5,9]
        )
        self.assertListEqual(
            list(self.ndarray_sum["y"]),
            [2,4,6,8]
        )

if __name__ == "__main__":
    unittest.main()