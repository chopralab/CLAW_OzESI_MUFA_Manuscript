import unittest, sys
sys.path.append("../../JSON")

from pydantic import BaseModel, ValidationError

from models.parameter.base import ParameterModel
from models.parameter.vectorized import VParameterModel, VParameter

import numpy as np
import pandas as pd
import torch

class TestVParameterModel(unittest.TestCase):
    
    def setUp(self) -> None:
        self.ndarray_model = VParameterModel(
            name="Ndarray",
            data_type="float",
            vector_type="ndarray",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )
        self.pd_series_model = VParameterModel(
            name="Series",
            data_type="float",
            vector_type="series",
            default=[1,2,3,4],
            upper_limit=10,
            lower_limit=0,
        )
        self.pd_df_model = VParameterModel(
            name="DataFrame",
            data_type="float",
            vector_type="dataframe",
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
            lower_limit=0
        )

        self.Ndarray = self.ndarray_model.to_param()
        self.Series = self.pd_series_model.to_param()
        self.DataFrame = self.pd_df_model.to_param()
        self.Tensor = self.tensor_model.to_param()

        self.allowed_ndarray_model = VParameterModel(
            name="Ndarray",
            data_type="float",
            vector_type="ndarray",
            default=[5,6,7,8],
            allowed_values=[5,6,7,8]
        )
        self.allowed_pd_series_model = VParameterModel(
            name="Series",
            data_type="float",
            vector_type="series",
            default=[5,6,7,8],
            allowed_values=[5,6,7,8]
        )
        self.allowed_pd_df_model = VParameterModel(
            name="DataFrame",
            data_type="float",
            vector_type="dataframe",
            default=[5,6,7,8],
            allowed_values=[5,6,7,8]
        )
        self.allowed_tensor_model = VParameterModel(
            name="Tensor",
            data_type="float",
            vector_type="tensor",
            default=[5,6,7,8],
            allowed_values=[5,6,7,8]
        )

        self.AllowedNdarray = self.allowed_ndarray_model.to_param()
        self.AllowedSeries = self.allowed_pd_series_model.to_param()
        self.AllowedDataFrame = self.allowed_pd_df_model.to_param()
        self.AllowedTensor = self.allowed_tensor_model.to_param()

    def test_v_param_model_init(self):
        # Test class heierarchy init
        self.assertIsInstance(self.ndarray_model, BaseModel)
        self.assertIsInstance(self.ndarray_model, ParameterModel)
        self.assertIsInstance(self.ndarray_model, VParameterModel)

        self.assertIsInstance(self.pd_series_model, BaseModel)
        self.assertIsInstance(self.pd_series_model, ParameterModel)
        self.assertIsInstance(self.pd_series_model, VParameterModel)

        self.assertIsInstance(self.pd_df_model, BaseModel)
        self.assertIsInstance(self.pd_df_model, ParameterModel)
        self.assertIsInstance(self.pd_df_model, VParameterModel)

        self.assertIsInstance(self.tensor_model, BaseModel)
        self.assertIsInstance(self.tensor_model, ParameterModel)
        self.assertIsInstance(self.tensor_model, VParameterModel)

        # Test private attributes init
        self.assertEqual(self.ndarray_model._vector_type, np.ndarray)
        self.assertEqual(self.pd_series_model._vector_type, pd.Series)
        self.assertEqual(self.pd_df_model._vector_type, pd.DataFrame)
        self.assertEqual(self.tensor_model._vector_type, torch.Tensor)

        # Test default vector type casting
        self.assertIsInstance(self.ndarray_model.default, np.ndarray)
        self.assertIsInstance(self.pd_series_model.default, pd.Series)
        self.assertIsInstance(self.pd_df_model.default, pd.DataFrame)
        self.assertIsInstance(self.tensor_model.default, torch.Tensor)

    def test_v_param_obj_init(self):
        # Create parameters
        array_1 = self.Ndarray()
        series_1 = self.Series()
        dataframe_1 = self.DataFrame()
        tensor_1 = self.Tensor()

        # Test class heierarchy init
        self.assertIsInstance(array_1, BaseModel)
        self.assertIsInstance(array_1, VParameter)

        self.assertIsInstance(series_1, BaseModel)
        self.assertIsInstance(series_1, VParameter)

        self.assertIsInstance(dataframe_1, BaseModel)
        self.assertIsInstance(dataframe_1, VParameter)

        self.assertIsInstance(tensor_1, BaseModel)
        self.assertIsInstance(tensor_1, VParameter)
        
        # Test default value init
        self.assertTrue((array_1.value == np.array([1,2,3,4])).all())
        self.assertTrue((series_1.value == pd.Series([1,2,3,4])).all())
        self.assertTrue((dataframe_1.value == pd.DataFrame([1,2,3,4])).all().all())
        self.assertTrue((tensor_1.value == torch.Tensor([1,2,3,4])).all())

    def test_v_param_op_overload(self):
        # Create parameters
        array_1 = self.Ndarray()
        series_1 = self.Series()
        dataframe_1 = self.DataFrame()
        tensor_1 = self.Tensor()

        # Test __eq__
        self.assertTrue((array_1 == np.array([1,2,3,4])).all())
        self.assertTrue((series_1 == pd.Series([1,2,3,4])).all())
        self.assertTrue((dataframe_1 == pd.DataFrame([1,2,3,4])).all().all())
        self.assertTrue((tensor_1 == torch.Tensor([1,2,3,4])).all())

        # Test __ne__
        self.assertListEqual(
            list(array_1 != np.array([1,2,3,5])), 
            [False, False, False, True]
        )
        self.assertListEqual(
            list(series_1 != pd.Series([1,2,3,5])),
            [False, False, False, True]
        )
        self.assertListEqual(
            list((dataframe_1 != pd.DataFrame([1,2,3,5]))[0].values),
            [False, False, False, True]
        )
        self.assertListEqual(
            list(tensor_1 != torch.Tensor([1,2,3,5])),
            [False, False, False, True]
        )
        
        # Test __lt__
        self.assertTrue((array_1 < 5).all())
        self.assertTrue((series_1 < 5).all())
        self.assertTrue((dataframe_1 < 5)[0].all())
        self.assertTrue((tensor_1 < 5).all())

        # Test __gt__
        self.assertFalse((array_1 > 5).all())
        self.assertFalse((series_1 > 5).all())
        self.assertFalse((dataframe_1 > 5)[0].all())
        self.assertFalse((tensor_1 > 5).all())

        # Test __le__
        self.assertTrue((array_1 <= 4).all())
        self.assertTrue((series_1 <= 4).all())
        self.assertTrue((dataframe_1 <= 4)[0].all())
        self.assertTrue((tensor_1 <= 4).all())

        # Test __ge__
        self.assertTrue((array_1 >= 1).all())
        self.assertTrue((series_1 >= 1).all())
        self.assertTrue((dataframe_1 >= 1)[0].all())
        self.assertTrue((tensor_1 >= 1).all())

        # Test __add__
        self.assertListEqual(list(array_1 + 5), [6,7,8,9])
        self.assertListEqual(list(series_1 + 5), [6,7,8,9])
        self.assertListEqual(list((dataframe_1 + 5)[0]), [6,7,8,9])
        self.assertListEqual(list(tensor_1 + 5), [6,7,8,9])

        # Test __sub__
        self.assertListEqual(list(array_1 - 5), [-4,-3,-2,-1])
        self.assertListEqual(list(series_1 - 5), [-4,-3,-2,-1])
        self.assertListEqual(list((dataframe_1 - 5)[0]), [-4,-3,-2,-1])
        self.assertListEqual(list(tensor_1 - 5), [-4,-3,-2,-1])

        # Test __mul__
        self.assertListEqual(list(array_1 * 5), [5,10,15,20])
        self.assertListEqual(list(series_1 * 5), [5,10,15,20])
        self.assertListEqual(list((dataframe_1 * 5)[0]), [5,10,15,20])
        self.assertListEqual(list(tensor_1 * 5), [5,10,15,20])

        # Test __truediv__
        self.assertListEqual(list(array_1 / 5), [.2,.4,.6,.8])
        self.assertListEqual(list(series_1 / 5), [.2,.4,.6,.8])
        self.assertListEqual(list((dataframe_1 / 5)[0]), [.2,.4,.6,.8])
        self.assertListEqual(list(tensor_1 / 5), [.2,.4,.6,.8])

        # Test __pow__
        self.assertListEqual(list(array_1 ** 2), [1,4,9,16])
        self.assertListEqual(list(series_1 ** 2), [1,4,9,16])
        self.assertListEqual(list((dataframe_1 ** 2)[0]), [1,4,9,16])
        self.assertListEqual(list(tensor_1 ** 2), [1,4,9,16])

        # Reassign because we would have issues with limits
        array_1.value = np.array([2,2,2,2])
        series_1.value = pd.Series([2,2,2,2])
        dataframe_1.value = pd.DataFrame([2,2,2,2])
        tensor_1.value = torch.Tensor([2,2,2,2])

        # Test __isub__
        array_1 -= 1; series_1 -= 1; dataframe_1 -= 1; tensor_1 -= 1

        self.assertListEqual(list(array_1.value), [1,1,1,1])
        self.assertListEqual(list(series_1.value), [1,1,1,1,])
        self.assertListEqual(list(dataframe_1.value[0]), [1,1,1,1])
        self.assertListEqual(list(tensor_1.value), [1,1,1,1])

        # Test __iadd__
        array_1 += 1; series_1 += 1; dataframe_1 += 1; tensor_1 += 1

        self.assertListEqual(list(array_1.value), [2,2,2,2])
        self.assertListEqual(list(series_1.value), [2,2,2,2])
        self.assertListEqual(list(dataframe_1.value[0]), [2,2,2,2])
        self.assertListEqual(list(tensor_1.value), [2,2,2,2])

        # Test __imul__
        array_1 -= 1; series_1 -=1; dataframe_1 -= 1; tensor_1 -= 1
        array_1 *= 3; series_1 *=3; dataframe_1 *= 3; tensor_1 *= 3

        self.assertListEqual(list(array_1.value), [3,3,3,3])
        self.assertListEqual(list(series_1.value), [3,3,3,3])
        self.assertListEqual(list(dataframe_1.value[0]), [3,3,3,3])
        self.assertListEqual(list(tensor_1.value), [3,3,3,3])

        # Test __ipow__
        array_1 -= 1; series_1 -= 1; dataframe_1 -= 1; tensor_1 -= 1
        array_1 **= 2; series_1 **= 2; dataframe_1 **= 2; tensor_1 **= 2

        self.assertListEqual(list(array_1.value), [4,4,4,4])
        self.assertListEqual(list(series_1.value), [4,4,4,4])
        self.assertListEqual(list(dataframe_1.value[0]), [4,4,4,4])
        self.assertListEqual(list(tensor_1.value), [4,4,4,4])

    def test_v_param_validation(self):
        # Create parameters
        array_1 = self.Ndarray()
        series_1 = self.Series()
        dataframe_1 = self.DataFrame()
        tensor_1 = self.Tensor()

        # Test incompatable data type validation
        with self.assertRaises(ValidationError): array_1.value = 5
        with self.assertRaises(ValidationError): series_1.value = None
        with self.assertRaises(ValidationError): dataframe_1.value = "value"
        with self.assertRaises(ValidationError): tensor_1.value = 1.5

        # Test upper limit validation
        with self.assertRaises(ValueError):
            array_1.value = np.array([1,2,3,100])
        self.assertListEqual(list(array_1.value), [1,2,3,4])

        with self.assertRaises(ValueError):
            series_1.value = pd.Series([1,2,3,100])
        self.assertListEqual(list(series_1.value), [1,2,3,4])
        
        with self.assertRaises(ValueError):
            dataframe_1.value = pd.DataFrame([1,2,3,100])
        self.assertListEqual(list(dataframe_1.value[0].values), [1,2,3,4])

        with self.assertRaises(ValueError):
            tensor_1.value = torch.Tensor([1,2,3,100])
        self.assertListEqual(list(tensor_1.value), [1,2,3,4])

        # Test lower limit validation
        with self.assertRaises(ValueError):
            array_1.value = np.array([-100,2,3,5])
        self.assertListEqual(list(array_1.value), [1,2,3,4])

        with self.assertRaises(ValueError):
            series_1.value = pd.Series([-100,2,3,5])
        self.assertListEqual(list(array_1.value), [1,2,3,4])
        
        with self.assertRaises(ValueError):
            dataframe_1.value = pd.DataFrame([-100,2,3,5])
        self.assertListEqual(list(dataframe_1.value[0].values), [1,2,3,4])

        with self.assertRaises(ValueError):
            tensor_1.value = torch.Tensor([-100,2,3,5])
        self.assertListEqual(list(tensor_1.value), [1,2,3,4])

        # Test allowed value validation
        array_2 = self.AllowedNdarray()
        series_2 = self.AllowedSeries()
        dataframe_2 = self.AllowedDataFrame()
        tensor_2 = self.AllowedTensor()

        with self.assertRaises(ValueError):
            array_2.value = np.array([5,6,7,10])

        with self.assertRaises(ValueError):
            series_2.value = pd.Series([5,6,7,10])
        
        with self.assertRaises(ValueError):
            dataframe_2.value = pd.DataFrame([5,6,7,10])

        with self.assertRaises(ValueError):
            tensor_2.value = torch.Tensor([5,6,7,10])

    def test_v_param_vector_casting(self):
        array_1 = self.Ndarray()
        series_1 = self.Series()
        dataframe_1 = self.DataFrame()
        tensor_1 = self.Tensor()

        # Test ndarray successful casting
        array_1.value = [5,6,7,8]
        self.assertIsInstance(array_1.value, np.ndarray)
        self.assertListEqual(list(array_1.value), [5,6,7,8])

        array_1.value = pd.Series([1,3,5,9])
        self.assertIsInstance(array_1.value, np.ndarray)
        self.assertListEqual(list(array_1.value), [1,3,5,9])

        array_1.value = pd.DataFrame([[1,2,3,4],[5,6,7,8]])
        self.assertIsInstance(array_1.value, np.ndarray)
        self.assertListEqual(list(array_1.value[0]), [1,2,3,4])
        self.assertListEqual(list(array_1.value[1]), [5,6,7,8])

        array_1.value = torch.Tensor([2,4,6,8])
        self.assertIsInstance(array_1.value, np.ndarray)
        self.assertListEqual(list(array_1.value), [2,4,6,8])

        # Test ndarray out of bounds cast
        with self.assertRaises(ValueError):
            array_1.value = [9,10,11,12]
        self.assertListEqual(list(array_1.value), [2,4,6,8])

        # Test Series successful casting
        series_1.value = [5,6,7,8]
        self.assertIsInstance(series_1.value, pd.Series)
        self.assertListEqual(list(series_1.value), [5,6,7,8])

        series_1.value = np.array([2,4,6,8])
        self.assertIsInstance(series_1.value, pd.Series)
        self.assertListEqual(list(series_1.value), [2,4,6,8])

        series_1.value = torch.Tensor([1,3,5,9])
        self.assertIsInstance(series_1.value, pd.Series)
        self.assertListEqual(list(series_1.value), [1,3,5,9])

        # Test Series casting out of bounds
        with self.assertRaises(ValueError):
            series_1.value = [8,9,10,11]
        self.assertListEqual(list(series_1.value), [1,3,5,9])

        # Test DataFrame successful casting
        dataframe_1.value = [5,6,7,8]
        self.assertIsInstance(dataframe_1.value, pd.DataFrame)
        self.assertListEqual(list(dataframe_1.value[0]), [5,6,7,8])

        dataframe_1.value = np.array([1,3,5,9])
        self.assertIsInstance(dataframe_1.value, pd.DataFrame)
        self.assertListEqual(list(dataframe_1.value[0]), [1,3,5,9])

        dataframe_1.value = pd.Series([2,4,6,8])
        self.assertIsInstance(dataframe_1.value, pd.DataFrame)
        self.assertListEqual(list(dataframe_1.value[0]), [2,4,6,8])

        dataframe_1.value = torch.Tensor([1,2,8,9])
        self.assertIsInstance(dataframe_1.value, pd.DataFrame)
        self.assertListEqual(list(dataframe_1.value[0]), [1,2,8,9])

        # Test Dataframe casting out of bounds
        with self.assertRaises(ValueError):
            series_1.value = [8,9,10,11]
        self.assertListEqual(list(dataframe_1.value[0]), [1,2,8,9])

        # Test Tensor successful casting
        tensor_1.value = [5,6,7,8]
        self.assertIsInstance(tensor_1.value, torch.Tensor)
        self.assertListEqual(list(tensor_1.value), [5,6,7,8])

        tensor_1.value = np.array([1,3,5,9])
        self.assertIsInstance(tensor_1.value, torch.Tensor)
        self.assertListEqual(list(tensor_1.value), [1,3,5,9])

        tensor_1.value = pd.Series([2,4,6,8])
        self.assertIsInstance(tensor_1.value, torch.Tensor)
        self.assertListEqual(list(tensor_1.value), [2,4,6,8])

        # Test Tensor casting out of bounds
        with self.assertRaises(ValueError):
            tensor_1.value = pd.Series([8,9,10,11])
        self.assertListEqual(list(tensor_1.value), [2,4,6,8])

    def test_v_param_value_casting(self):
        array_1 = self.Ndarray()
        series_1 = self.Series()
        dataframe_1 = self.DataFrame()
        tensor_1 = self.Tensor()

        # Test successful ndarray cast
        array_1.value = ["2","4","6","8"]
        self.assertIsInstance(array_1.value, np.ndarray)
        self.assertListEqual(list(array_1.value), [2,4,6,8])
        self.assertEqual(str(array_1.value.dtype), "float64")

        # Test failed ndarray cast
        with self.assertRaises(ValidationError):
            array_1.value = ["a", "b", "c", "d"]
        self.assertListEqual(list(array_1.value), [2,4,6,8])
        self.assertEqual(str(array_1.value.dtype), "float64")

        # Test successful series cast
        series_1.value = ["2","4","6","8"]
        self.assertIsInstance(series_1.value, pd.Series)
        self.assertListEqual(list(series_1.value), [2,4,6,8])
        self.assertEqual(str(series_1.value.dtype), "float64")

        # Test failed series cast
        with self.assertRaises(ValidationError):
            series_1.value = ["a", "b", "c", "d"]
        self.assertListEqual(list(series_1.value), [2,4,6,8])
        self.assertEqual(str(series_1.value.dtype), "float64")

        # Test successful dataframe cast
        dataframe_1.value = [["2","4","6","8"],["1","3","5","9"]]
        self.assertIsInstance(dataframe_1.value, pd.DataFrame)
        self.assertListEqual(list(dataframe_1.value.iloc[0]), [2,4,6,8])
        self.assertListEqual(list(dataframe_1.value.iloc[1]), [1,3,5,9])
        self.assertListEqual(list(dataframe_1.value.dtypes), 4*["float64"])

        # Test failed dataframe cast
        with self.assertRaises(ValidationError):
            dataframe_1.value = [["2","4","6","8"],["a","b","c","d"]]

        # Test successful tensor cast
        tensor_1.value = ["2", "4", "6", "8"]
        self.assertIsInstance(tensor_1.value, torch.Tensor)
        self.assertListEqual(list(tensor_1.value), [2,4,6,8])
        self.assertEqual(str(tensor_1.value.dtype), "torch.float64")

        tensor_1.value = np.array(["1", "3", "5", "7"])
        self.assertIsInstance(tensor_1.value, torch.Tensor)
        self.assertListEqual(list(tensor_1.value), [1,3,5,7])

        # Test failed tensor cast
        with self.assertRaises(ValidationError):
            tensor_1.value = ["a", "b", "c", "d"]

if __name__ == "__main__":
    unittest.main()