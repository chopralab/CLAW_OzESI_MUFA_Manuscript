import unittest, sys
from ASPIRE_LINQX.core.parameter.base import ParameterModel, Parameter
from pydantic import BaseModel
import torch
import json

class TestParameterModel(unittest.TestCase):
    def setUp(self):
        self.voltage_model = ParameterModel(
            name="Voltage",
            data_type="float",
            upper_limit=40,
            lower_limit=10
        )

        self.voltage_obj = self.voltage_model.to_param()
        self.voltage_15 = self.voltage_obj(value=15)
        self.voltage_20 = self.voltage_obj(value=20)

        self.bool_flag = ParameterModel(
            name="Flag",
            data_type='bool',
            default=True
        )
        
        self.flag_obj = self.bool_flag.to_param()

    def test_param_model_init(self):
        ParameterModel(
            name="Voltage",
            data_type="int"
        )
        ParameterModel(
            name="Voltage",
            data_type="int",
            upper_limit=40,
            lower_limit=10
        )
        ParameterModel(
            name="Voltage",
            data_type="float",
            upper_limit=40,
            lower_limit=10
        )
        ParameterModel(
            name="Voltage",
            data_type="float",
            allowed_values=[1,2,3,4,5]
        )
        ParameterModel(
            name="flag",
            data_type='bool',
        )

        self.assertEqual(self.voltage_model.name, "Voltage")
        self.assertEqual(self.voltage_model.data_type, "float")
        self.assertEqual(self.voltage_model._data_type, float)
        self.assertEqual(self.voltage_model.upper_limit, 40)
        self.assertEqual(self.voltage_model.lower_limit, 10)
    
    def test_param_model_to_param(self):
        self.assertIsInstance(self.voltage_15, self.voltage_obj)
        self.assertIsInstance(self.voltage_15, Parameter)
        self.assertIsInstance(self.voltage_15, BaseModel)

        self.assertIsInstance(self.flag_obj(), Parameter)
        self.assertEqual(self.flag_obj(), True)

        self.assertEqual(self.voltage_15.value, 15)
        self.assertEqual(self.voltage_20.value, 20)
        with self.assertRaises(ValueError):
            self.voltage_obj(value=45)
        with self.assertRaises(Exception):
            self.voltage_obj(value="not valid")

    def test_param_binary_overload(self):
        self.assertEqual(self.voltage_15 + 5, 20)
        self.assertEqual(self.voltage_20 - 5, 15)
        self.assertEqual(self.voltage_15 * 2, 30)
        self.assertEqual(self.voltage_20 / 5, 4)
        self.assertEqual(self.voltage_20 ** 2, 400)

    def test_param_comparison_overload(self):
        self.assertEqual(self.voltage_15, 15)
        self.assertNotEqual(self.voltage_15, 20)
        self.assertLess(self.voltage_15, 20)
        self.assertGreater(self.voltage_15, 10)
        self.assertLessEqual(self.voltage_15, 15)
        self.assertGreaterEqual(self.voltage_15, 15)

    def test_param_assignment_overload(self):
        # Test assignment operator overloading
        self.voltage_15 += 5
        self.assertEqual(self.voltage_15, 20)
        self.assertIsInstance(self.voltage_15, Parameter)
        self.voltage_20 -= 8
        self.assertEqual(self.voltage_20, 12)
        self.assertIsInstance(self.voltage_15, Parameter)
        self.voltage_15 *= 2
        self.assertEqual(self.voltage_15, 40)
        self.assertIsInstance(self.voltage_15, Parameter)

        # Make sure validation on assignment works
        with self.assertRaises(Exception):
            self.voltage_15 *= 2
        self.assertEqual(self.voltage_15, 40)
        self.assertIsInstance(self.voltage_15, Parameter)

        with self.assertRaises(Exception):
            self.voltage_15 += 100
        self.assertEqual(self.voltage_15, 40)
        self.assertIsInstance(self.voltage_15, Parameter)

        with self.assertRaises(Exception):
            self.voltage_15 -= 100
        self.assertEqual(self.voltage_15, 40)
        self.assertIsInstance(self.voltage_15, Parameter)

    def test_param_as_tensor(self):
        list_model = ParameterModel(
            name="BaseList",
            data_type="float",
            is_list=True,
            default=[0,0,0,0]
        )
        ListModel = list_model.to_param()
        test_list = ListModel()
        test_tensor = torch.Tensor([0.1, 0.2, 0.3, 0.4])
        test_list.value = test_tensor
        test_list.value = [round(number=elem, ndigits=4) for elem in test_list.value]
        
        self.assertListEqual(test_list.value, [0.1, 0.2, 0.3, 0.4])

    def test_param_ser(self):
        # Test parameter serializability
        self.assertDictEqual(
            json.loads(self.voltage_15.model_dump_json()), 
            {'value': 15.0, 'from_var': False, 'var_name': ''})

        # Test parameter model serializability
        new_voltage_model = ParameterModel(**json.loads(self.voltage_model.model_dump_json()))
        self.assertEqual(new_voltage_model.name, "Voltage")
        self.assertEqual(new_voltage_model.data_type, "float")
        self.assertEqual(new_voltage_model._data_type, float)
        self.assertEqual(new_voltage_model.upper_limit, 40)
        self.assertEqual(new_voltage_model.lower_limit, 10)

if __name__ == "__main__":
    unittest.main()