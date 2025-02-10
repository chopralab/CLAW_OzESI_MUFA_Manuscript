import unittest, sys, inspect, json
from pydantic import BaseModel
from typing import Callable
from types import ModuleType
from ASPIRE_LINQX.core.parameter.base import (
    ParameterModel,
    Parameter 
)
from ASPIRE_LINQX.core.command.base import (
    BaseCommand, 
    BaseRunCommand, 
    BaseInfoCommand, 
    BaseDriverCommand,
)
import ASPIRE_LINQX.testing.models.drivers.LiquidHandler as lh

class TestBaseCommand(unittest.TestCase):

    def setUp(self):
        self.volume = ParameterModel(
            name="volume",
            data_type="float",
            lower_limit=2.0,
            upper_limit=200.0,
            default=20.0,
            desc="volume in uL"
        )
        self.position = ParameterModel(
            name="position",
            data_type="int",
            lower_limit=1,
            upper_limit=10,
            default=1,
            desc="position of LH arm"
        )

        self.aspirate = BaseCommand(
            name="aspirate",
            microservice='liquid_handler',
            parameters= {
                "volume": self.volume,
                "position": self.position,
            }
        )
    
    def test_command_init(self):
        self.assertIsInstance(self.aspirate,BaseCommand)
        self.assertIsInstance(self.aspirate, BaseModel)
        self.assertEqual(self.aspirate.name, "aspirate")
        self.assertEqual(self.aspirate["volume"], self.volume)
        self.assertEqual(self.aspirate["position"], self.position)

    def test_command_values(self):
        self.assertDictEqual(
            dict(self.aspirate.items()),
            {"volume": self.volume, "position": self.position}
        )

class TestBaseRunCommand(unittest.TestCase):
    def setUp(self):
        vol_mol = ParameterModel(
            name="volume",
            data_type="float",
            lower_limit=2.0,
            upper_limit=200.0,
            default=20.0,
            desc="volume in uL"
        )
        position_mol = ParameterModel(
            name="position",
            data_type="int",
            lower_limit=1,
            upper_limit=10,
            default=1,
            desc="position of LH arm"
        )

        self.Volume = vol_mol.to_param()
        self.Position = position_mol.to_param()

        self.aspirate = BaseRunCommand(
            uuid="LH01",
            name="aspirate",
            microservice='liquid_handler',
            parameters= {
                "volume": self.Volume(),
                "position": self.Position()
            }
        )

    def test_run_command_init(self):
        # Check typing
        self.assertIsInstance(self.aspirate, BaseRunCommand)
        self.assertIsInstance(self.aspirate,BaseCommand)
        self.assertIsInstance(self.aspirate, BaseModel)

        # Check uuid
        self.assertEqual(self.aspirate.uuid, "LH01")

        # Test retention of super class functionality
        self.assertEqual(self.aspirate["volume"], self.Volume())
        self.assertEqual(self.aspirate["position"], self.Position())
        self.assertDictEqual(
            dict(self.aspirate.items()),
            {"volume": self.Volume(), "position": self.Position()}
        )

    def test_run_command_to_json(self):
        original_message = self.aspirate.model_dump_json()
        rebuilt_aspirate = BaseRunCommand(**json.loads(original_message))
        rebuilt_message = rebuilt_aspirate.model_dump_json()
        self.assertEqual(original_message, rebuilt_message)
          
class TestBaseLibraryCommand(unittest.TestCase):
    def setUp(self):
        self.volume = ParameterModel(
            name="volume",
            data_type="float",
            lower_limit=2.0,
            upper_limit=200.0,
            default=20.0,
            desc="volume in uL"
        )
        self.position = ParameterModel(
            name="position",
            data_type="int",
            lower_limit=1,
            upper_limit=10,
            default=1,
            desc="position of LH arm"
        )

        self.aspirate = BaseInfoCommand(
            name="aspirate",
            microservice='liquid_handler',
            parameters= {
                "volume": self.volume,
                "position": self.position,
            },
            has_return=True,
            return_signature = {"result": str}
        )

    def test_library_command_init(self):
        # Check typing
        self.assertIsInstance(self.aspirate, BaseInfoCommand)
        self.assertIsInstance(self.aspirate,BaseCommand)
        self.assertIsInstance(self.aspirate, BaseModel)

        # Test retention of super class functionality
        self.assertEqual(self.aspirate["volume"], self.volume)
        self.assertEqual(self.aspirate["position"], self.position)
        self.assertEqual(self.aspirate.has_return, True)
        self.assertDictEqual(self.aspirate.return_signature, {"result": str})
        self.assertDictEqual(
            dict(self.aspirate.items()),
            {"volume": self.volume, "position": self.position}
        )

    def test_library_command_to_run_command(self):
        # Convert to run command
        run_command = self.aspirate.to_run_command(uuid="LH01")
        self.assertIsInstance(run_command, BaseRunCommand)

        self.assertDictEqual(
            dict(run_command.items()),
            {'volume': self.volume.to_param()(), 'position': self.position.to_param()()}
        )

    def test_library_command_ser(self):
        json_dumps = self.aspirate.model_dump_json(indent=2)
        new_library_command = BaseInfoCommand(**json.loads(json_dumps))
        # Make sure the return signature is correct
        self.assertDictEqual(new_library_command.return_signature, {"result": str})

# TODO - Test __call__ with globals added
class TestBaseDriverCommand(unittest.TestCase):
    def setUp(self):
        # Define parameter models for volume and position
        self.volume = ParameterModel(
            name="volume",
            data_type="float",
            lower_limit=2.0,
            upper_limit=200.0,
            default=20.0,
            desc="volume in uL"
        )
        self.position = ParameterModel(
            name="position",
            data_type="int",
            lower_limit=1,
            upper_limit=10,
            default=1,
            desc="position of LH arm"
        )

        # Define aspirate function
        def aspirate(volume: float, position: int) -> dict:
            return {
                "command": "aspirate",
                "volume": volume,
                "position": position
            }
        
        self.aspirate_fn = aspirate

        # Define aspirate driver command
        self.aspirate_model = BaseDriverCommand(
            name="aspirate",
            microservice='liquid_handler',
            parameters ={
                "volume": self.volume,
                "position": self.position,
            },
            uuid="LH01",
            fn=aspirate,
            has_return=True
        )

    def test_driver_command_init(self):
        # Check model typing
        self.assertIsInstance(self.aspirate_model, BaseDriverCommand)
        self.assertIsInstance(self.aspirate_model,BaseCommand)
        self.assertIsInstance(self.aspirate_model, BaseModel)

        # Check model attributes
        self.assertEqual(self.aspirate_model.name, "aspirate")
        self.assertIsInstance(self.aspirate_model.parameters, dict)
        self.assertEqual(self.aspirate_model.uuid, "LH01")
        self.assertTrue(self.aspirate_model.has_return)
        self.assertIsInstance(self.aspirate_model.fn, Callable)
        self.assertIsInstance(self.aspirate_model._function, Callable)

        # Check parameters
        self.assertEqual(self.aspirate_model["volume"], 20.0)
        self.assertEqual(self.aspirate_model["position"], 1)

        # Check function signature parameter and return annotation
        self.assertListEqual(
            list(inspect.signature(self.aspirate_model._function).parameters.keys()),
            ["volume", "position"]
        )
        self.assertIs(
            inspect.signature(self.aspirate_model._function).return_annotation,
            dict
        )

    def test_driver_command_init_from_module(self):
        command_from_module = BaseDriverCommand(
            name="move",
            microservice='liquid_handler',
            parameters = {
                "x_position": self.position,
                "y_position": self.position,
            },
            uuid="LH01",
            module="drivers.LiquidHandler",
            fn="move",
            has_return=True
        )

        # Check model typing
        self.assertIsInstance(command_from_module, BaseDriverCommand)
        self.assertIsInstance(command_from_module,BaseCommand)
        self.assertIsInstance(command_from_module, BaseModel)

        # Check model attributes
        self.assertEqual(command_from_module.name, "move")
        self.assertIsInstance(command_from_module.parameters, dict)
        self.assertEqual(command_from_module.uuid, "LH01")
        self.assertTrue(command_from_module.has_return)
        self.assertEqual(command_from_module.fn, "move")
        self.assertEqual(command_from_module.module, "drivers.LiquidHandler")
        self.assertIsInstance(command_from_module._function, Callable)
        self.assertIsInstance(command_from_module._module, ModuleType)

        # Check parameters
        self.assertEqual(command_from_module["x_position"], 1)
        self.assertEqual(command_from_module["y_position"], 1)
       
       # Check function signature parameter and return annotation
        self.assertListEqual(
            list(inspect.signature(command_from_module._function).parameters.keys()),
            ["x_position", "y_position"]
        )
        self.assertIs(
            inspect.signature(command_from_module._function).return_annotation,
            dict
        )

    def test_driver_command_call(self):
        # Call with default args
        result = self.aspirate_model()
        self.assertIsInstance(result, dict)
        self.assertDictEqual(
            result,
            {"command": "aspirate", "volume": 20.0, "position": 1}
        )

        # Call with new args
        result = self.aspirate_model(volume=45.0,position=5)
        self.assertIsInstance(result, dict)
        self.assertDictEqual(
            result,
            {"command": "aspirate", "volume": 45.0, "position": 5}
        )

        # Call with invalid args
        with self.assertRaises(Exception):
            self.aspirate_model(volume=100.0, position=3, location=2)

        # Call with invalid arg type
        with self.assertRaises(Exception):
            self.aspirate_model(volume=100.0, position="A")

        # Call with invalid arg value
        with self.assertRaises(Exception):
            self.aspirate_model(volume=100.0, position=20)


    def test_driver_command_from_module_call(self):
        command_from_module = BaseDriverCommand(
            name="move",
            microservice='liquid_handler',
            parameters = {
                "x_position": self.position,
                "y_position": self.position,
            },
            uuid="LH01",
            module="drivers.LiquidHandler",
            fn="move",
            has_return=True
        )

        # Call with default arguments
        result = command_from_module()
        self.assertIsInstance(result, dict)
        self.assertDictEqual(
            result,
            {"x_position": 1, "y_position": 1}
        )

        # Call with new args
        result = command_from_module(x_position=3, y_position=4)
        self.assertIsInstance(result, dict)
        self.assertDictEqual(
            result,
            {"x_position": 3, "y_position": 4}
        )

        # Call with invalid arg and validate model is not updated
        with self.assertRaises(Exception):
            command_from_module(x_position=7, z_position=2)

        # Call with invalid arg value and see that model is not updated
        with self.assertRaises(Exception):
            command_from_module(x_position=7, y_position=25)

    def test_driver_command_ser(self):
        # We cannot re-build this command from a mapping because of the function
        json_dump = self.aspirate_model.model_dump_json(indent=2)
        self.assertEqual(json.loads(json_dump)['name'], 'aspirate')
        self.assertEqual(json.loads(json_dump)['uuid'], 'LH01')
        self.assertEqual(json.loads(json_dump)['module'], None)
        self.assertEqual(json.loads(json_dump)['has_return'], True)

        # This will be able to be loaded from a mapping
        command_from_module = BaseDriverCommand(
            name="move",
            microservice='liquid_handler',
            parameters = {
                "x_position": self.position,
                "y_position": self.position,
            },
            uuid="LH01",
            module="drivers.LiquidHandler",
            fn="move",
            has_return=True
        )
        json_dump = command_from_module.model_dump_json(indent=2)
        new_command = BaseDriverCommand(**json.loads(json_dump))

        # Call with new args
        result = new_command(x_position=3, y_position=4)
        self.assertIsInstance(result, dict)
        self.assertDictEqual(
            result,
            {"x_position": 3, "y_position": 4}
        )

        self.assertIsInstance(new_command._parameters['x_position'], Parameter)
        self.assertIsInstance(new_command._parameters['y_position'], Parameter)

if __name__ == "__main__":
    unittest.main()