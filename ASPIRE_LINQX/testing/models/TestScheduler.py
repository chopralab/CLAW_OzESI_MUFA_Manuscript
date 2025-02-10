import unittest, sys, json
sys.path.append("../../JSON")
from pydantic import BaseModel, ValidationError
from models.parameter.base import ParameterModel
from models.scheduler.base import (
    _BaseScheduleModel,
    BaseScheduleTemplate,
    BaseScheduleObject
)

class TestBaseScheduleModels(unittest.TestCase):
    def setUp(self):
        self.solvent_model = ParameterModel(
            name="Solvent",
            data_type='str',
            allowed_values=[
                "Acetone",
                "Hexane",
                "DMSO",
                "THF"
            ],
            default="Acetone",
            desc="Organic solvents for rotary evaporation"
        )
        self.solvent = self.solvent_model.to_param()
        self.temperature_model = ParameterModel(
            name="Temperature",
            data_type="int",
            upper_limit=100,
            lower_limit=27,
            default=27,
            desc="Temperature of the evaporator water bath"
        )
        self.volume_model = ParameterModel(
            name="Volume",
            data_type="float",
            upper_limit=100,
            lower_limit=10,
            default=50,
            desc="Volume of the flask"
        )
        self.type_model = ParameterModel(
            name="Type",
            data_type="str",
            allowed_values=[
                "Tube",
                "Round",
            ],
            default="Round",
            desc="Flask to be used"
        )
        self.flask = BaseScheduleTemplate(
            name="Flask",
            template={
                "type": self.type_model,
                "volume": self.volume_model
            }
        )
        self.evaporator = BaseScheduleTemplate(
            name="RotaryEvaporator",
            template={
                "solvent": self.solvent_model,
                "temperature": self.temperature_model,
                "flask": self.flask
            }
        )

    def test_schedule_field_init(self):
        # Test successful init
        self.assertIsInstance(self.flask, BaseScheduleTemplate)
        self.assertIsInstance(self.flask, _BaseScheduleModel)
        self.assertIsInstance(self.flask, BaseModel)

        self.assertIsInstance(self.evaporator, BaseScheduleTemplate)
        self.assertIsInstance(self.evaporator, _BaseScheduleModel)
        self.assertIsInstance(self.evaporator, BaseModel)

        # Test failed init, template value cannot be a float
        with self.assertRaises(ValidationError):
            BaseScheduleTemplate(
                name="RotaryEvaporator",
                template={
                    "solvent": self.solvent_model,
                    "temperature": 100.0
                }
            )

    def test_get_item(self):
        # Ensure that we can get all valid items
        self.assertIsInstance(self.flask["type"], ParameterModel)
        self.assertEqual(self.flask["type"].data_type, "str")
        self.assertListEqual(self.flask["type"].allowed_values, ["Tube", "Round"])

        self.assertIsInstance(self.flask["volume"], ParameterModel)
        self.assertEqual(self.flask["volume"].data_type, "float")
        self.assertIs(self.flask["volume"]._data_type, float)
        self.assertEqual(self.flask["volume"].upper_limit, 100)
        self.assertEqual(self.flask["volume"].lower_limit, 10)

        # We should get a KeyError if the key is not present in the template
        with self.assertRaises(KeyError):
            self.flask["other"]

        # Ensure that we can access templates in a nested manner
        self.assertIsInstance(self.evaporator["flask"]["type"], ParameterModel)
        self.assertEqual(self.evaporator["flask"]["type"].data_type, "str")
        self.assertListEqual(self.evaporator["flask"]["type"].allowed_values, ["Tube", "Round"])

    def test_set_item(self):
        # Ensure that we can set an item
        self.evaporator["type"] = self.type_model
        self.assertListEqual
        self.assertIsInstance(self.evaporator["type"], ParameterModel)

    def test_to_object(self):
        # Convert both of the templates to objects
        flask_object = self.flask.to_obj()
        evaporator_object = self.evaporator.to_obj()

        # Ensure that object is of the correct typing
        self.assertIsInstance(flask_object, BaseScheduleObject)
        self.assertIsInstance(flask_object, _BaseScheduleModel)
        self.assertIsInstance(flask_object, BaseModel)

        self.assertIsInstance(evaporator_object, BaseScheduleObject)
        self.assertIsInstance(evaporator_object, _BaseScheduleModel)
        self.assertIsInstance(evaporator_object, BaseModel)

        # Test keys and values work correctly
        self.assertListEqual(list(flask_object.keys()), ["type", "volume"])
        self.assertListEqual(list(flask_object.values()), ["Round", 50])
        self.assertListEqual(list(evaporator_object.keys()), ["solvent", "temperature", "flask"])

        # Test that BaseScheduleObject has correct parameter values
        self.assertEqual(flask_object["type"], "Round")
        self.assertEqual(flask_object["volume"], 50)
        self.assertEqual(evaporator_object["solvent"], "Acetone")
        self.assertEqual(list(evaporator_object["flask"].keys()), ["type", "volume"])

        # Test that BaseScheduleObject Parameter can be maniuplated
        flask_object["volume"] += 5
        self.assertEqual(flask_object["volume"], 55)

        # Test invalid update of BaseScheduleObject Parameter
        with self.assertRaises(ValidationError):
            flask_object["volume"] += 500
        self.assertEqual(flask_object["volume"], 55)

    def test_schedule_template_ser(self):
        # Test JSON seralizability
        flask_json = json.loads(self.flask.model_dump_json())
        new_flask = BaseScheduleTemplate(**flask_json)

        evaporator_json = json.loads(self.evaporator.model_dump_json())
        new_evaporator = BaseScheduleTemplate(**evaporator_json)

        # Ensure that created objects are of the correct class
        self.assertIsInstance(new_flask, BaseScheduleTemplate)
        self.assertIsInstance(new_flask, _BaseScheduleModel)
        self.assertIsInstance(new_flask, BaseModel)

        self.assertIsInstance(new_evaporator, BaseScheduleTemplate)
        self.assertIsInstance(new_evaporator, _BaseScheduleModel)
        self.assertIsInstance(new_evaporator, BaseModel)

        # Ensure that fields are correct
        self.assertIsInstance(new_flask["type"], ParameterModel)
        self.assertEqual(new_flask["type"].data_type, "str")
        self.assertListEqual(new_flask["type"].allowed_values, ["Tube", "Round"])

        self.assertIsInstance(new_flask["volume"], ParameterModel)
        self.assertEqual(new_flask["volume"].data_type, "float")
        self.assertIs(new_flask["volume"]._data_type, float)
        self.assertEqual(new_flask["volume"].upper_limit, 100)
        self.assertEqual(new_flask["volume"].lower_limit, 10)

        self.assertIsInstance(new_evaporator["flask"], BaseScheduleTemplate)
        self.assertIsInstance(new_evaporator["flask"]["type"], ParameterModel)
        self.assertEqual(new_evaporator["flask"]["type"].data_type, "str")
        self.assertListEqual(new_evaporator["flask"]["type"].allowed_values, ["Tube", "Round"])

if __name__ == "__main__":
    unittest.main()