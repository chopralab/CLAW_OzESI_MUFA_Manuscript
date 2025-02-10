import unittest
from ASPIRE_LINQX.core.parameter.base import ParameterModel
from ASPIRE_LINQX.core.command.base import BaseDriverCommand
from ASPIRE_LINQX.core.library.base import (
    BaseInfoMicroservice,
    BaseInfoCommandLibrary,
    BaseDriverMicroservice,
    BaseDriverCommandLibrary
)

class TestDriverMicroservice(unittest.TestCase):

    def setUp(self) -> None:
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
        self.aspirate = BaseDriverCommand(
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

    def test_driver_microservice_init(self):
        with self.assertRaises(KeyError):
            BaseDriverMicroservice(
                name='liquid_handler',
                desc='A virtual liquid handler',
                commands={'key': self.aspirate}
            )
        with self.assertRaises(ValueError):
            BaseDriverMicroservice(
                name='not_a_liquid_handler',
                desc='A virtual liquid handler',
                commands={'key': self.aspirate}
            )
        liquid_handler = BaseDriverMicroservice(
            name='liquid_handler',
            desc='A virtual liquid handler',
            commands={'aspirate': self.aspirate}
        )

if __name__ == "__main__":

    unittest.main()