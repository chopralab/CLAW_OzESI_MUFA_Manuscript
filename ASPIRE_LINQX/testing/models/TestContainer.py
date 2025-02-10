import unittest, sys, os, json
from pydantic import BaseModel
from ASPIRE_LINQX.core.parameter.base import ParameterModel
from ASPIRE_LINQX.core.command.base import BaseDriverCommand
from ASPIRE_LINQX.core.command.container import (
    ContainerDriverCommand,
    ApptainerDriverCommand,
)
from spython.utils.terminal import check_install
from spython.instance import Instance
from spython.main.base import Client as SpythonClient

# Define microservice path, change to env later
KURSAWE_MICROSERVICE = '../../microservices/kursawe.sif'

@unittest.skipIf(
    not check_install(),
    'Singularity/Apptainer install not detected, please install to run test case'
)
@unittest.skipUnless(
    os.path.isfile(KURSAWE_MICROSERVICE),
    'Kursawe microservice not found, please build kursawe.sif to run test case'
)
class TestApptainerDriverCommand(unittest.TestCase):
    def setUp(self):
        # Create parameter model for kursawe optimization
        self.vector_model = ParameterModel(
            name='Vector',
            data_type='float',  # The vector is a list of floats
            upper_limit=5,      # The upper bound of the optimization problem is 5
            lower_limit=-5,     # The lower bound of the optimization problem is -5
            default=[0,0,0],    # Give all vectors a default of [0,0,0]
            is_list=True        # The vector is a list
        )

        # Create the parameter class
        self.Vector = self.vector_model.to_param()

        # Build a command for running f1 of the kursawe microservice (app f1)
        self.f1 = ApptainerDriverCommand(
            name='run_function_1',
            microservice='kursawe',
            parameters={
                "VECTOR": self.vector_model                # The only parameter is the vector
            },
            uuid="cluster",                         
            image_name=KURSAWE_MICROSERVICE, # The path to the .sif file
            run_app=True,                           # We are running an app of the instance
            app='f1',                               # The app we are running is f1 (function 1)
            start_delay=0,                          # There is no need to have a start delay
            has_return=True,                        # We are expecting a result to be returned
        )

        # Build a command for running f2 of the kursawe microservice (app f2)
        self.f2 = ApptainerDriverCommand(
            name='run_function_2',
            microservice='kursawe',
            parameters={
                "VECTOR": self.vector_model   
            },
            uuid="cluster",
            image_name=KURSAWE_MICROSERVICE,
            run_app=True,
            app='f2',                               # We now want to run the f2 app instead of f1
            start_delay=0,
            has_return=True,
        )

    def tearDown(self):
        # Stop the microservices after each test case
        # if they are running
        self.f1.stop()
        self.f2.stop()
    
    def test_apptainer_driver_command_init(self):
        # Test inheritance
        self.assertIsInstance(self.f1, ApptainerDriverCommand)
        self.assertIsInstance(self.f1, ContainerDriverCommand)
        self.assertIsInstance(self.f1, BaseDriverCommand)
        self.assertIsInstance(self.f1, BaseModel)

        # Test client init
        self.assertIsInstance(self.f1._client, SpythonClient)
        
        # Instance should initially be none
        self.assertIsNone(self.f1._instance)

        # Ensure instance starts correctly
        instance = self.f1.start()
        self.assertIsInstance(instance, Instance)
        self.assertIsInstance(self.f1._instance, Instance)
        
        # Ensure the instance stops correctly
        self.f1.stop()
        self.assertIsNone(self.f1._instance)

    def test_apptainer_driver_command_call(self):
        # Call the apptainer command
        wf_globals = {}
        log = self.f1(
            wf_globals=wf_globals,
            save_vars={"f1": "f1"},
            VECTOR=[-1,3,4]
        )

        # Ensure the output is correct
        self.assertDictEqual(log,{'f1': -8.991649627685547})
        self.assertDictEqual(wf_globals,{'f1': -8.991649627685547})

        self.assertListEqual(self.f1["VECTOR"], [-1,3,4])

    def test_apptainer_driver_command_ser(self):
        # Dump the model as a JSON and reload to new command
        json_dump = self.f1.model_dump_json(indent=2)
        new_f1 = ApptainerDriverCommand(**json.loads(json_dump))

        # Test inheritance
        self.assertIsInstance(new_f1, ApptainerDriverCommand)
        self.assertIsInstance(new_f1, ContainerDriverCommand)
        self.assertIsInstance(new_f1, BaseDriverCommand)
        self.assertIsInstance(new_f1, BaseModel)

        # Test client init
        self.assertIsInstance(new_f1._client, SpythonClient)

        # Test call
        wf_globals = {}
        log = new_f1(
            wf_globals=wf_globals,
            save_vars={"f1": "f1"},
            VECTOR=[-1,3,4]
        )
        self.assertIsInstance(new_f1._instance, Instance)

        # Ensure the output is correct
        self.assertDictEqual(log,{'f1': -8.991649627685547})
        self.assertDictEqual(wf_globals,{'f1': -8.991649627685547})

        self.assertListEqual(new_f1["VECTOR"], [-1,3,4])
        
        # Stop the new command
        new_f1.stop()

if __name__ == "__main__":
    unittest.main()