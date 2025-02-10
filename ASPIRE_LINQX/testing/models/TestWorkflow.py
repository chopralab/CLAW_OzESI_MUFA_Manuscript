import unittest, sys, inspect, json
from pydantic import BaseModel
from typing import (
    Callable, 
    List, 
    Dict
)
import random
from types import ModuleType
from ASPIRE_LINQX.core.parameter.base import ParameterModel
from ASPIRE_LINQX.core.command.base import (
    BaseCommand, 
    BaseRunCommand, 
    BaseInfoCommand, 
    BaseDriverCommand,
)
from ASPIRE_LINQX.core.workflow.base import (
    BaseWorkflow,
    BaseLibraryWorkflow,
    BaseRunWorkflow,
    BaseDriverWorkflow
)

class TestLibraryWorkflow(unittest.TestCase):
    def setUp(self):
        self.vector_model = ParameterModel(
            name="vector",
            data_type="float",
            lower_limit=-1.0,
            upper_limit=1.0,
            default=[0,0,0,0],
            is_list=True,
            desc="Bit vector representation of a molecule"
        )
        self.smiles_model = ParameterModel(
            name="smiles",
            data_type="str",
            desc="SMILES string representation of a molecule",
        )

        self.Vector = self.vector_model.to_param()()
        self.Smiles = self.smiles_model.to_param()()
        self.Smiles.from_var = True
        self.Smiles.var_name = "smiles"

        self.generate = BaseInfoCommand(
            name="generate_mol",
            microservice='molecular_generator',
            parameters={
                "vector": self.vector_model
            },
            has_return=True,
            return_signature={"smiles": str}
        )
        self.score = BaseInfoCommand(
            name="score_mol",
            microservice='molecular_generator',
            parameters={
                "smiles": self.smiles_model
            },
            has_return=True,
            return_signature={"score": float}
        )

        self.workflow = BaseLibraryWorkflow(
            name="generate_and_score",
            commands=[
                self.generate,
                self.score
            ]
        )
        
    def test_lib_workflow_init(self):
        # Check class inheritance
        self.assertIsInstance(self.workflow, BaseLibraryWorkflow)
        self.assertIsInstance(self.workflow, BaseWorkflow)
        self.assertIsInstance(self.workflow, BaseModel)

        # Check class vars
        self.assertEqual(self.workflow.name, "generate_and_score")
        for elem in self.workflow.commands: self.assertIsInstance(elem, BaseInfoCommand)
        self.assertEqual(self.workflow.commands[0].name, "generate_mol")
        self.assertEqual(self.workflow.commands[1].name, "score_mol")
        
    def test_lib_workflow_to_run_workflow(self):
        uuids = 2 * ["slurm"]

        # This means that "smiles" output of command 1 (generate mol) should be saved
        # to global dict smiles
        save_var_list = [{"smiles": "smiles"}, None]
        var_name_list = [None, {'smiles': 'smiles'}]

        # Make sure it is a list of strings
        run_workflow = self.workflow.to_run_workflow(
            uuids=uuids,
            var_name_list=var_name_list, 
            save_var_list=save_var_list,
        )
        self.assertIsInstance(run_workflow,BaseRunWorkflow)
        for run_command in run_workflow:
            self.assertIsInstance(run_command, BaseRunCommand)

        # Check the generate command after loading
        gen_command = run_workflow[0]
        self.assertEqual(gen_command.name, "generate_mol")
        self.assertListEqual(list(gen_command.parameters.keys()), ["vector"])
        self.assertEqual(gen_command.uuid, "slurm")
        self.assertDictEqual(gen_command.save_vars, {"smiles": 'smiles'})

        # Check the score command after loading
        score_command = run_workflow[1]
        self.assertEqual(score_command.name, "score_mol")
        # This one was set to read in from a varaible
        self.assertDictEqual(score_command.parameters, {"smiles": self.Smiles})
        self.assertEqual(score_command.uuid, "slurm")
        self.assertDictEqual(score_command.save_vars, {})

        # Make sure the errors are thrown when lists of incorrect sizes are passed in
        with self.assertRaises(Exception):
            self.workflow.to_run_workflow(
                uuids=["slurm"], 
                save_var_list=save_var_list
            )
        with self.assertRaises(Exception):
            self.workflow.to_run_workflow(
                uuids=uuids,
                save_var_list=[{"smiles": "smiles"}]
            )

class TestDriverWorkflow(unittest.TestCase):
    def setUp(self):
        self.vector_model = ParameterModel(
            name="vector",
            data_type="float",
            lower_limit=-1.0,
            upper_limit=1.0,
            default=[0,0,0,0],
            is_list=True,
            desc="Bit vector representation of a molecule"
        )
        self.smiles_model = ParameterModel(
            name="smiles",
            data_type="str",
            desc="SMILES string representation of a molecule",
            from_var=True,
            var_name="smiles"
        )

        self.Vector = self.vector_model.to_param()
        self.Smiles = self.smiles_model.to_param()

        def fn_generate(vector: List[float]) -> Dict[str, str]:
            if len(vector) != 4: return {"smiles":"ERROR"}
            if vector[0] > 0.5: return {"smiles": "CCC=O"}
            if vector[1] > 0.5: return {"smiles": "CCOCC"}
            if vector[2] > 0.5: return {"smiles": "CCCN"}
            if vector[3] > 0.5: return {"smiles": "CCC=OOCCN"}
            else: return{"smiles": "CCCCC"}

        def fn_score(smiles: str) -> Dict[str, float]:
            if smiles is None: return {"score": "None"}
            elif smiles == "CCC=O": return {"score": random.uniform(0, 1.0)}
            elif smiles == "CCOCC": return {"score": 1 + random.uniform(0, 1.0)}
            elif smiles == "CCCN": return {"score": 2 + random.uniform(0, 1.0)}
            elif smiles == "CCC=OOCCN": return {"score": 3 + random.uniform(0, 1.0)}
            elif smiles == "CCCCC": return {"score": 4 + random.uniform(0, 1.0)}
            else: return {"score": "Error"}
            
        
        self.fn_generate = fn_generate
        self.fn_score = fn_score

        self.generate = BaseDriverCommand(
            name="generate_mol",
            microservice='molecular_generator',
            uuid="gilbreth",
            fn=fn_generate,
            has_return=True,
            parameters={
                "vector": self.vector_model
            }
        )

        self.score = BaseDriverCommand(
            name="score_mol",
            microservice='molecular_generator',
            uuid="gilbreth",
            fn=fn_score,
            has_return=True,
            parameters = {
                "smiles": self.smiles_model
            }
        )

        self.workflow = BaseDriverWorkflow(
            name="generate_and_score_mol",
            commands=[
                self.generate,
                self.score
            ]
        )

    def test_driver_workflow_init(self):
        # Check inheritance
        self.assertIsInstance(self.workflow, BaseDriverWorkflow)
        self.assertIsInstance(self.workflow, BaseWorkflow)
        self.assertIsInstance(self.workflow, BaseModel)

        # Check attributes
        self.assertEqual(self.workflow.name, "generate_and_score_mol")
        self.assertDictEqual(self.workflow.wf_globals, {})
        generate = self.workflow.commands[0]
        score = self.workflow.commands[1]
        self.assertIsInstance(generate, BaseDriverCommand)
        self.assertIsInstance(score, BaseDriverCommand)
        self.assertEqual(generate.name, "generate_mol")
        self.assertEqual(score.name, "score_mol")
        self.assertEqual(generate.fn, self.fn_generate)
        self.assertEqual(score.fn, self.fn_score)

    def test_driver_workflow_exec(self):
        # Run without any runtime args, save off output of first command to smiles
        list_kwargs = [{}, {}]
        list_save_vars = [{"smiles": "smiles"}, None]
        log = self.workflow.exec(
            list_kwargs=list_kwargs,
            list_save_vars=list_save_vars
        )
        # Ensure that the outputs and globals match expected command output
        self.assertDictEqual(log[0], {"smiles": "CCCCC"})
        self.assertGreater(log[1]["score"], 4.0)
        self.assertLess(log[1]["score"], 5.0)
        self.assertDictEqual(self.workflow.wf_globals, {"smiles": "CCCCC"})
        
        self.workflow.clear_wf_globals()
        # Rerun with different args
        list_kwargs = [{"vector": [1.0,0,0,0]}, {}]
        list_save_vars = [{"smiles": "smiles"}, {"score": "sa_score"}]
        log = self.workflow.exec(
            list_kwargs=list_kwargs,
            list_save_vars=list_save_vars
        )
        

if __name__ == "__main__":
    unittest.main()