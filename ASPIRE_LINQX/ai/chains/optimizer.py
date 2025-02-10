from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser

from langchain_openai import ChatOpenAI

from typing import Dict, Any

from ASPIRE_LINQX.core.workflow.base import BaseInfoWorkflow
from ASPIRE_LINQX.ai.schema.optimizer import ObjectiveFunctionSchemaV1
from ASPIRE_LINQX.ai.prompts.optimizer import BASE_OBJECTIVE_FUNCTION_CONSTRUCTOR_PROMPT

def create_obj_fn_from_workflow(
    workflow: BaseInfoWorkflow,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1),
) -> Dict[str, Any]:
    '''
    Builds an objective function based on the provided information workflow.

    Parameters
    ```
    workflow: BaseInfoWorkflow, # The information workflow to optimize
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1), # The LLM to build the objective function
    ```
    '''
    # Create output parser based on objective function structure
    output_parser = JsonOutputParser(pydantic_object=ObjectiveFunctionSchemaV1)

    # Make prompt template instructing LLM to create an objective function
    prompt_template = PromptTemplate(
        template=BASE_OBJECTIVE_FUNCTION_CONSTRUCTOR_PROMPT,
        input_variables=['info_workflow'],
        partial_variables={
            'format_instructions': output_parser.get_format_instructions()
        }
    )

    # Create the LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_parser=output_parser
    )

    # Invoke with provided info workflow
    return chain.invoke(input={'info_workflow': workflow.model_dump_json(indent=2)})

def create_evotorch_problem_from_query():
    pass