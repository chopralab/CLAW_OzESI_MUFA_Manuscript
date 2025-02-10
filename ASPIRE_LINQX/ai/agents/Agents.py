from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.agents import (
    StructuredChatAgent, 
    AgentOutputParser
)
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationKGMemory
from ASPIRE_LINQX.ai.prompts.Prompt import (
    template_structured_prompt,
    request_structured_prompt,
    microservice_selector_prompt,
    app_selector_prompt,
    microservice_summary_prompt, 
)
from ASPIRE_LINQX.ai.tools.Tools import (
    template_builder_tool_names,
    request_builder_tool_names,
    microservice_selector_tool_names,
    app_selector_tool_names
)
from typing import Union
import re

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check to see if the agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: '{llm_output}'")
        action = match.group(1).strip()
        acton_input = match.group(2)
        return AgentAction(
            tool=action, 
            tool_input=acton_input.strip(" ").strip('"'),
            log=llm_output
        )
    
custom_output_parser = CustomOutputParser()

# Define LLM
# llm = OpenAI(temperature=0)
llm = ChatOpenAI(temperature=0, model='gpt-4')

# Define KG memory
memory = ConversationKGMemory(llm=llm, memory_key="chat_history", k=10)

# LLM Chains
template_structured_llm_chain = LLMChain(
    llm=llm, 
    prompt=template_structured_prompt
)
request_structured_llm_chain = LLMChain(
    llm=llm, 
    prompt=request_structured_prompt
)
microservice_selector_llm_chain = LLMChain(
    llm=llm, 
    prompt=microservice_selector_prompt
)
app_selector_llm_chain = LLMChain(
    llm=llm,
    prompt=app_selector_prompt
)

microservice_summary_llm_chain = LLMChain(
    llm=llm,
    prompt=microservice_summary_prompt
)

# Agents
template_structured_agent = StructuredChatAgent(
    llm_chain=template_structured_llm_chain,
    allowed_tools=template_builder_tool_names,
)

request_structured_agent = StructuredChatAgent(
    llm_chain=request_structured_llm_chain,
    allowed_tools=request_builder_tool_names
)

microservice_selector_structured_agent = StructuredChatAgent(
    llm_chain=microservice_selector_llm_chain,
    allowed_tools=microservice_selector_tool_names
)

app_selector_structured_agent = StructuredChatAgent(
    llm_chain=app_selector_llm_chain,
    allowed_tools=app_selector_tool_names
)

microservice_summary_structure_agent = StructuredChatAgent(
    llm_chain=microservice_summary_llm_chain,
    allowed_tools=microservice_selector_tool_names
)