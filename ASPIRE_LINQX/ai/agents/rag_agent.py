from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.chains import create_extraction_chain, RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor

# from ASPIRE_LINQX.ai.tools.rag_tool import get_answer_from_information #TODO: Solve this in future

def rag_agent(
    question: str,
    path_to_embeddings: str,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1) #?QUESTION Is this a good default value?
):
    """
    Create a RAG chain with the tools.
    """
    embeddings = OpenAIEmbeddings()
    # # #?QUESTION Do we assume that the FAISS is used to make the embeddings? or do we need to switch this to a different vector store if required?
    db = FAISS.load_local(path_to_embeddings, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    @tool
    def get_answer_from_information(situation: str) -> str:
        """
        The user will provide a situation and the tool will return the answer from the information.
        Provide a well formatted answer that is easy for the user to understand.
        """
        return retriever.invoke(situation)

    tools = [get_answer_from_information]

    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are very powerful assistant. 
                You can answer questions about user query using relevant sources. You have to provide factual answers.
                If requested to provide answer to the question you may use following steps:
                    1. Rephrase the question in a query in the most explanatory way possible.
                    2. Call the get_answer_from_information tool to get relevant information from the documents.
                    3. Understand the retrieved information and provide the answer in a clear format to the user.
                """, 
            ),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    chain = (
        {
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    rag_agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True)

    return rag_agent_executor