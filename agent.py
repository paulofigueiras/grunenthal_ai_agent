from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import tools.fda_api as fda_api
import tools.financial_report_rag as financial_report_rag
import tools.neo4j_rag as neo4j_rag

# Load environment variables from .env file
load_dotenv()

# This tool is used to query the FDA API for adverse events related to a drug
@tool
def fda_tool(drug_name: str) -> str:
    """Query the FDA Adverse Events API for adverse consequences or adverse interactions and events for a drug (e.g. 'ibuprofen')."""
    return fda_api.query_fda(drug_name)

#This tool is used to query the financial report PDF for Grünenthal for the years 2023 and 2024
@tool
def financial_rag_tool(question: str) -> str:
    """Answer questions from Grünenthal’s financial report PDF fpr the years 2023 and 2024."""
    return financial_report_rag.query_financial_report(question)

#This tool is used to query the Neo4j Healthcare Analytics graph database for medical questions
@tool
def neo4j_tool(query: str) -> str:
    """Answer questions about drug adverse event cases, drug manufacturers and outcomes from the Neo4j Healthcare Analytics medical graph database by transforming questions into Cypher queries and building human responses from the results."""
    return neo4j_rag.query_neo4j(query)

# Define the tools to be used
tools = [financial_rag_tool, neo4j_tool, fda_tool]

# Initialize the Google Generative AI model
llm = init_chat_model(
    "gemini-2.0-flash", 
    model_provider="google_genai",
    temperature=0.5,
    max_tokens=512)

llm_with_tools = llm.bind_tools(tools)

async def execute_agent(query: str) -> str:
    messages = [HumanMessage(query)]

    ai_msg = llm_with_tools.invoke(messages)

    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"neo4j_tool": neo4j_tool, "fda_tool": fda_tool, "financial_rag_tool": financial_rag_tool}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    
    return llm_with_tools.invoke(messages).content
