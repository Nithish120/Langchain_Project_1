import os
from pathlib import Path
from typing import List
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily=TavilyClient()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Set it in React_Search_agent/.env")


#THis is a regular python function but to make it a tool we have to add @tool above it
@tool("search")
def search(query:str)->str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for 
    Returns:
        The search result    
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)


class Source(BaseModel):
    """
    Schema for a source used by agent 
    """
    url:str =Field(description="The URL of the Source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str=Field(description="The answer to agent's query")   
    sources: List[Source]=Field(default_factory=list,description="List of sources used to generate the answer")



llm=ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)
tools=[search]
agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You have exactly one available tool named `search`. "
        "Never call any other tool name such as `brave_search`. "
        "Use `search` when needed, then provide a final answer."
    ),
    response_format=AgentResponse

)



def main():
    print("Hello from react-search-agent!")
    result=agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details along with their apply links or urls ")]},
        config={"recursion_limit": 4},
    )
    print(result)

if __name__ == "__main__":
    main()
