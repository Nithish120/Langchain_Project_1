import os
from pathlib import Path

from dotenv import load_dotenv
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
@tool
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

llm=ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)
tools=[search]
agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You can use the search tool at most once. "
        "After receiving tool output, provide a final answer and do not call tools again."
        "Return the answer tidyly in bulletin format"
    ),
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
