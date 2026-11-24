
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch


# Load environment variables from .env
load_dotenv()


# ---------------------------------------
# 1. Define structured output schemas
# ---------------------------------------

class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(
        description="The URL of the source"
    )


class AgentResponse(BaseModel):
    """Schema for the final agent response."""

    answer: str = Field(
        description="The answer to the user's query"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="List of source URLs used to generate the answer"
    )


# ---------------------------------------
# 2. Initialize LLM and Tavily
# ---------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

tavily = TavilySearch(
    max_results=5,
    search_depth="basic"
)


# ---------------------------------------
# 3. Create a simplified search tool
# ---------------------------------------

@tool
def search_web(query: str) -> str:
    """
    Search the web for information using a search query.
    Use this tool when current or external information is required.
    """

    result = tavily.invoke({
        "query": query
    })

    return str(result)


# ---------------------------------------
# 4. Create the agent
# ---------------------------------------

agent = create_agent(
    model=llm,
    tools=[search_web]
)


# ---------------------------------------
# 5. Main function
# ---------------------------------------

def main():

    question = "What is the weather in Hyderabad?"

    print("\n========== USER QUESTION ==========")
    print(question)

    # Run the agent and execute its search tool
    print("\n========== RUNNING AGENT ==========")

    result = agent.invoke({
        "messages": [
            HumanMessage(content=question)
        ]
    })

    # Extract tool results
    search_results = []

    for message in result["messages"]:

        if isinstance(message, ToolMessage):
            search_results.append(message.content)

    print("\n========== SEARCH RESULTS ==========")

    for item in search_results:
        print(item)

    # Check whether the search tool returned anything
    if not search_results:
        print("\nNo search results were returned.")
        return

    # ---------------------------------------
    # 6. Generate structured output separately
    # ---------------------------------------

    print("\n========== GENERATING STRUCTURED OUTPUT ==========")

    structured_llm = llm.with_structured_output(
        AgentResponse
    )

    final_result = structured_llm.invoke([
        SystemMessage(
            content=(
                "Answer the user's question using the supplied search results. "
                "Include the relevant source URLs found in those results. "
                "Do not invent facts or URLs. "
                "If the results do not contain enough information, "
                "clearly say so."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Search results: {search_results}"
            )
        )
    ])

    # ---------------------------------------
    # 7. Display final structured response
    # ---------------------------------------

    print("\n========== STRUCTURED RESPONSE ==========")

    print(final_result.model_dump())

    print("\n========== FINAL ANSWER ==========")

    print(final_result.answer)

    print("\n========== SOURCES ==========")

    for source in final_result.sources:
        print(source.url)


# ---------------------------------------
# 8. Entry point
# ---------------------------------------

if __name__ == "__main__":
    main()
