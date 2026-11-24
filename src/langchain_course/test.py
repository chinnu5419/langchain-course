
from typing import List
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# 1. Define the Pydantic schema

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The answer to the user's query")

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )


# 2. Initialize Ollama

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 3. Convert LLM into structured LLM

structured_llm = llm.with_structured_output(
    AgentResponse,
    method="json_schema"
)


# 4. Give sample information to the model

prompt = """
Extract the information below and populate every field.

Weather: Cloudy
Temperature: 28.3 Celsius
Humidity: 50 percent
Location: Hyderabad, India
Source URL: https://www.weatherapi.com/

Provide a meaningful answer containing the weather information.
Include the source URL in the sources list.
Do not leave the answer empty.
"""


# 5. Invoke the model

response = structured_llm.invoke(prompt)


# 6. Print the result

print("\nSTRUCTURED OUTPUT:")
print(response.model_dump())

print("\nANSWER:")
print(response.answer)

print("\nSOURCES:")
for source in response.sources:
    print(source.url)
