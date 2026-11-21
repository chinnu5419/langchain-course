from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os
load_dotenv()

def main():
    print("Hello from langchain-course!")
    print(os.environ.get("OLLAMA_API_KEY"))
    information = '''
    Subhas Chandra Bose[d] (23 January 1897 – 18 August 1945) was an Indian anti-colonial nationalist whose defiance of British authority in India made him a hero among many Indians,[h] but his wartime alliances with Nazi Germany and the Empire of Japan left a legacy vexed by authoritarianism,[l] antisemitism,[r] and military failure.[v] The honorific Netaji (Hindustani for "Respected Leader") was first applied to Bose in Germany in early 1942—by the Indian soldiers of the Indische Legion and by the German and Indian officials in the Special Bureau for India in Berlin. It is now used throughout India.[w][26]

Bose was born into wealth and privilege in a large Bengali family in Orissa during the British Raj. The early recipient of an Anglo-centric education, he was sent after college to England to take the Indian Civil Service examination. He succeeded with distinction in the vital first exam but demurred at taking the routine final exam, citing nationalism to be the higher calling. Returning to India in 1921, Bose joined the nationalist movement led by Mahatma Gandhi and the Indian National Congress. He followed Jawaharlal Nehru to leadership in a group within the Congress which was less keen on constitutional reform and more open to socialism.[x] Bose became Congress president in 1938. After reelection in 1939, differences arose between him and the Congress leaders, including Gandhi, over the future federation of British India and princely states, but also because discomfort had grown among the Congress leadership over Bose's negotiable attitude to non-violence, and his plans for greater powers for himself.[28] After the large majority of the Congress Working Committee members resigned in protest,[29] Bose resigned as president and was eventually ousted from the party.
    '''
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=[information],template=summary_template)
    llm = ChatOllama(temperature=0,model="gemma3:latest")
    chain = summary_prompt_template | llm
    responce = chain.invoke(input=information)
    print(responce.content)
    
main()


