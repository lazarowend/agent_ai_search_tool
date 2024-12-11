import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada. Verifique o arquivo .env.")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang='pt'
    )
)

agent_executor = create_python_agent(
    llm=model,
    tool=wikipedia_tool,
    verbose=True
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Pesquise na web sobre {query} e forneça um resumo sobre,
    responda tudo em pt-br
    '''
)

query = 'Alan Turing'

prompt = prompt_template.format(query=query)

response = agent_executor.invoke(prompt)
print(response.get('output'))