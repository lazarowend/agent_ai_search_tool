import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


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
