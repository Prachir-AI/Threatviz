import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI



_ = load_dotenv()


def load_llm(provider: str):
    #-------supported_models--------
    #   qwen/qwen3-32b
    #   meta-llama/llama-4-maverick-17b-128e-instruct
    #   openai/gpt-oss-120b
    #   openai/gpt-oss-20b
    #   openai/gpt-4o-mini
    #   moonshotai/kimi-k2-instruct-0905
    #------------------------------------
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("❌ GROQ_API_KEY not found in .env")
        return ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("❌ OPENAI_API_KEY not found in .env")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)


    raise ValueError("Unsupported model provider")