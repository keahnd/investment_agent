import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,     # low temperature for financial analysis
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0.1,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")