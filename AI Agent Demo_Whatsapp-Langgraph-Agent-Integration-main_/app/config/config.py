import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def setup_model(config: dict):
    provider = config.get("provider", "openai")
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.6)

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
