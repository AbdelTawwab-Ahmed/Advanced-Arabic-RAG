from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from src import config


def get_llm(temperature: float = config.TEMP_DETERMINISTIC):
    primary = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=temperature)
    fallback = ChatGroq(
        model=config.GROQ_FALLBACK_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=temperature,
        reasoning_effort="low",
        max_tokens=4096,
    )
    return primary.with_fallbacks([fallback])


def get_eval_llm(temperature: float = config.TEMP_DETERMINISTIC):
    return ChatGroq(
        model=config.GROQ_FALLBACK_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=temperature,
        reasoning_effort="low",
        max_tokens=4096,
    )