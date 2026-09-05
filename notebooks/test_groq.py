# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# response = requests.get(
#     "https://api.groq.com/openai/v1/models",
#     headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
# )
# for model in response.json()["data"]:
#     print(model["id"])


# from src.llm_client import get_eval_llm

# llm = get_eval_llm()
# response = llm.invoke("ما هي عاصمة الأردن؟")
# print(response.content)

from langchain_groq import ChatGroq
from src import config

llm = ChatGroq(model=config.GROQ_FALLBACK_MODEL, api_key=config.GROQ_API_KEY,
                temperature=0, reasoning_effort="low")

long_prompt = "اشرح بالتفصيل جميع خطوات إصدار وإيقاف وإلغاء البطاقات والبصمات في البنك، مع ذكر كل دور ومسؤولية بدقة." * 3

response = llm.invoke(long_prompt)
print(f"Output length (chars): {len(response.content)}")
print(response.content[:300])