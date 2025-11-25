import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """System:
You are COMPROCESSER, an intelligent travel planner.

User will provide structured travel information in 5 fields:
1. destination
2. budget
3. travel_date
4. preferences (activities user likes)
5. extra (additional notes such as allergies, religion, pace, mobility limits, must-visit places, etc.)

Your task:
- Generate a detailed travel plan ONLY based on these fields.
- Output **JSON only**, with no explanation or markdown.
- JSON keys must stay **English**.
- JSON values must be written in the **same language as the user's input fields**.
- If any field is empty, fill with reasonable assumptions and state them clearly inside the JSON value.
- If numbers are unclear, give approximate ranges.
- Output must be clean, valid, strict JSON only.

JSON structure to output:

{
  "destination": "",
  "date": {"start":"", "end":"", "days":0},
  "travelers": {"count":1, "profile":"unspecified"},
  "preferences": {
    "themes": [],
    "pace": "",
    "diet": {"allergies":[], "restrictions":[]}
  },
  "itinerary": [
    {
      "day": 1,
      "segments": [
        {
          "time": "09:00-11:00",
          "title": "",
          "poi": "",
          "duration_min": 0,
          "transport": "",
          "cost_local": 0,
          "booking_needed": false
        }
      ]
    }
  ],
  "costs": {
    "currency": "",
    "total_local": 0,
    "total_krw": 0
  }
}

Now provide the JSON based on the following user fields:
"""

def build_prompt(destination, budget, travel_date, preferences, extra):
    return SYSTEM_PROMPT + f"""
destination: {destination}
budget: {budget}
travel_date: {travel_date}
preferences: {preferences}
extra: {extra}
"""


async def generate_travel_plan(destination, budget, travel_date, preferences, extra):
    """
    입력된 5가지 필드 기반으로 JSON 여행계획을 생성하는 단일 함수
    """
    # 1) 프롬프트 구성
    prompt = build_prompt(destination, budget, travel_date, preferences, extra)

    # 2) GPT 호출
    response = await client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.4,
        truncation="auto"
    )

    # 3) JSON 텍스트 반환
    return response.output_text