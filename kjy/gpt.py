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

def build_prompt_from_fields(destination, budget, travel_date, preferences, extra):
    merged_user_input = f"""
destination: {destination}
budget: {budget}
travel_date: {travel_date}
preferences: {preferences}
extra: {extra}
"""
    return SYSTEM_PROMPT + merged_user_input


async def request_itinerary(prompt: str) -> str:
    response = await client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.4,
        truncation="auto"
    )
    return response.output_text


async def generate_itinerary(destination, budget, travel_date, preferences, extra):
    prompt = build_prompt_from_fields(destination, budget, travel_date, preferences, extra)
    result = await request_itinerary(prompt)
    return result


async def main():
    # 예시 입력
    dest = "오사카"
    budget = "약 80만원"
    date = "2025년 3월 1일 ~ 3월 4일"
    prefs = "맛집, 쇼핑, 가벼운 산책"
    extra = "해산물 알레르기 있음, 너무 타이트한 일정 싫음"

    result = await generate_itinerary(dest, budget, date, prefs, extra)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
