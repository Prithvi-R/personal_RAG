import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENAI_KEY"),
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="moonshotai/kimi-k2:free",
  messages=[
    {
      "role": "user",
      "content": "What about mission Sindoor by india it happened in 2025?"
    }
  ]
)

print(completion.choices[0].message.content)
