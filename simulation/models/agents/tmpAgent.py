from openai import OpenAI

client = OpenAI(
    base_url="https://mj.chatgptten.com/v1",
    api_key="sk-IMblefS5KQ5ET8izUvenvX71tOXiIZDp3ICQ33mFcUtKV8lq"
)

response = client.chat.completions.create(
  model="gpt-4o",

  messages=[
      {"role": "user", "content": "天津大学怎么样。"}
  ],
  # timeout=100,

)
print(response.choices[0].message.content)