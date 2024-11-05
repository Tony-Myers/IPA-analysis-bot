from openai import OpenAI
import os

# Initialize OpenAI client with an environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Test API call
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how can I assist you today?"}]
    )
    print(response)
except Exception as e:
    print("Error:", e)
