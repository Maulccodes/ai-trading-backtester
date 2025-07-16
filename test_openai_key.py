import openai
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read the key
api_key = os.getenv("OPENAI_API_KEY")

# Show loaded key status
if not api_key:
    print("❌ OPENAI_API_KEY is missing or not loaded.")
else:
    print("✅ API Key Loaded.")

    # Assign key to OpenAI client
    openai.api_key = api_key

    # Simple API call to test
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if you have access
            messages=[
                {"role": "user", "content": "Say hello!"}
            ]
        )
        print("✅ OpenAI API Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print("❌ Error when calling OpenAI API:", e)
