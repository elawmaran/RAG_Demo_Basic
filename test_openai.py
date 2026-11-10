import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load variables from .env
load_dotenv()

# 2. Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# 3. Make sure the key exists
if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in .env")

# 4. Create the OpenAI client
client = OpenAI(api_key=api_key)

print("OpenAI API key loaded successfully.")

# 5. Send a simple request to OpenAI
response = client.responses.create(
    model="gpt-5-mini",
    input="Say hello in one short sentence."
)

# 6. Print the answer
print("\nOpenAI response:")
print(response.output_text)