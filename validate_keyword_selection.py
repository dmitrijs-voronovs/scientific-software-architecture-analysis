import json
import os
import re
from pathlib import Path

import dotenv
import google.generativeai as genai
import pandas as pd
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError, wait_incrementing
from tqdm import tqdm

# Load environment variables from .env file
dotenv.load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_TOKEN_2")

YOUR_APP_NAME = "Keyword Selection"


def request_llm(prompt):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "X-Title": f"{YOUR_APP_NAME}",  # Optional. Shows in rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": "google/gemma-2-9b-it:free",  # Optional
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who provides thoughtful and concise responses and tightly follows user instroctuctions. You never output any extra information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        })
    )
    return response.json()


def get_resp(index, row):
    try:
        resp = request_llm(row["prompt"])
        if resp['error']:
            print(f"Error in row {index}, {resp['error']}")
            exit(1)
        text_resp = resp['choices'][0]['message']['content']
        text_resp = re.sub(r'```json|```', "", text_resp)
        json_resp = json.loads(text_resp)
        print(json_resp)
        r = (json_resp['false_positive'], json_resp['reasoning'])
    except Exception as e:
        print(f"Error in row {index}, {e}")
        r = (None, None)
    return r


@retry(
    stop=stop_after_attempt(5),
    wait=wait_incrementing(5, 10),
    before_sleep=lambda retry_state: print(retry_state),
)
def request_google_ailab(model, prompt):
    response = model.generate_content(prompt)
    text_resp = re.sub(r'```json|```', "", response.text)
    json_resp = json.loads(text_resp)
    print(json_resp)
    return json_resp['false_positive'], json_resp['reasoning']


def main():
    file_path = Path("./metadata/keywords/verification/sample.csv")

    genai.configure(api_key=os.getenv("GOOGLE_AI_STUDIO_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # df = pd.read_csv(file_path).sample(10)
    df = pd.read_csv(file_path)
    res = []
    res_filename = "sample_with_responses_ailab_all2"
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # r = get_resp(index, row)
        try:
            r = request_google_ailab(model, row["prompt"])
            res.append(r)

            if (index + 1) % 2 == 0:
                df_to_save = df.iloc[0:index + 1].copy()
                df_to_save['false_positive'], df_to_save['reasoning'] = zip(*res)
                df_to_save.to_csv(file_path.with_stem(res_filename), index=False)
        except RetryError as e:
            print(f"Error in row {index}, {e}")
            exit(1)

    df['false_positive'], df['reasoning'] = zip(*res)
    df.to_csv(file_path.with_stem(res_filename), index=False)


if __name__ == "__main__":
    main()
