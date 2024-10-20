import os
from typing import Union
from fastapi import FastAPI
from dotenv import dotenv_values
from groq import Groq
from pydantic import BaseModel
from typing import List
from textblob import TextBlob

class Item(BaseModel):
    messages:List[str]

config = dotenv_values(".env")
API_KEY=""
for k,v in config.items():
    API_KEY=v


client = Groq(
    api_key=API_KEY
)
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/sentiment")
def read_item(item: Item):
    response_list = []
    for message in item.messages:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following text into English and return only the translated text in double quotes: {message}",
                }
            ],
            model="llama3-8b-8192",
        )
        # Extract the translated message from the response
        translated_message = chat_completion.choices[0].message.content.strip('"')  # Strip the quotes if needed
        blob = TextBlob(translated_message)
        response_list.append({translated_message, blob.sentiment.polarity})
    return response_list
