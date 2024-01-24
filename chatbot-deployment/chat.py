import json
import pandas as pd
import random
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import pipeline

def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

filename = './intents.json'

intents = load_json_file(filename)

def create_df():
    df = pd.DataFrame({
        'Pattern' : [],
        'Tag' : []
    })

    return df

df = create_df()

def extract_json_info(json_file, df):

    for intent in json_file['intents']:

        for pattern in intent['patterns']:

            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag

    return df

df = extract_json_info(intents, df)

labels = df['Tag'].unique().tolist()
labels = [s.strip() for s in labels]

label2id = {label:id for id, label in enumerate(labels)}

model_path = "./model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
bot_name = "AnnaBella"


def get_response(text):
    score = chatbot(text)[0]['score']

    if score < 0.8:
        return "Sorry, I can't answer that."

    label = label2id[chatbot(text)[0]['label']]
    response = random.choice(intents['intents'][label]['responses'])

    return response

if __name__ == "__main__":
    print("Chatbot: Hi! I am your virtual assistance. Feel free to ask, and I'll do my best to provide you with answers and assistance.")
    print("Type 'quit' to exit the chat\n\n")

    while True:
        text = input("You: ").strip().lower()
        if text == 'quit':
            break
        print(f"Chatbot: {get_response(text)}\n\n")