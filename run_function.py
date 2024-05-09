from promptflow.core import tool
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = "", 
  api_version = "2024-02-01",
  azure_endpoint ="https://confdemo-openai-3rde3acypqeqg.openai.azure.com/"
)

with open('sessions_with_embeddings.json', 'r') as f:
    sessions = json.load(f)
    
# create a function to find sessions by speaker by lowercasing the speaker name and use contains to find the speaker name in the list of speakers & remove the Embedding field from the response
def find_sessions_by_speaker(speaker_name):
    speaker_name = speaker_name.lower()
    ret = [session for session in sessions if any(speaker_name in speaker.lower() for speaker in session['Speakers'])]
    # removes Embedding field
    for session in ret:
        del session['Embedding']
    return ret


# find sessions based on a query
def find_sessions_by_query(query):
    response = client.embeddings.create(
        input = query,
        model= "text-embedding-3-small"
    )
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
    # sort the sessions by similarity & take the top 1 & return the score as well
    for session in sessions:
        session['Similarity'] = cosine_similarity(np.array(session['Embedding']).reshape(1, -1), query_embedding)[0][0]
    sessions.sort(key=lambda x: x['Similarity'], reverse=True)
    # removes Embedding field
    for session in sessions:
        del session['Embedding']
    return sessions[:1]


# def get_current_weather(location, unit="fahrenheit"):
#     """Get the current weather in a given location"""
#     weather_info = {
#         "location": location,
#         "temperature": "72",
#         "unit": unit,
#         "forecast": ["sunny", "windy"],
#     }
#     return weather_info


# def get_n_day_weather_forecast(location, format, num_days):
#     """Get next num_days weather in a given location"""
#     weather_info = {
#         "location": location,
#         "temperature": "60",
#         "format": format,
#         "forecast": ["rainy"],
#         "num_days": num_days,
#     }
#     return weather_info


@tool
def run_function(response_message: dict) -> str:
    function_call = response_message.get("function_call", None)
    if function_call and "name" in function_call and "arguments" in function_call:
        function_name = function_call["name"]
        function_args = json.loads(function_call["arguments"])
        print(function_args)
        result = globals()[function_name](**function_args)
    else:
        print("No function call")
        if isinstance(response_message, dict):
            result = response_message.get("content", "")
        else:
            result = response_message
    return result
