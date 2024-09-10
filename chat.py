import streamlit as st
import pandas as pd
# import csv

#LLM
import os
from groq import Groq
from typing import Dict, List


with st.sidebar:
    st.title("Enter your API Key here")
    API_KEY = st.text_input("")

os.environ["GROQ_API_KEY"] = API_KEY

LLAMA3_405B_INSTRUCT = "llama-3.1-405b-reasoning" # Note: Groq currently only gives access here to paying customers for 405B model
LLAMA3_70B_INSTRUCT = "llama-3.1-70b-versatile"
LLAMA3_8B_INSTRUCT = "llama3.1-8b-instant"
DEFAULT_MODEL = LLAMA3_70B_INSTRUCT
client = Groq()


def assistant(content: str):
    return { "role": "assistant", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def chat_completion(
    messages: List[Dict],
    model = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content
        

def completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    return chat_completion(
        [user(prompt)],
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

def complete_and_print(prompt: str, model: str = DEFAULT_MODEL):
    print(f'==============\n{prompt}\n==============')
    response = completion(prompt, model)
    print(response, end='\n\n')
  

#csv data
csv_material = "MATERIALS_V2.csv"
csv_material_history = "MATERIAL_HISTORY_V2.csv"
# Load the CSV files into DataFrames
material_df = pd.read_csv(csv_material)
material_history_df = pd.read_csv(csv_material_history)


# Extract the relevant columns to use for filtering
material_filter_df = material_df[['MATERIAL NUMBER', 'PLANT']]

# Perform an inner join to filter material_history_df based on material_filter_df
filtered_history_df = pd.merge(material_history_df, material_filter_df, on=['MATERIAL NUMBER', 'PLANT'], how='inner')

merged_df = pd.merge(material_df, filtered_history_df, on=['MATERIAL NUMBER', 'PLANT'], how='inner')

# csv_link = "DATA\MATERIALS.csv"


#INTERFACE
st.header("Celonis Chatbot")

st.write(material_df)
# st.write(filtered_history_df)

#if has data
if merged_df is not None:    

    # get user question
    prompt_assistant = f"""

        <Output in JSON format> 
        Material Number: [Material Number]
        Plant: [Plant]
        Categories: [Provide only top Category]
        Confidence Level: [Confidence Level]
        Recommendation: [Recommendation]
        </output>

        Datasets Provided

        Material Master Data: {material_df}
        Material History Demand: {filtered_history_df}

        Criteria for Output Evaluation. Look into the internet if needed.

        Seasonality: Evaluate seasonal base transactions and patterns.
        Market Trends: Consider industry trends and current market conditions.
        External Factors: Account for significant external factors such as COVID-19 or other disruptions from the internet.
        Lead Time: Assess the timing for replenishing materials and managing lead times effectively.
        Historical Data: Analyze past sales and usage data for accurate forecasting.
        Criteria End
        Reference Categories

        Excess Inventory
        Safety Stock
        Reorder Point
        Lead Time Inventory
        Slow-Moving Inventory
        Obsolete Inventory
        Just-in-Time (JIT) Inventory
        High-Value Inventory
        Seasonal Inventory
        Economic Order Quantity (EOQ)

    """    
    prompt_user = st.text_input("Type Your question here")

    # do similarity search
    if prompt_user:
        response = chat_completion(messages=[
            assistant(prompt_assistant),
            user(prompt_user),
        ])
        # st.text_input("")
        st.write(response)
        # st.write(prompt_assistant)

        

    




