from groq import Groq
import os
import json
import requests
import gradio as gr
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool

client = Groq(api_key = os.getenv('GROQ API_KEY'))
MODEL = 'llama3-70b-8192'

def get_PDP_results(feature_name):
    """Get the tuples that represent the result of PDP where the first value of a tuple is the value of the feature and the second value is the average for the target variable for a given feature name by querying the Flask API."""
    url = f'http://127.0.0.1:4000/results?feature={feature_name}'
    response = requests.get(url)
    if response.status_code == 200:
        return json.dumps(response.json())
    else:
        return json.dumps({"error": "API request failed", "status_code": response.status_code})

def get_shap_results(client_index):
    """Get the message that represent the result of shap values where the first value is the value of the feature and the second value is the importance score of the feature for the target variable for a given client by querying the Flask API."""
    url = f'http://127.0.0.1:4444/shap?client_index={client_index}'
    response = requests.get(url)
    if response.status_code == 200:
        return json.dumps(response.json())
    else:
        return json.dumps({"error": "API request failed", "status_code": response.status_code})

def run_conversation(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "You're a function that utilizes data from the 'get_shap_results' function to analyze the importance score of each feature and the 'get_PDP_results' function to analyze how each feature relates to the target variable, which represents being a good or bad payer. The closer the value is to 1, the more likely it indicates bad payers. Please incorporate the client index or the feature name in your input."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_PDP_results",
                "description": "Get the results of PDP for a given feature name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feature_name": {
                            "type": "string",
                            "description": "The name of the feature (e.g. 'Age')",
                        }
                    },
                    "required": ["feature_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_shap_results",
                "description": "Get the results of shap for a given client, If the SHAP value of a feature is negative, it means that this feature negatively influences the prediction, bringing the predicted value closer to zero. If the SHAP value of a feature is positive, it means that this feature is bringing the target variable closer to 1. The higher the absolute SHAP value, the more influence the feature has on the prediction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_index": {
                            "type": "string",
                            "description": "The index of the client (e.g. '2')",
                        }
                    },
                    "required": ["client_index"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",  
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_PDP_results": get_PDP_results,
            "get_shap_results": get_shap_results,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "get_PDP_results":
                function_response = function_to_call(
                    feature_name=function_args.get("feature_name")
                )
            elif function_name == "get_shap_results":
                function_response = function_to_call(
                    client_index=function_args.get("client_index")
                )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content

# user_prompt = ''''
# If my LoanDuration is 40 months and I increase to 60 months, will this interfere with my loan approval? Think step by step before answering the question
# '''
# print(run_conversation(user_prompt))    

def sentence_builder(client_id, feature):
    if client_id != 0:
        if feature != "NaN":
            user_prompt = f"""How does the {feature} variable affect the approval of my loan? And which variables are most important for the client with index {client_id}? Think step by step before answering the question"""
        else:
            user_prompt = f"""Which variables are most important for the client with index {client_id}? Think step by step before answering the question"""
    else:
        user_prompt = f"""How does the {feature} variable affect the approval of my loan? Think step by step before answering the question"""
    return run_conversation(user_prompt)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Interface(
        sentence_builder,
        [
            gr.Slider(0, 599, value=1, label="Client ID", info="Choose a client ID between 1 and 599 to analyze the most important features for that client. Select 0 if you don't want information about any specific client"),
            gr.Dropdown(
                ["NaN", "LoanDuration", "LoanAmount", "PurpuoseOfLoan", "YearsAtCurrentHome", "LoanRateAsPercentOfIncome", "Age", "NumberOfOtherLoansAtBank"], label="Feature", info="Analyze how each feature relates to the target variable"
            ),   
        ],
        "text",
        # examples=[
        #     [2, "Age"],
        #     [4, "LoanDuration"],
        #     [10, "LoanAmount"],
        # ],
    )

if __name__ == "__main__":
   demo.launch(share=True)

# def gradio_interface(user_prompt):
#     return run_conversation(user_prompt)

# interface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text")
# interface.launch(share=True)
