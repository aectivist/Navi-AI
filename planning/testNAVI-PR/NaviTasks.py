import ollama 
import yfinance as yf
from typing import Dict, Any, Callable
import json
import httpx
import subprocess
import time
import shlex
from SettingsAPI_Disc import weather_api_key

Function_Passed = 0

#FUNCTIONS:=======================================================================

#STOCKS
def get_stock_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    return stock.info.get('regularMarketPrice') or stock.fast_info.last_price


#WEATHER 
def get_weather(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  # Use 'imperial' for Fahrenheit
    }

    try: 
        response = httpx.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        json_response = {
            "city": data['name'],
            "temperature": data['main']['temp'],
            "description": data['weather'][0]['description'],
            "humidity": data['main']['humidity'],
            "wind_speed": data['wind']['speed']
        }
        return json.dumps(json_response, indent=4)  # Return as a formatted JSON string
    except httpx.HTTPStatusError as e:
        return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
    
def get_weather_tool(): 
    return {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a given city.',
            'properties': {
                'type': 'string',
                'description': 'Weather information',
            }
        },
        'required': ['city']
    }

def should_enable_tools(prompt: str) -> bool: #specific keywords to enable tools
    keywords = ["stocks", "stock", "weather", "price", "city", "temperature", "humidity"]
    return any(word in prompt.lower() for word in keywords)

#Creates the available functions that can be called by the model

#Generates output using the NAVI model
def NAVI_Output(user_input: str): 
    response = ollama.generate(model="NAVI", prompt=user_input)
    output = str(response["response"])
    print(output)


Function_Found = False ## Flag to check if a function was called
import json
#Tools list for the model to use
tools_list = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_stock_price",
                        "description": "Get the current stock price of a company.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL, TSLA)"}
                            },
                            "required": ["ticker"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name to check weather for"}
                            },
                            "required": ["city"]
                        }
                    }
                }
            ]

#Function to handle the NAVI model and its function calls
def NAVI_FUNCTION():
    global Function_Found
    messages = [ # Initial system message to set context
         {
            "role": "system",
            "content": "You will receive structured data (like JSON strings). When you do, interpret it and explain it clearly in a natural sentence."

         }
    ]

    while True: #Starts looping continuously for user input and ollama output, both being appended for context
        prompt = input("Enter your prompt (type 'exit' to quit): ") 
        if prompt.lower() == 'exit':
            print("Exiting the function call loop.")
            break

        messages.append({'role': "user", "content": prompt})

        # Call model
        
        if should_enable_tools(prompt):
            response = ollama.chat(model="NAVI", messages=messages, tools=tools_list)
        else:
            response = ollama.chat(model="NAVI", messages=messages)

        messages.append(response['message'])

        if not response['message'].get('tool_calls'):
            print("No function calls found in the response.")
            print(response['message']['content'])
            continue

        # Map available functions
        available_functions = {
            "get_stock_price": get_stock_price,
            "get_weather": lambda city: get_weather(city, weather_api_key)
        }

        for tool in response['message']['tool_calls']:
            name = tool['function']['name']
            args = tool['function']['arguments']  

            function_to_call = available_functions.get(name)
            if function_to_call:
                print(f"Calling: {name} with args: {args}")
                result = function_to_call(**args)

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result if isinstance(result, str) else json.dumps(result)
                })

                print("Function Output:", result)
                Function_Found = True
            else:
                print(f"Function {name} not found.")

        
        # Generate final response after all tool calls
        final_response = ollama.chat(model="NAVI", messages=messages)
        print("Final Response:", final_response['message']['content'])
            

                
NAVI_FUNCTION()
 
            


#so cooked







#References:https://www.youtube.com/watch?v=5HxXtvj-mb8, https://www.youtube.com/watch?v=QzRPtorrZVo {just for confirmation that I'm not missing anythin}