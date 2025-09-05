import ollama 
import json
from NaviApiKeys import weather_api_key

from tools.toolsFunctions import get_weather, get_stock_price, code_function, Doing_Search_Now
Navi_Function_Usecase = 0
Function_Passed = 0



def should_enable_tools(prompt: str) -> bool: #specific keywords to enable tools
    keywords = ["me the stocks", "stocks for", "weather", "price for", "city", "temperature", "humidity", "wind", "me a program that", "generate code", "write code", "create a code", 
                "What happened in", "Search up", "Explain to me", 
                "latest version", "give me an update on", "changelog", "What is the",
                "score", "match result", "who won", "standings",
                "election", "poll results", "bill passed", "new law", "sanctions", "when does", "Search"
                ]
    
    if any(word.lower() in prompt.lower() for word in keywords):
        print("Key word detected, finding a match")
        return True
    else:
        print("nothing found loser")
        return False

#Creates the available functions that can be called by the model


Function_Found = False ## Flag to check if a function was called
import json
#Tools list for the model to use
tools_list = [ #Only to note but the weather model was only implemented for proof of concept that it works and Im not restarted 
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "code_function",
                        "description": "Generates code and outputs it into a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_prompt": {"type": "string", "description": "Pass the user's prompt here without interpretation or revision."}
                            },
                            "required": ["user_prompt"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Doing_Search_Now",
                        "description": "Searches items on the web.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_prompt": {"type": "string", "description": "Pass the user's prompt here without interpretation or revision."}
                            },
                            "required": ["user_prompt"]
                        }
                    }
                }
            ]

#Function to handle the NAVI model and its function calls
def NAVI_FUNCTION(prompt, messageX):
    global Function_Found, Navi_Function_Usecase
    Navi_Function_Usecase = 1
    messages = []
    messages.extend(messageX)
    system_notifymess =  {
            "role": "system",
            "content": "You will receive structured data (like JSON strings). When you do, interpret it and explain it clearly in a natural sentence."

         }# Initial system message to set context
    messages.append(system_notifymess)
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

    # Map available functions
    available_functions = {
    "get_stock_price": get_stock_price,
    "get_weather": lambda **kwargs: get_weather(kwargs["city"], weather_api_key),
    "code_function": lambda **kwargs: code_function(kwargs["user_prompt"]),
    "Doing_Search_Now": lambda **kwargs: Doing_Search_Now(kwargs["user_prompt"])
    }

    if 'tool_calls' in response ['message']:
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
    else:
        print("No tools found")
    
    
    # Generate final response after all tool calls
    return messages
    
