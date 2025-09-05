import json
import httpx
import ollama

import yfinance as yf

import requests
import trafilatura #extract from main site without any rules
from bs4 import BeautifulSoup

"""GET WEATHER"""
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
    
"""CODE GENERATION"""

def code_function(user_prompt: str):
    print(user_prompt)
    try:
        filetype_prompt = f"ONLY provide the file type related to the prompt (py, cpp, c, etc.): {user_prompt}. Do not actually provide code, just the type of file the prompt could be using, without any comments or remarks, no periods, no words such as python or c sharp, not a single sentence other than the file type (py, cpp, c, java, etc.) as it will be used for naming the file." #asks for file type (I know its redudant but it wont listen to me and Im kinda noob)
        filetype = ollama.generate(model="codellama", prompt=filetype_prompt) #creates the code
        
        filetype_output = filetype["response"].strip()  # Get the response and strip any extra whitespace
        print(f"Filetype generated: {filetype_output}")
        
        File_Name = str(f"code_generated.{filetype_output}") #creates file name 
        print(f"File name generated: {File_Name}")
        
        user_prompt = f"Create a code snippet based on the following prompt, do not provide any comments, quotations, or explanations, just the code: {user_prompt}"  # Prepare the prompt for code generation
        response = ollama.generate(model="codellama", prompt=user_prompt) #creates code
        code = response["response"]

        code_temp = code.strip()  
        for character in code_temp:
            if character == '`':
                code_temp = code_temp.replace(character, " ")

        with open(f"main/output/{File_Name}", 'w') as file:
            file.write(code_temp)
        
        return print(f"Code saved to {File_Name}")
    except Exception as e:
        print(f"Error generating code: {e}")
    
"""STOCK FUNCTION"""
def get_stock_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    return stock.info.get('regularMarketPrice') or stock.fast_info.last_price





"""SEARCH FUNCTION"""
Search_Function_Worked = {
        'role': 'system', 
        'content': (
            'You are an AI assistant that has another AI model working to get you live data from search engine results that will be attached before a USER PROMPT. analyze the SEARCH RESULT and use relevant data to generate the most useful and intelligent response an AI assistant that always impresses the user would generate. Additionally, if requested for sources, provide them accordingly.'
        )
    }

TorF_Function_Search = {
        'role': 'system', 
        'content': 
            'You are not an AI assistant. Your only task is to decide if the last user prompt in a conversation with an AI assistant requires more data to be retrieved from searching Google for the assistant to respond correctly. The conversation may or may not already have exactly the context data needed. If the assistant should search google for more data before responding to ensure a correct response, simply respond "True". If the conversation already has the context, or a Google search is not what an intelligent human would do to respond correctly to the last message in the convo, respond "False". Do not generate any explanations. Only generate "True" or "False" as a response in this conversation using the logic in these instructions.'
    }

Best_Search_Msg = {'role': 'system',
    'content':'You are not an AI assistant that responds to a user. You are an AI model trained to select the best search result out of a list of ten results. The bes search result is the link to an expert human search engine user would click first to find the data to respond to a USER_PROMTP after searching DuckDuckGo for the SEARCH_QUERY. \n All users messages you receive in this conversation will have the format of: \nSEARCH_RESULTS: [{},{},{}] \nUSER_PROMPT: "This will be an actual prompt to a web search enabled AI assistant"\nSEARCH_QUERY: "Search query ran to get the above X links \n\n You must select the index from the 0 indexed SEARCH_RESULTS list and only respond with the index of the best search result to check for the data the AI assistant needs to respond. That means your response to this conversation should always be 1 token, being an integer between 0-X'}

Query_Msg = {'role': 'system',
    'content': 'You are NOT an AI assistant that responds to a user. You are an AI web search query model. You will be given a prompt to an AI assistant with web search capabilities. You must determine what the data is the assistant needs from search and generate the best possible DuckDuckGo query to find that data. Do not respond with anything but a query that an expert human search engine would type into DuckDuckGo to find the needed data. Keep your queries simple, without any search engine code. Just type a query likely to retrieve the data we need. At the same time, do not explain that you cannot provide real-time information or that you cannot help, just rephrase the users prompt as needed for the entire AI search engine to run.'}

contains_data_msg = {
    'role': 'system',
    'content':'You are not an AI assistant that responds to a user. You are an AI model designed to analyze data scraiped from web pages text to assist an actual AI assistant in responding correctly with up to date information. Consider the USER_PROMPT that was sent to the actual AI assistant and analyze the web PAGE_TEXT to see if it does contain the data needed to construct an intelligent, correct response. This web PAGE_TEXT was retrieved from a search engine using the SEARCH_QUERY that is also attached to user messages in this conversation. All user messages in this conversation will have the format of: \n PAGE_TEXT: "entire page text from the best search results based off the search snippet." \n USER_PROMPT: "The prompt sent to an asctual web search enabled AI assisatant." \n SEARCH_QUERY: "the search query that was used to find data determined necessary for the assistant to respond correctly and usefully. \n You must determine whether the PAGE_TEXT actually contains reliable and necessary data for the AI assistant to respond. However, once you come to a conclusion, do not provide your own reason why, and instead, only ensure that you add true or false if the page contains data or not. Ex: "This page contains data. true.", or "This page does not contain data. false.'}


search_assistant_msg = []
def TorF_search_function(userprompt):
    # Add the "decider" system prompt
    torf = [TorF_Function_Search, {
        'role': 'user',
        'content': userprompt
        }]
    
    print(torf)
    # Pass the WHOLE list of messages (not as content!)
    response = ollama.chat(
        model='searchbot',
        messages=torf
    )

    content = response['message']['content'].strip()
    print(content)
    print("This printed")
    if 'true' in content.lower():
        print("T")
        return True
    else:
        print("F")
        return False

def Query_Generator(userprompt):
    print("QUERY...")
    Query_Prompt = f'{userprompt}'
    print(Query_Prompt)
    response = ollama.chat(model='searchbot', messages=[Query_Msg, {'role': 'user', 'content': Query_Prompt}])
    context = response['message']['content'].strip()
    print("Generated query:", repr(context))
    print(context)

    generation_failed_checker = ["I can't", "I am unable to"]
    if any(word.lower() in context.lower() for word in generation_failed_checker):
        return userprompt

    if not context:
        return userprompt
    else:
        return context

def duckduckgo_Search(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    }
    url = f'https://html.duckduckgo.com/html/?q={query}'
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    results = []

    for i, result in enumerate(soup.find_all('div', class_='result'), start=1):
        if i > 5:  # limit to 10
            break

        title_tag = result.find('a', class_='result__a')
        if not title_tag:
            continue

        link = title_tag['href']
        if link.startswith("//"):
            link = "https:" + link

        # Try different places where snippets appear
        snippet_tag = (
            result.find('div', class_='result__snippet') or
            result.find('a', class_='result__snippet') or
            result.find('span', class_='result__snippet') or
            result.find('div', class_='result__body')
        )

        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else "No description available"

        results.append({
            'id': i,
            'link': link,
            'title': title_tag.get_text(" ", strip=True),
            'search_description': snippet
        })

    return results


def best_Search(s_results, query):
    global search_assistant_msg
    sys_msg = Best_Search_Msg
    best_msg = f'SEARCH_RESULTS: {s_results} \nUSER_PROMPT: {search_assistant_msg[-1]["content"]} \nSEARCH_QUERY: {query}'

    for _ in range(2):
        try:
            response = ollama.chat(model='searchbot', messages=[sys_msg,{'role': 'user', 'content': best_msg}])
            return int(response['message']['content'].strip())
        except:
            continue
    return 0 #if nada comes up then it'll just provide 0

from urllib.parse import urlparse, parse_qs, unquote

def clean_duckduckgo_url(url: str) -> str:
    """If it's a DuckDuckGo redirect, extract the real target URL"""
    if "duckduckgo.com/l/?" in url:
        qs = parse_qs(urlparse(url).query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return url

def scrape_webpage(url):
    try: 
        real_url = clean_duckduckgo_url(url)
        downloaded = trafilatura.fetch_url(real_url)
        extracted = trafilatura.extract(
            downloaded,
            include_formatting=True,
            include_links=True,
            favor_recall=True
        )
        return extracted
    except Exception as e:
        print(e)
        return None

def ai_search(userprompt, queryGen): #web search context prommpts
    context = None
    print("GENERATING SEARCH QUERY.")
    search_query = queryGen
    print(search_query)
    if not search_query:
        return None

    if search_query.startswith('"') and search_query.endswith('"'):
        search_query = search_query[1:-1]

    search_results = duckduckgo_Search(search_query)
    print(search_results)

    for _ in range(5):  # max 10 attempts
        if not search_results:
            break
        best_result = best_Search(search_results, search_query)
        print(best_result)
        try:
            page_link = search_results[best_result]['link']
            print(page_link)
        except:
            print('FAILED TO SELECT BEST SEARCH RESULT, TRYING AGAIN')
            continue

        page_text = scrape_webpage(page_link)
        
        print(page_text)
        search_results.pop(best_result)

        if contains_data_needed(page_text, search_query):
            context_search_bot.append(page_text)
           
        
    return context_search_bot

context_search_bot = []
def contains_data_needed(search_content, query):
    sys_msg = contains_data_msg
    needed_prompt = f'PAGE_TEXT: {search_content} \nUSER_PROMPT: {search_assistant_msg[-1]["content"]} \nSEARCH_QUERY: {query}'
    response = ollama.chat(
    model='searchbot',
    messages=[sys_msg, {'role': 'user', 'content': needed_prompt}]
    )

    content = response['message']['content'].strip()
    global context_search_bot
    print("contains data: " + content)
    if "true" in content.lower():
        return True
    else:
        return False


def Doing_Search_Now(userprompt: str):
    global search_assistant_msg, context_search_bot
    search_assistant_msg.append({'role':'user', 'content': userprompt})
    print("Check")
    search_query = Query_Generator(userprompt)
    if TorF_search_function(userprompt):
        print("WEB SEARCH REQUIRED")
        print(search_assistant_msg)
        context = ai_search(userprompt,search_query)
        if context:
            search_assistant_msg[-1] = {'role': 'user', 'content': f'SEARCH RESULT: {context} \n\nUSERPROMPT: {userprompt}'}
        else:
            search_assistant_msg[-1] = {'role': 'user', 'content': f'USER PROMPT: \n{userprompt} \n\nFAILED SEARCH: ...'}

        if context:
            prompt = f'SEARCH RESULT: {context} \n\nUSERPROMPT: {userprompt} \n\n INSTRUCTIONS: Analyze the SEARCH RESULT given, as they are derived from a web search.'
        else:
            prompt = (f'USER PROMPT: \n{userprompt} \n\nFAILED SEARCH: The AI could not extract any data. Please explain and ask if the user would like to search again or respond without web search context. Do not respond if a search was needed and you are getting this message with anything but the above request of how the user would like to proceed')
        print(prompt)
        search_assistant_msg.append({'role':'user', 'content': prompt})
        print(f"saearch assst {search_assistant_msg}")
        return prompt
    
#https://www.youtube.com/watch?v=9KKnNh89AGU
