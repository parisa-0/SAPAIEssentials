%md
If you haven't started the course on Success Map Learning yet, please begin the course ["Code-Based Agents"](https://performancemanager5.successfactors.eu/sf/learning?destUrl=https%3A%2F%2Fsap%2eplateau%2ecom%2Flearning%2Fuser%2Fdeeplink%5fredirect%2ejsp%3FlinkId%3DITEM%5fDETAILS%26componentID%3DDEV%5f00010190%5fWBT%26componentTypeID%3DCOURSE%26revisionDate%3D1746006862000%26fromSF%3DY&company=SAP).
%md
# ReAct Agent From Scratch Using SAP's Generative AI Hub

#### 1. Introduction
In this lesson, you will be introduced to [ReAct Agent](https://arxiv.org/abs/2210.03629)—an innovative approach for building AI agents that combine reasoning and action. Unlike traditional prompt-based interactions, REACT agents use a structured chain-of-thought that guides the agent to generate a “Thought,” decide on an “Action” (such as calling a tool), observe the result, and then either continue iterating or provide a final answer. This iterative thought–action–observation loop is a powerful paradigm for creating intelligent, code-based agents capable of interacting with external systems.

#### 2. Key Concepts
- **REACT Pattern:**  
  - **Thought:** The agent internally reasons about the request.  
  - **Action:** The agent decides on a tool to call and supplies it with parameters (formatted in JSON).  
  - **Observation:** The result returned by the tool is incorporated back into the agent’s chain of thought.  
  - **Final Answer:** When the agent is satisfied, it terminates the loop with a final answer.
  
- **Tool Integration:**  
  - REACT agents can dynamically call external tools (e.g., an API's to perform operations like checking order statuses or fetching business data.
  
- **Iterative Reasoning:**  
  - The agent continuously updates its chain-of-thought based on observations, which helps it refine its responses through multiple iterations.
%md
### 🔐 Important Notice: Training Environment API Keys

The API keys used in this notebook are provided strictly for training purposes.
These keys:
- incur real consumption cost,
- are intended only for use within this specific training notebook, and
- must not be used for any other purpose, including personal experiments, external scripts, or private proof‑of‑concepts.

Any use of these keys **outside this controlled training environment** is prohibited.
Unauthorized usage may be treated as a **compliance violation** and can lead to a **cybersecurity incident response**, including revocation of access and escalation to the security team.
Please handle these credentials responsibly.

🍃 If you would like to experiment with AI Core services beyond this training notebook, you can request your own set of API keys for your experiments using the link [Generative AI Experience Lab Services](https://sap.sharepoint.com/sites/205734/SitePages/Generative-AI-Experience-Lab.aspx?xsdata=MDV8MDJ8fDk4ODJmZDJhYTUzNTQ0MjBkMzJkMDhkZGM0NGYxYjE0fDQyZjc2NzZjZjQ1NTQyM2M4MmY2ZGMyZDk5NzkxYWY3fDB8MHw2Mzg4ODI1NjYxOTIyODg1NTd8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDJOb1lYUnpMekU1T2pOaU1qUTNOekV4TFRoaFl6VXRORE16TnkwNFpUTTJMVFl3TVdVell6TTNPVEJoTkY4NU1HRXlNemt6WXkwM1pHUm1MVFJtWWpZdFlqZzJaQzAyTTJZME5USmtOVEUwTURSQWRXNXhMbWRpYkM1emNHRmpaWE12YldWemMyRm5aWE12TVRjMU1qWTFPVGd4T0RjNU53PT18Yzc3OTA3NGI0MDkwNDM4MWQzMmQwOGRkYzQ0ZjFiMTR8ZGNmNmNjMmQxNmIwNDVjYmI3NjU3NWI3MGM4ZTEyZjE%3D&sdata=U0JEVUthcEJxc0lFVHFoVUlvM3VlRy85TVVua1MvUExSSkljSFdBK1B3UT0%3D&ovuser=42f7676c-f455-423c-82f6-dc2d99791af7%2Cjenifer.sam%40sap.com&OR=Teams-HL&CT=1753250130963&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiI1MC8yNTA3MDMxODgwNiIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D#generative-ai-experience-lab-services).

🍃 If you want to productize an AI use case at SAP, you will need to [provision an AI Core service in your BTP subaccount](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/enabling-service-in-cloud-foundry).

from IPython.display import IFrame

# URL of the video you want to embed
video_url = "https://video.sap.com/embed/secure/iframe/entryId/1_c0hmw9fx/uiConfId/54310412/st/0"

# Embed the video using IFrame
IFrame(src=video_url, width=768, height=432)
%md
To display the video in a separate browser window, you can click on [React Agents from Scratch Hands On Part 1](https://video.sap.com/media/t/1_c0hmw9fx)
%pip install -q "sap-ai-sdk-gen[all]==5.6.3" "wikipedia"
dbutils.library.restartPython()
# We set a 'reasoning' model as default for this lesson.
dbutils.widgets.dropdown("llm_model", "o4-mini", ["o4-mini","gpt-5", "gpt-5-mini", "gemini-2.5-pro"])
%md
This Lesson is based on [Mathis Boerner's talk to Devtoberfest 2024](https://www.youtube.com/watch?v=NgWwuJ7mya0)
%md
#### Setup Orchestration Service

We'll initialize the orchestration service to interact with SAP's Generative AI Hub. This service allows us to configure and run LLM requests with templates.

If you haven't completed it yet, consider the module [Orchestration Workflows with Generative AI Hub SDK](https://performancemanager5.successfactors.eu/sf/learning?destUrl=https%3A%2F%2Fsap%2eplateau%2ecom%2Flearning%2Fuser%2Fdeeplink%5fredirect%2ejsp%3FlinkId%3DITEM%5fDETAILS%26componentID%3DDEV%5f00010196%5fWBT%26componentTypeID%3DCOURSE%26revisionDate%3D1746020851000%26fromSF%3DY&company=SAP) for a comprehensive introduction.
import os
import json
    
# Load your service key from a Secret Store
secret = dbutils.secrets.get(scope="PROD_XGTP_SCOPE", key="LEARNING_GENAIXL")
svcKey = json.loads(secret) 
# Set environment variables
os.environ["AICORE_AUTH_URL"] = svcKey["url"]
os.environ["AICORE_CLIENT_ID"] = svcKey["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = svcKey["clientsecret"]
os.environ["AICORE_RESOURCE_GROUP"] = "default"
os.environ["AICORE_BASE_URL"] = svcKey["serviceurls"]["AI_API_URL"]
%md
#### Import Required Libraries

We'll import the necessary components from the Generative AI Hub SDK to work with orchestration.
from typing import List, Callable
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
%md
#### Helper Function to run LLM requests

We will define a simple helper function that accepts a prompt template, model name and values for the placeholders in the prompt template.
To understand how the Orchestration service works we advise completing the following course: [Orchestration Workflows With SAP Cloud SDK for AI](https://performancemanager5.successfactors.eu/sf/learning?destUrl=https://sap.plateau.com/learning/user/deeplink_redirect.jsp?linkId%3DITEM_DETAILS%26componentID%3DDEV_00010196_WBT%26componentTypeID%3DCOURSE%26revisionDate%3D1746020851000%26fromSF%3DY&company=SAP)
# Initialize the orchestration service
# This simple initialization automatically handles the deployment setup
orchestration_service = OrchestrationService()
def send_request(prompt, _print=False, _model='o4-mini', **kwargs):
    config = OrchestrationConfig(
        llm=LLM(name=_model, parameters={"max_tokens": 8192}),
        template=Template(messages=[UserMessage(prompt)])
    )
    template_values = [TemplateValue(name=key, value=value) for key, value in kwargs.items()]
    answer = orchestration_service.run(config=config, template_values=template_values)
    result = answer.module_results.llm.choices[0].message.content 
    if _print:
        formatted_prompt = answer.module_results.templating[0].content
        print(f"<-- PROMPT --->\n{formatted_prompt if _print else prompt}\n<--- RESPONSE --->\n{result}")  
    return result

template = "Write a haiku about {{?topic}}"

_ = send_request(template, topic="SAP AI Developer Foundation Program", _print=True, _model=dbutils.widgets.get("llm_model"))
%md
## Building a ReAct Agent from Scratch

We need to define tools for the agent. To simplify this we will write a helper class that extracts the name of the tool, description and input schema from a python function definition.

from IPython.display import IFrame

# URL of the video you want to embed
video_url = "https://video.sap.com/embed/secure/iframe/entryId/1_krw756ag/uiConfId/54310412/st/0"

# Embed the video using IFrame
IFrame(src=video_url, width=768, height=432)
%md
To display the video in a separate browser window, you can click on [React Agents from Scratch Hands On Part 2](https://video.sap.com/media/t/1_krw756ag)
from dataclasses import dataclass
from typing import Callable
import json

@dataclass
class ToolInfo:
    name: str
    description: str
    schema: dict

    @classmethod
    def from_callable(cls, func: Callable):
        # Get the function's name
        func_name = func.__name__
        # Get the function's docstring, default to an empty string if it's None
        func_description = func.__doc__ or ""
        # Get the function's type annotations
        annotations = func.__annotations__
        # Filter out any arguments without a type annotation
        schema = {k: str(v) for k, v in annotations.items() if k != 'return'}
        # Return an instance of ToolInfo with the extracted name, description, and schema
        return cls(name=func_name, description=func_description, schema=schema)

    def __str__(self):
        return f"Tool: {self.name}\nUsage Info: {self.description}\nSchema: {self.schema}"
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

print(str(ToolInfo.from_callable(add)))

%md
 **REACT Template**  
  The prompt instructs the LLM to follow the `Thought -> Action -> Observation` flow, whose output is a JSON containing an `action` and an `input`. If the LLM determines that a tool should be invoked, it specifies the tool name (e.g., `"CheckOrderStatus"`) and provides parameters in JSON format. When the agent is ready to provide the final answer, it outputs an action `"Finish"`.
REACT_PROMPT = """Respond to the human as helpfully and accurately as possible. You are an agent that gets a set of tools and can use these tools in actions.

The goal is to provide an answer to the query of the user. To find the correct answer you can use tools. As input your are given the user query and steps already taken.
Your response should always follow the pattern shown below. You in your response you have to fill ... with actual output. Text in [] is to explain what kind of output is expected.

--- start: response format in case of an action---
Thought: ... [Your though on what to do next]
Action:
```
... [A json blob describing the next action]
```
Observation: [always finish with "Observation" to trigger the execution of the action and to see the result]
--- end: response format in case of an action---
If you don't have to use a tool because you know the final answer use the finish tool to return this answer. 

You have access to the following tools:

{{?tools}}

Use a json blob to specify a tool by providing an "name" key (tool name) and an "input" key (tool input).
Provide only ONE action per json, as shown:
--- start: json blob ---
{
  "name": ... [Name of the tool, valid tools, are: {{?tool_names}}]
  "input": ... [Arguments of the tool, expected schemas of the tools are listed above]
}
--- end: json blob ---
You can't use comments in the json blob.


To recap follow this format:

--- start: recap ---
User Query: ... [input query to answer]
Thought: ... [your thoughts on the action to take]
Action:
```
{
  "name": ... [Name of the tool, valid tools, are: {{?tool_names}}]
  "input": ... [Arguments of the tool, expected schemas of the tools are listed above]
}
```
Observation: ... [Result of the action]

... [repeat Thought/Action/Observation N times]

Thought: I know what to respond
Action:
```
{
  "name": "finish",
  "input": {
      "answer": ... [The final answer to return the result of exeucting the to plan to answer the original input. Using the "finish" tool is the only way to finish the conversation.]
  }
}
```
--- end: recap ---

Begin! Always start your response with "Thought" and finish by an Action + json blob with key "name" and "input". No text outside that schema. Only one Action per response. Never give an Observation as part of your response.

User Query: {{?query}}

{{?scratchpad}}"""
%md
Now we need the code that:
1. Calls the LLM
2. Parses the response
3. Executes the tools
4. Triggers the next agent loop is not finished
# helper functions to parse the response and append observation to the response
import re

def extract_and_parse_json(text):
    # Use regex to find text encapsulated within ``` ... ``` or ```json ... ```blocks
    pattern = r"```(?:json)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        json_text = match.group(1).strip()
        try:
            # Parse the extracted JSON-like text
            parsed_json = json.loads(json_text)
            return parsed_json
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format: {text}")
    else:
        return ValueError(f"No JSON found in text: {text}")

def append_for_scratchpad(input_text: str, value: str, suffix: str = "Observation: ") -> str:
    # Split the input text by lines
    lines = input_text.splitlines()
    last_line = lines[-1]
    desired_line = f"{suffix}{value}"
    for c_exisiting, c_wanted in zip(last_line, desired_line):
        if c_exisiting != c_wanted:
            return f"{input_text}\n{desired_line}"
    return f"{input_text}{desired_line[len(last_line):]}"

def print_scratchpad(scratchpad):
    for i, step in enumerate(scratchpad):
        print(f"\x1b[31m >>> STEP {i+1} <<< \x1b[0m" + f"\n{step}")
# Tool to return the final answer
class FinalAnswer(Exception):
    def __init__(self, text: str, *args, **kwargs):
        self.text = text
        super().__init__(*args, **kwargs)

def finish(answer: str) -> str:
    """Give the final answer to the user query."""
    raise FinalAnswer(text=answer) 


import traceback

class ReactAgent:
    def __init__(self, tools: List[Callable], model=dbutils.widgets.get("llm_model")):
        self.model = model
        self.tools = {}
        self.tool_names = []
        self.tool_descriptions = []
        for tool in tools + [finish]: # always add the finish tool
            tool_info = ToolInfo.from_callable(tool)
            self.tool_names.append(tool_info.name)
            self.tool_descriptions.append(str(tool_info))
            self.tools[tool_info.name] = tool
    
    
    def run(self, query: str, max_turns: int = 20):
        scratchpad = []
        turns = 0
        while True and turns < max_turns:
            response = send_request(
                REACT_PROMPT,
                _model=self.model,
                _print=False,
                query=query,
                scratchpad="\n".join(scratchpad),
                tools="\n\n".join(self.tool_descriptions),
                tool_names=", ".join(self.tool_names)
            )
            try:
                action = extract_and_parse_json(response)
                observation = str(self.tools[action["name"]](**action["input"]))
            except FinalAnswer as answer:
                scratchpad.append(append_for_scratchpad(response, answer.text))
                return answer.text, scratchpad
            except Exception as e:
                observation = "Error when calling the tool" + "\n".join(traceback.format_exc().splitlines()[-10:])
            scratchpad.append(append_for_scratchpad(response, observation))
            turns += 1
        return None, scratchpad          
%md
Now, let's test the agent.

As is well known, LLMs often struggle with even simple mathematical calculations. Therefore, we will provide our agent with tools for addition, subtraction, multiplication, and division.
# Tools
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def divide(a: int, b: int) -> int:
    """Divide two numbers."""
    return a / b
agent = ReactAgent([add, multiply, subtract, divide])
res, scratchpad = agent.run("What is 5+5*2-10?")
%md
Let us take a look at the scratchpad.
print_scratchpad(scratchpad)
%md
Obviously it is a waste of ressources to use all this power to do elementary school level arithmetic . We will move now to a more business oriented scenario. We introduce below a tool that can craft SQL queries to fetch products from a database table stored on Databricks. It contains synthetic data about products and orders, you can get a sample by running the following query:

%sql
SELECT * FROM xgtp_prod_data.`prod-int_challenge_206`.synthesized_orders LIMIT 10;
%md
Now we can define our function, which relies the selected LLM to create a SQL query mathing the user input in natural language.
from IPython.display import IFrame

# URL of the video you want to embed
video_url = "https://video.sap.com/embed/secure/iframe/entryId/1_rsx6zb6i/uiConfId/54310412/st/0"

# Embed the video using IFrame
IFrame(src=video_url, width=768, height=432)
%md
To display the video in a separate browser window, you can click on [React Agents from Scratch Hands On Part 3](https://video.sap.com/media/t/1_rsx6zb6i)
def sql_retriever(user_query: str) -> str:
    """Executes a Databricks Spark SQL Query on the Product table that contains the columns order_ID, quantity, product_ID, customer_ID and country. Provide your query in natural language. This function will translate it into SQL and execute it."""
    
    # Create a template for SQL generation using the orchestration service
    sql_template = Template(
        messages=[
            SystemMessage(
                "You are a SQL expert. Generate valid SQL queries based on user requests. "
                "You have access to the table xgtp_prod_data.`prod-int_challenge_206`.synthesized_orders. "
                "The table contains these columns: order_ID, quantity, product_ID, customer_ID and country. "
                "Your response must be a valid SQL query only. Do not include any explanations or markdown formatting."
            ),
            UserMessage("Generate a SQL query for: {{?user_message}}")
        ]
    )
    
    # Configure the LLM and orchestration
    sql_config = OrchestrationConfig(
        llm=LLM(name=dbutils.widgets.get("llm_model"), parameters={"max_tokens": 8192}),
        template=sql_template
    )
    
    # Generate the SQL query using orchestration service
    response = orchestration_service.run(
        config=sql_config,
        template_values=[TemplateValue(name="user_message", value=user_query)]
    )
    
    # Extract and clean the SQL query
    query = response.orchestration_result.choices[0].message.content
    query = query.strip().rstrip('```').lstrip('```').lstrip('```sql')
    print(f"SQL query: {query}")
    
    # Execute the query and return results
    result_df = spark.sql(query)
    result_str = result_df.toPandas().to_string(index=False)
    print(result_str)
    return result_str
sql_retriever("What is the total quantity of products sold in the USA?")
%md
To confirm, we run the same query in plain SQL:
%sql
SELECT SUM(quantity) AS total_quantity_sold
FROM xgtp_prod_data.`prod-int_challenge_206`.synthesized_orders
WHERE country = 'USA'
%md
Now we introduce another tool example, read the weather forecast for a certain location using a web service.
import requests
def fetch_weather(city: str, day_range: int = 1, unit: str = 'm') -> str:
    """
    Fetches the weather forecast for a specified city for a given number of days.

    Args:
        city (str): The name of the city to fetch the weather for.
        day_range (int, optional): The number of days to fetch the weather forecast for (up to 3 days). Defaults to 1.
        unit (str, optional): The unit of measurement for temperature ('m' for metric, 'e' for imperial). Defaults to 'm'.

    Returns:
        str: A string containing the weather forecast for the specified city and number of days.
    """
    api_key = 'c322ef22435d40bfa2ef22435df0bfbe'
    base_url = 'https://api.weather.com/v3'
    
    # Search for location
    location_search_url = f"{base_url}/location/search?query={city}&language=en-US&format=json&apiKey={api_key}"
    location_response = requests.get(location_search_url)
    location_data = location_response.json()
    
    city_name = location_data['location']['address'][0]
    place_id = location_data['location']['placeId'][0]
    
    # Fetch weather forecast
    weather_url = f"{base_url}/wx/forecast/daily/3day?placeid={place_id}&units={unit}&language=en-US&format=json&apiKey={api_key}"
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()
    
    weather_result = ""
    for i in range(min(day_range, 3)):
        weather_result += f"{city_name}, on {weather_data['dayOfWeek'][i]} temperature will be between {weather_data['calendarDayTemperatureMin'][i]} °C and {weather_data['calendarDayTemperatureMax'][i]} °C. Details: {weather_data['narrative'][i]}\n"
    return weather_result.strip()
%md
Let's test the weather function
fetch_weather('London', 2)
task = "Assuming all French customers are in Paris, collect the 5 customer ids with the biggest orders. I want to message them to tell that their deliveries will be on time if the weather is mild."
agent = ReactAgent([fetch_weather, sql_retriever])
res, scratchpad = agent.run(task)
print_scratchpad(scratchpad)
task = "Are there any major city in the East coast where deliveries would be affected because of Blizzard in the next three dayt what is the total amount of products that would be at risk?"
%md
This may take some time...
res, scratchpad = agent.run(task)
print_scratchpad(scratchpad)
%md



You can proceed to [Lesson 2](https://sap.plateau.com/icontent_e/CUSTOM/sap/self-managed/sap_content/DEV/code_based_agents/scormcontent/index.html#/lessons/lAlzArYzcl70XvrKDNu0SClrJcw0RIq0) for an overview of Agentic Frameworks. 

You can also choose to explore more creating new ReAct agent instances using the tools sugested below.

Please remember to complete the [Knowledge Check](https://sap.plateau.com/icontent_e/CUSTOM/sap/self-managed/sap_content/DEV/code_based_agents/scormcontent/index.html#/lessons/SprhGS6gCn471FRFftf8yMPomA-DFMn5) on Success Map Learning, as this is necessary for recording the completion of the course. Additionally, ensure that you finish and submit the **Hands-on Assignment** on the ML Workbench to qualify for the **Credly Badge** for AI Developer Skills. Thank you!
%md

#### Going beyond

You can explore the tool suggestions below and crete different agents with the today and Wikisearch tool to solve different kinds of tasks. 
from datetime import datetime
def today():
    """Get today's time in isoformat."""
    return datetime.now()
agent = ReactAgent([today])
res, scratchpad = agent.run("How many days ago was the 2024 Super Bowl? Spend some thoughts on own to do proper calculations with dates.")
print_scratchpad(scratchpad)
"""Util that calls Wikipedia."""
from typing import Any, Optional
import logging
import wikipedia

WIKIPEDIA_MAX_QUERY_LENGTH = 300

class WikipediaAPIWrapper:
    """Wrapper around WikipediaAPI.

    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results. It limits the content by doc_content_chars_max.
    """

    def __init__(self, top_k_results=3, lang="en", doc_content_chars_max=1000):
        """Initialize WikipediaAPIWrapper with optional parameters."""
        self.top_k_results = top_k_results
        self.doc_content_chars_max = doc_content_chars_max
        wikipedia.set_lang(lang)

    def wiki_search(self, query: str) -> str:
        """Perform a Wikipedia search and retrieve page summaries.
        Good queries are for specific people, places, etc . things that have existing articles in the lexicon, rather than responding to questions."""
        try:
            page_titles = wikipedia.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results)
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return "Error occurred during Wikipedia search."

        summaries = []
        for page_title in page_titles[:self.top_k_results]:
            wiki_page = self._fetch_page(page_title)
            if wiki_page:
                summary = self._formatted_page_summary(page_title, wiki_page)
                if summary:
                    summaries.append(summary)

        if not summaries:
            return "No good Wikipedia Search Result was found"

        return "\n\n".join(summaries)[:self.doc_content_chars_max]

    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    @staticmethod
    def _fetch_page(page_title: str) -> Optional[Any]:
        """Fetch the Wikipedia page."""
        try:
            return wikipedia.page(title=page_title, auto_suggest=False)
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            return None

wiki_wrapper = WikipediaAPIWrapper()
wiki_wrapper.wiki_search("SAP")
agent = ReactAgent([add, multiply, subtract, divide, today, wiki_wrapper.wiki_search])
task = "What is ([Age of barack obama] * 1000) / ([SAP founding year] - [Age of Taylor Swift] - 437)?"

#Let's check the response from the model without any access to tools and updated information
print(send_request(task))

#Now let's see how accurately our agent can respond to this extravagant request
res, scratchpad = agent.run(task)

print_scratchpad(scratchpad)


    print("""
  === TOOLS DOCUMENTATION ===

  Tool 1: convert_currency
    Name:    convert_currency
    Input:   amount (float), from_currency (str, 3-letter e.g. USD), to_currency (str, 3-letter e.g. EUR)
    Output:  {success: bool, amount: float, from: str, to: str, converted: float, rate: float}
    Purpose: Converts a monetary amount between currencies using fixed exchange rates

  Tool 2: convert_units
    Name:    convert_units
    Input:   value (float), from_unit (str), to_unit (str)
    Output:  {success: bool, value: float, from: str, to: str, result: float}
    Purpose: Converts between length (km/miles/meters/feet), weight (kg/lbs/grams),
             and temperature (celsius/fahrenheit/kelvin)

  ReAct Agent: SimpleReActAgent
    Pattern: Thought -> Action -> Observation -> Finish
    Returns: {success, tools_used, results, final_text, trace}
  """)
  
  import re
  import json
  from typing import List, Callable

  # ── Tool definitions ──────────────────────────────────────────────────────────

  def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
      """
      Tool: convert_currency
      Input schema: amount (float), from_currency (str, 3-letter code e.g. USD), to_currency (str, 3-letter code e.g. EUR)
      Purpose: Converts a monetary amount between currencies using fixed exchange rates.
      Output schema: {success: bool, amount: float, from: str, to: str, converted: float, rate: float}
      """
      rates_to_usd = {
          "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067,
          "CHF": 1.13, "CAD": 0.74, "AUD": 0.65, "INR": 0.012
      }
      if not isinstance(amount, (int, float)) or amount < 0:
          return {"success": False, "error": "amount must be a positive number"}
      from_currency = from_currency.upper().strip()
      to_currency = to_currency.upper().strip()
      if len(from_currency) != 3:
          return {"success": False, "error": f"Invalid currency code: {from_currency}"}
      if len(to_currency) != 3:
          return {"success": False, "error": f"Invalid currency code: {to_currency}"}
      if from_currency not in rates_to_usd:
          return {"success": False, "error": f"Currency {from_currency} not supported"}
      if to_currency not in rates_to_usd:
          return {"success": False, "error": f"Currency {to_currency} not supported"}
      amount_in_usd = amount * rates_to_usd[from_currency]
      converted = round(amount_in_usd / rates_to_usd[to_currency], 2)
      rate = round(rates_to_usd[from_currency] / rates_to_usd[to_currency], 4)
      return {"success": True, "amount": amount, "from": from_currency, "to": to_currency, "converted": converted, "rate": rate}


  def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
      """
      Tool: convert_units
      Input schema: value (float), from_unit (str), to_unit (str)
      Purpose: Converts between length (km, miles, meters, feet), weight (kg, lbs, grams) and temperature (celsius, fahrenheit, kelvin).
      Output schema: {success: bool, value: float, from: str, to: str, result: float}
      """
      aliases = {
          "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
          "mile": "miles", "meter": "meters", "metre": "meters", "metres": "meters", "m": "meters",
          "foot": "feet", "kilogram": "kg", "kilograms": "kg",
          "pound": "lbs", "pounds": "lbs", "gram": "grams",
          "c": "celsius", "f": "fahrenheit", "k": "kelvin"
      }
      from_unit = aliases.get(from_unit.lower().strip("."), from_unit.lower().strip("."))
      to_unit   = aliases.get(to_unit.lower().strip("."), to_unit.lower().strip("."))

      length_to_meters = {"meters": 1, "km": 1000, "miles": 1609.34, "feet": 0.3048}
      weight_to_grams  = {"grams": 1, "kg": 1000, "lbs": 453.592}

      if from_unit in length_to_meters and to_unit in length_to_meters:
          result = round(value * length_to_meters[from_unit] / length_to_meters[to_unit], 4)
          return {"success": True, "value": value, "from": from_unit, "to": to_unit, "result": result}

      if from_unit in weight_to_grams and to_unit in weight_to_grams:
          result = round(value * weight_to_grams[from_unit] / weight_to_grams[to_unit], 4)
          return {"success": True, "value": value, "from": from_unit, "to": to_unit, "result": result}

      temp_conversions = {
          ("celsius", "fahrenheit"): lambda v: round(v * 9/5 + 32, 2),
          ("fahrenheit", "celsius"): lambda v: round((v - 32) * 5/9, 2),
          ("celsius", "kelvin"):     lambda v: round(v + 273.15, 2),
          ("kelvin", "celsius"):     lambda v: round(v - 273.15, 2),
          ("fahrenheit", "kelvin"):  lambda v: round((v - 32) * 5/9 + 273.15, 2),
          ("kelvin", "fahrenheit"):  lambda v: round((v - 273.15) * 9/5 + 32, 2),
      }
      fn = temp_conversions.get((from_unit, to_unit))
      if fn:
          return {"success": True, "value": value, "from": from_unit, "to": to_unit, "result": fn(value)}

      return {"success": False, "error": f"Conversion from {from_unit} to {to_unit} is not supported"}


  # ── ReAct-style deterministic agent ───────────────────────────────────────────

  class SimpleReActAgent:
      """
      ReAct-style agent (Reason + Act) using a deterministic regex router.
      Each step produces a Thought -> Action -> Observation trace.
      Returns a structured dict: {success, tools_used, results, final_text, trace}
      """
      def __init__(self, tools: List[Callable]):
          self.tools = {fn.__name__: fn for fn in tools}
          self.tool_names = list(self.tools.keys())

      def _parse_float(self, s: str) -> float:
          return float(s.replace(",", ""))

      def run(self, query: str) -> dict:
          scratchpad = []
          tools_used = []
          results = []
          answers = []

          currency_pattern = re.search(
              r"(\d[\d,]*(?:\.\d+)?)\s*([A-Za-z]{3})\s*(?:to|in)\s*([A-Za-z]{3})"
              r"|([A-Za-z]{3})\s+(\d[\d,]*(?:\.\d+)?)\s*(?:to|in)\s*([A-Za-z]{3})",
              query, re.IGNORECASE
          )
          unit_re = r"(\d[\d,]*(?:\.\d+)?)\s*(km|miles|meters|feet|kg|lbs|pounds|grams|celsius|fahrenheit|kelvin)\s*(?:to|in)\s*(km|miles|meters|feet|kg|lbs|pounds|grams|celsius|fahrenheit|kelvin)"
          unit_pattern = re.search(unit_re, query, re.IGNORECASE)

          if currency_pattern:
              g = currency_pattern.groups()
              if g[0]:
                  amount, from_c, to_c = self._parse_float(g[0]), g[1].upper(), g[2].upper()
              else:
                  from_c, amount, to_c = g[3].upper(), self._parse_float(g[4]), g[5].upper()

              thought = f"Query contains currency codes {from_c} and {to_c} with amount {amount}. Calling convert_currency."
              action_dict = {"name": "convert_currency", "input": {"amount": amount, "from_currency": from_c, "to_currency": to_c}}
              result = self.tools["convert_currency"](amount, from_c, to_c)
              observation = f"{result['amount']} {result['from']} = {result['converted']} {result['to']} (rate: {result['rate']})" if result["success"] else result["error"]

              scratchpad.append(f"Thought: {thought}\nAction:\n```\n{json.dumps(action_dict, indent=2)}\n```\nObservation: {observation}")
              tools_used.append("convert_currency")
              results.append(result)
              answers.append(observation)

          if unit_pattern:
              value     = self._parse_float(unit_pattern.group(1))
              from_unit = unit_pattern.group(2).lower()
              to_unit   = unit_pattern.group(3).lower()

              thought = f"Query contains units {from_unit} and {to_unit} with value {value}. Calling convert_units."
              action_dict = {"name": "convert_units", "input": {"value": value, "from_unit": from_unit, "to_unit": to_unit}}
              result = self.tools["convert_units"](value, from_unit, to_unit)
              observation = f"{result['value']} {result['from']} = {result['result']} {result['to']}" if result["success"] else result["error"]

              scratchpad.append(f"Thought: {thought}\nAction:\n```\n{json.dumps(action_dict, indent=2)}\n```\nObservation: {observation}")
              tools_used.append("convert_units")
              results.append(result)
              answers.append(observation)

          if not answers:
              final_text = "Could not determine which tool to use. Please include values and units or currency codes."
              success = False
          else:
              final_text = " | ".join(answers)
              success = True

          finish_action = json.dumps({"name": "finish", "input": {"answer": final_text}}, indent=2)
          scratchpad.append(f"Thought: I have all results needed to answer.\nAction:\n```\n{finish_action}\n```\nObservation: {final_text}")

          print("\n--- AGENT TRACE ---")
          for i, step in enumerate(scratchpad):
              print(f"\x1b[31m >>> STEP {i+1} <<< \x1b[0m\n{step}\n")

          return {"success": success, "tools_used": tools_used, "results": results, "final_text": final_text, "trace": scratchpad}


  # ── Instantiate ───────────────────────────────────────────────────────────────

  agent = SimpleReActAgent([convert_currency, convert_units])
  print("Agent ready. Tools:", agent.tool_names)


  # ── Test 1: Currency conversion ───────────────────────────────────────────────

  print("\n" + "=" * 60)
  print("TEST 1: Currency Conversion — 500 USD to EUR")
  print("=" * 60)
  r1 = agent.run("Convert 500 USD to EUR")
  assert r1["success"] is True
  assert "convert_currency" in r1["tools_used"]
  assert r1["results"][0]["from"] == "USD"
  assert r1["results"][0]["to"] == "EUR"
  assert isinstance(r1["results"][0]["converted"], float)
  print("FINAL ANSWER:", r1["final_text"])
  print("TEST 1 PASSED")


  # ── Test 2: Unit conversion — distance ───────────────────────────────────────

  print("\n" + "=" * 60)
  print("TEST 2: Unit Conversion — 100 miles to km")
  print("=" * 60)
  r2 = agent.run("Convert 100 miles to km")
  assert r2["success"] is True
  assert "convert_units" in r2["tools_used"]
  assert r2["results"][0]["from"] == "miles"
  assert r2["results"][0]["to"] == "km"
  assert abs(r2["results"][0]["result"] - 160.934) < 0.1
  print("FINAL ANSWER:", r2["final_text"])
  print("TEST 2 PASSED")


  # ── Test 3: Unit conversion — weight ─────────────────────────────────────────

  print("\n" + "=" * 60)
  print("TEST 3: Unit Conversion — 200 lbs to kg")
  print("=" * 60)
  r3 = agent.run("Convert 200 lbs to kg")
  assert r3["success"] is True
  assert abs(r3["results"][0]["result"] - 90.718) < 0.1
  print("FINAL ANSWER:", r3["final_text"])
  print("TEST 3 PASSED")


  # ── Test 4: Unit conversion — temperature ────────────────────────────────────

  print("\n" + "=" * 60)
  print("TEST 4: Unit Conversion — 100 celsius to fahrenheit")
  print("=" * 60)
  r4 = agent.run("Convert 100 celsius to fahrenheit")
  assert r4["success"] is True
  assert r4["results"][0]["result"] == 212.0
  print("FINAL ANSWER:", r4["final_text"])
  print("TEST 4 PASSED")


  # ── Test 5: Error case ────────────────────────────────────────────────────────

  print("\n" + "=" * 60)
  print("TEST 5: Error case — unsupported currency XYZ")
  print("=" * 60)
  r5 = convert_currency(100, "USD", "XYZ")
  assert r5["success"] is False
  assert "error" in r5
  print("Error result:", r5)
  print("TEST 5 PASSED")

  print("\nAll tests passed.")
