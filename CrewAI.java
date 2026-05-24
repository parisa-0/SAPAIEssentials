%md
# CrewAI for Business AI: Building Collaborative Agent Crews

This notebook demonstrates how to build collaborative agentic workflows using CrewAI—a framework for orchestrating role-playing, autonomous AI agents. You will learn to:
- Install and set up CrewAI and the SAP Cloud SDK for AI.
- Configure a custom LLM provider to connect CrewAI with SAP GenAI Hub.
- Create custom tools for agents to use.
- Design a crew of agents that work together to complete complex tasks.
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
%md
## Lesson 1: Getting Started with CrewAI

### 1.1 Install the Required Libraries

Run the following cell to install the necessary packages:
# %%capture
%pip install -q "sap-ai-sdk-gen[all]==5.6.3" litellm crewai crewai[tools] 

dbutils.library.restartPython()
%md
### 1.2 Configure SAP GenAI Credentials and Custom LLM

First, we read your SAP GenAI Experience Lab credentials from Databricks secrets. Then, we set up a series of custom classes to make SAP GenAI Hub models compatible with CrewAI through LiteLLM.
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
def init_proxy_client():
    client_id = os.getenv("AICORE_CLIENT_ID")
    client_secret = os.getenv("AICORE_CLIENT_SECRET")
    auth_url = os.getenv("AICORE_AUTH_URL")
    base_url = os.getenv("AICORE_BASE_URL")

    proxy_client = get_proxy_client(
        proxy_version="gen-ai-hub",
        base_url=base_url,
        auth_url=auth_url,
        client_id=client_id,
        client_secret=client_secret,
    )

    return proxy_client

import json
from typing import List
import litellm
from litellm import CustomLLM, ModelResponse
from crewai import Crew, Agent, Task
from crewai import LLM as CrewLLM
from ai_core_sdk.credentials import CREDENTIAL_VALUES, fetch_credentials
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.gen_ai_hub_proxy.client import Deployment
from gen_ai_hub.proxy.native.openai import OpenAI

# --- 1. Credential Setup for Databricks ---
def _get_value(creds, keys):
    for key in keys:
        creds = creds[key]
    if not isinstance(creds, dict):
        return creds
    raise ValueError

def setup_gen_ai_hub(scope='PROD_XGTP_SCOPE', key='BRAND_GENAIXL'):
    try:
        key_data = json.loads(dbutils.secrets.get(scope=scope, key=key))
    except Exception:
        # Fallback for local dev; in Databricks, this should be configured
        print("Could not load Databricks secrets. Ensure they are configured.")
        return
    for value in CREDENTIAL_VALUES:
        if value.name == 'resource_group':
            value.default = 'default'
        if value.vcap_key is None:
            continue
        else:
            try:
                value.default = _get_value(key_data, value.vcap_key[1:])
            except (KeyError, ValueError):
                continue

setup_gen_ai_hub()

# --- 2. Custom LiteLLM Handler for GenAI Hub ---
GENAIHUBMODEL_SUFFIX = "gen_ai_hub_"

def init_llm_model(model: str):
    proxy_client = init_proxy_client()

    llm = init_llm_hub(
        model,
        proxy_client=proxy_client,
        temperature=0,
        max_tokens=15000,
    )

    return llm

class GenAIHubBase:
    def __init__(self):
        self.proxy_client = init_proxy_client()

    def _get_deployments(self) -> List[str]:
        available_models : List[str] = []
        deployment : Deployment
        for deployment in self.proxy_client.deployments:
            available_models.append(deployment.model_name)
        return available_models

    def validate_model_existance(self, model_name : str) -> bool:
        available_models = self._get_deployments()
        if model_name in available_models:
            return True

class GenAIHubCrewLLM(GenAIHubBase):
    def __init__(self, model_name: str = "gpt-4-32k", max_tokens: int = 1024) -> None:
        super().__init__()
        self.validate_model_existance(model_name)
        self._register_custom_litellm_handler()

        model_name = GENAIHUBMODEL_SUFFIX + model_name

        self.llm = CrewLLM(
            model =f"gen-ai-hub-llm/{model_name}",
            max_tokens=max_tokens
        )

    def _register_custom_litellm_handler(self) -> None:
        gen_ai_hub_llm = GenAiHubLiteLLM()
        litellm.custom_provider_map = [
            {"provider": "gen-ai-hub-llm", "custom_handler": gen_ai_hub_llm}
        ]

    def get_llm(self) -> CrewLLM:
        return self.llm    

class GenAiHubLiteLLM(CustomLLM, GenAIHubBase):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls._instance = super(GenAiHubLiteLLM, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        super().__init__()
        GenAIHubBase.__init__(self)
        self.proxy_client = init_proxy_client()
        from gen_ai_hub.proxy.native.openai import OpenAI, embeddings
        self.openai_client = OpenAI(proxy_client=self.proxy_client)

    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        assert kwargs["model"].startswith(GENAIHUBMODEL_SUFFIX)
        model_name = kwargs["model"].replace(GENAIHUBMODEL_SUFFIX, "")

        messages=kwargs["messages"]

        chat_completion = self.openai_client.chat.completions.create(
            model_name=model_name,
            messages=messages,
        )

        return chat_completion   

def monkey_patch_ag2():
    from gen_ai_hub.proxy.native.openai import OpenAI

monkey_patch_ag2()

crewai_llm = GenAIHubCrewLLM('gpt-4o').get_llm()

# Instantiate the LLM for use in the notebook
crewai_llm = GenAIHubCrewLLM('gpt-4o').get_llm()
%md
### 1.3 Run a Simple CrewAI Example
Below is a simple example where a single agent researches a topic.
# Define a simple agent
researcher = Agent(
  role='Expert SAP Researcher',
  goal='Find and summarize the latest information on a given topic.',
  backstory='You are an expert in SAP technologies and have access to a vast amount of information.',
  verbose=True,
  allow_delegation=False,
  llm=crewai_llm
)

# Define a simple task
research_task = Task(
  description='What is SAP Business Technology Platform (BTP) and what are its key pillars?',
  expected_output='A concise, one-paragraph summary explaining SAP BTP and listing its main components.',
  agent=researcher
)

# Form the crew and kick it off
simple_crew = Crew(
  agents=[researcher],
  tasks=[research_task],
  verbose=True
)

result = simple_crew.kickoff()
print("\n--- Simple Crew Final Answer ---")
print(result)
%md
## Lesson 2: Designing and Managing Multi-Agent Workflows with CrewAI

In this lesson, you will create a custom summarization tool and build a two-agent crew to perform a multi-step workflow.

### 2.1 Create a Summarization Tool
Define a tool using `BaseTool` that summarizes a given text into a one-line TLDR:
from crewai.tools import BaseTool

class SummarizationTool(BaseTool):
    name: str = "Summarization Tool"
    description: str = "Summarizes a given block of text into one concise sentence (TLDR)."

    def _run(self, text: str) -> str:
        """Useful to summarize long texts."""
        # In a real scenario, this could call a dedicated summarization model.
        # For this example, we use a simple heuristic.
        words = text.split()
        if len(words) > 15:
            summary = " ".join(words[:15]) + "..."
        else:
            summary = text
        return f"TLDR: {summary}"

# Instantiate the tool for the agent
summarization_tool = SummarizationTool()
%md
### 2.2 Build a Multi-Agent Crew
Now create a crew with a `ContentAnalyst` who provides the text and a `Summarizer` who refines it using the new tool.
# 1. Define Agents
content_analyst = Agent(
    role='Content Analyst',
    goal='Provide detailed text on enterprise data processes.',
    backstory='You are an expert at articulating the complexities of data management in large organizations.',
    verbose=True,
    llm=crewai_llm
)

summary_writer = Agent(
    role='Expert Summarizer',
    goal='Create a concise one-line summary (TLDR) of any text provided.',
    backstory='You are skilled at distilling complex information into a single, impactful sentence.',
    tools=[summarization_tool],
    verbose=True,
    llm=crewai_llm
)

# 2. Define Tasks
raw_text = (
    "SAP offers various enterprise solutions. However, many large organizations have complex data "
    "processes that require thorough summarization before any analytics can be derived. This often involves "
    "multiple steps of refinement and validation."
)

analysis_task = Task(
    description=f"Analyze and present the following text without modification:\n\n{raw_text}",
    expected_output="The full, unmodified text provided in the description.",
    agent=content_analyst,
)

summarization_task = Task(
    description="Using the text from the Content Analyst, create a one-line TLDR summary.",
    expected_output="A single sentence summary beginning with 'TLDR:'.",
    agent=summary_writer,
    context=[analysis_task]  # This task depends on the output of the analysis_task
)

# 3. Form the Crew and Kick it off
workflow_crew = Crew(
    agents=[content_analyst, summary_writer],
    tasks=[analysis_task, summarization_task],
    verbose=True
)

workflow_result = workflow_crew.kickoff()
print("\n--- Workflow Crew Final Answer ---")
print(workflow_result)
%md
## Lesson 3: Integrating CrewAI with Enterprise-like Systems

In this lesson, we wrap the public Northwind OData service as a tool and build a Crew that uses it autonomously to generate a product description.

### 3.1 Create a Fetch Product Data Tool
Define a tool that retrieves product details from the Northwind OData service:
import requests
from crewai.tools import BaseTool

class NorthwindProductTool(BaseTool):
    name: str = "Northwind Product Data Fetcher"
    description: str = "Retrieves product details from the Northwind OData service for a given product ID."

    def _run(self, product_id: str) -> str:
        """Useful for fetching product data. The input must be the integer product ID as a string, for example '1'."""
        BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"
        endpoint = f"{BASE_URL}/Products({product_id})?$format=json"
        
        try:
            resp = requests.get(endpoint, headers={"Accept": "application/json"}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            return f"[OData Fetch Error] Could not retrieve Product {product_id}: {e}"
            
        data = resp.json().get("d", {})
        name = data.get("ProductName", "Unknown Product")
        price = data.get("UnitPrice", "N/A")
        stock = data.get("UnitsInStock", "N/A")
        
        return (
            f"Product ID {product_id}:\n"
            f"- Name: {name}\n"
            f"- Unit Price: {price}\n"
            f"- Units In Stock: {stock}"
        )

northwind_tool = NorthwindProductTool()
%md
### 3.2 Build a Crew That Uses the Tool
Create a `ProductAnalyst` agent equipped with the Northwind tool to generate a product description.
# 1. Define the Agent
product_analyst_agent = Agent(
    role="Product Marketing Analyst",
    goal="Generate compelling product descriptions for a web shop using available data.",
    backstory="You are an expert in e-commerce marketing who can take raw product data and turn it into a customer-friendly description.",
    tools=[northwind_tool],
    verbose=True,
    llm=crewai_llm,
)

# 2. Define the Task
product_description_task = Task(
    description="Generate a brief, clear product description suitable for a web shop listing for the product with ID 2 (Chang). Use your tool to get the product data first.",
    expected_output="A short paragraph describing the product, including its name, price, and stock status.",
    agent=product_analyst_agent,
)

# 3. Form the Crew and Kick it off
product_crew = Crew(
    agents=[product_analyst_agent],
    tasks=[product_description_task],
    verbose=True
)

product_result = product_crew.kickoff()
print("\n--- Product Crew Final Answer ---")
print(product_result)
%md
## How It Works

1. **Lesson 1 – Getting Started:**  
   - **Installation & Setup:** The notebook installs CrewAI, SAP GenAI Hub SDK, and LiteLLM. It then configures credentials from Databricks secrets.
   - **Custom LLM Wrapper:** A custom handler (`GenAiHubLiteLLM`) and wrapper class (`GenAIHubCrewLLM`) are defined to bridge SAP GenAI Hub's API with CrewAI's LLM requirements.
   - **Simple Crew Example:** A basic `Agent` and `Task` are created to perform a simple research query, demonstrating the core components of CrewAI.

2. **Lesson 2 – Designing Workflows:**  
   - **Tool Definition:** A `SummarizationTool` is created by inheriting from `BaseTool`, complete with a name, description, and `_run` method.
   - **Multi-Agent Workflow:** A two-agent `Crew` is assembled. The first agent provides text, and the second agent uses the `SummarizationTool` to process that text. The `context` parameter in the second task ensures they execute sequentially.

3. **Lesson 3 – Enterprise System Integration:**  
   - **Product Data Tool:** A function named `NorthwindProductTool` is registered as a tool, designed to call the public Northwind OData service and return formatted product details.
   - **Autonomous Tool Usage:** A `ProductAnalyst` agent is instructed to generate a product description. When it realizes it needs product data, it autonomously uses the `NorthwindProductTool` to fetch the information before generating its final answer.
%md
## Exercise

**Task:** Enhance your CrewAI setup by adding a new SQL tool that calculates the average product price from a simulated receipts table. Then create a new agent and task so that if a user asks, "What is the average product price?" the crew automatically calls this new tool and returns the calculated value.

### Steps:
1. **Create a New SQL Tool:**  
   - Define a tool class named `SQLAveragePriceTool` that inherits from `BaseTool`.
   - In the `_run` method, connect to a database (e.g., an in-memory SQLite database) and execute a query to return the average price. The method should not require any arguments.

2. **Update the Crew:**  
   - Create a new `DatabaseAnalystAgent` and assign it the `SQLAveragePriceTool`.
   - Create a `CalculateAveragePriceTask` that instructs the agent to find the average product price.
   - Form a new `Crew` with this agent and task.

3. **Test Your Crew:**  
   - `kickoff()` the new crew and verify that it returns the correct average price.

*Hint:* You can use the `sqlite3` or `SQLAlchemy` libraries to create an in-memory database and run your query in the tool's `_run` method.
  import sqlite3
  from crewai.tools import BaseTool                                                                                                             
  from crewai import Agent, Task, Crew

  # --- Step 1: Create the SQL Tool ---

  class SQLAveragePriceTool(BaseTool):
      name: str = "SQL Average Price Tool"
      description: str = (
          "Calculates the average product price from the receipts table "
          "in an in-memory SQLite database. Call this when asked about average product prices."
      )

      def _run(self, **kwargs) -> str:
          conn = sqlite3.connect(":memory:")
          cursor = conn.cursor()

          cursor.execute("""
              CREATE TABLE receipts (
                  id INTEGER PRIMARY KEY,
                  product_name TEXT,
                  unit_price REAL,
                  quantity INTEGER
              )
          """)
          cursor.executemany(
              "INSERT INTO receipts (product_name, unit_price, quantity) VALUES (?, ?, ?)",
              [
                  ("Chai",           18.00, 10),
                  ("Chang",          19.00,  5),
                  ("Aniseed Syrup",  10.00, 13),
                  ("Chef Anton Mix", 22.00,  3),
                  ("Grandma Boysen", 25.00,  7),
              ]
          )
          conn.commit()

          cursor.execute("SELECT AVG(unit_price) FROM receipts")
          avg_price = cursor.fetchone()[0]
          conn.close()

          return f"The average product price is ${avg_price:.2f}"

  sql_avg_price_tool = SQLAveragePriceTool()


  # --- Step 2: Create the Agent, Task, and Crew ---

  database_analyst_agent = Agent(
      role="Database Analyst",
      goal="Answer questions about product pricing using SQL queries.",
      backstory="You are a data analyst who retrieves and interprets pricing data from databases.",
      tools=[sql_avg_price_tool],
      verbose=True,
      llm=crewai_llm,
  )

  calculate_average_price_task = Task(
      description="What is the average product price? Use your SQL tool to calculate it.",
      expected_output="A sentence stating the average product price as a dollar amount.",
      agent=database_analyst_agent,
  )

  pricing_crew = Crew(
      agents=[database_analyst_agent],
      tasks=[calculate_average_price_task],
      verbose=True,
  )


  # --- Step 3: Kick off the Crew ---

  result = pricing_crew.kickoff()
  print("\n--- Pricing Crew Final Answer ---")
  print(result)
