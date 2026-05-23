%md
# Lesson 1: Getting Started with Agentic AI using Autogen

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
## 1. Introduction

In this module you will:
- Install and configure the required dependencies: **ag2** (Autogen) and the **SAP Cloud SDK for AI Hub**.
- Set up SAP GenAI credentials by reading from secret scopes.
- Monkey-patch Autogen to use the SAP GenAI models.
- Initialize a basic assistant agent and a user proxy agent.
%md
## 2. Installation and Setup
# %%capture
# Install dependencies (run this cell in a Databricks or similar environment)
%pip install -qqq ag2 "sap-ai-sdk-gen[all]==5.6.3"


# Restart Python after installation:
dbutils.library.restartPython()
%md
## 3. Configure SAP GenAI Hub Credentials
import os
import json
    
# Load your service key from a Secret Store
secret = dbutils.secrets.get(scope="PROD_XGTP_SCOPE", key="LEARNING_GENAIXL")
svcKey = json.loads(secret) 
# Set environment variables so that SAP Cloud SDK can use them for authentication
os.environ["AICORE_AUTH_URL"] = svcKey["url"]
os.environ["AICORE_CLIENT_ID"] = svcKey["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = svcKey["clientsecret"]
os.environ["AICORE_RESOURCE_GROUP"] = "default"
os.environ["AICORE_BASE_URL"] = svcKey["serviceurls"]["AI_API_URL"]
%md
## 4. Monkey-Patch Autogen for SAP GenAI Models
def monkey_patch_ag2():
    """
    Monkey-patch Autogen to use the SAP GenAI Experience Lab models.
    """
    from gen_ai_hub.proxy.native.openai import OpenAI
    import autogen.oai.client as client_
    client_.OpenAI = OpenAI

monkey_patch_ag2()
%md
## 5. Initialize the Autogen Agents
# Define the configuration for your GenAI model (update the API key as needed)
gpt_4o_config = [
    {
        "model": "gpt-4o", #
        "api_key": None, 
        # Do NOT replace the api_key here. Authentication is handled
        # by setup_gen_ai_hub() using Databricks secrets and the SAP AI Core SDK.
    },
]

from autogen import AssistantAgent, UserProxyAgent

# Create the assistant agent and the user proxy agent.
assistant = AssistantAgent("assistant", llm_config={"config_list": gpt_4o_config})
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)


# A simple conversation: ask the agent to print "Hello World!"
result = user_proxy.initiate_chat(assistant, message="Write python code to print Hello World!")
result.chat_history[-2]['content']
%md
*Exercise:*  
Try modifying the initial user message or add another round of conversation for additional responses.
%md
# Lesson 2: Designing and Managing Multi-Step Agentic Workflows

## 1. Introduction

This module demonstrates how to construct a multi-step workflow with Autogen that simulates:
- Breaking down a complex task into multiple sequential steps.
- Checking an intermediate result (a “gate”).
- Refining or branching before arriving at a final validated output.

In this example, we’ll build a chain that summarizes a block of text in multiple stages.

## 2. Multi-Step Workflow Example
# Define helper functions that simulate each step using the user proxy to interact with the assistant agent

def generate_summary(text: str) -> str:
    """
    Step 1: Generate an initial summary of the raw text.
    """
    prompt = f"Summarize this text in one or two sentences:\n\n{text}"
    # Using user_proxy.initiate_chat to prompt the assistant and obtain a response.
    result = user_proxy.initiate_chat(assistant, message=prompt)
    # Extract the final message content from the conversation.
    return result.chat_history[-2]['content']

def check_summary_length(summary: str) -> str:
    """
    Check if the summary is within a desired word count (e.g., 30 words or fewer).
    """
    if len(summary.split()) <= 30:
        return "Pass"
    return "Fail"

def refine_summary(summary: str) -> str:
    """
    If the summary is too long, ask the agent to shorten it.
    """
    prompt = f"Your previous summary was too long. Please shorten it:\n\n{summary}"
    result = user_proxy.initiate_chat(assistant, message=prompt)
    return result.chat_history[-2]['content']

def finalize_summary(summary: str) -> str:
    """
    Finalize the summary for clarity and correctness.
    """
    prompt = f"Review this summary and provide a final, validated version:\n\n{summary}"
    result = user_proxy.initiate_chat(assistant, message=prompt)
    return result.chat_history[-2]['content']

# Example raw text to summarize
raw_text = (
    "SAP offers various enterprise solutions. However, many large organizations have complex data "
    "processes that require thorough summarization before any analytics can be derived. This often "
    "involves multiple steps of refinement and validation."
)

# Execute the multi-step workflow
initial_summary = generate_summary(raw_text)
print("Initial Summary:", initial_summary)

if check_summary_length(initial_summary) == "Fail":
    initial_summary = refine_summary(initial_summary)
    print("\nRefined Summary:", initial_summary)

validated_summary = finalize_summary(initial_summary)
print("\nFinal Validated Summary:", validated_summary)

%md
*Exercise:*  
Expand this workflow by adding another check. For example, add a gate that checks for spelling mistakes and—if found—calls a refinement step before finalizing the summary.
  def check_spelling(text: str) -> str:
      """
      Step: Check if the text contains spelling mistakes using the assistant agent.
      Returns "Pass" if no mistakes found, "Fail" otherwise.
      """
      prompt = (
          f"Check the following text for spelling mistakes. "
          f"Reply with ONLY 'no errors' if there are none, or 'errors found' if there are any:\n\n{text}"
      )
      result = user_proxy.initiate_chat(assistant, message=prompt)
      response = result.chat_history[-2]['content'].lower()
      return "Pass" if "no errors" in response else "Fail"

  def fix_spelling(text: str) -> str:
      """
      If spelling mistakes are found, ask the agent to correct them.
      """
      prompt = f"Fix any spelling mistakes in the following text and return only the corrected version:\n\n{text}"
      result = user_proxy.initiate_chat(assistant, message=prompt)
      return result.chat_history[-2]['content']

  # Execute the multi-step workflow
  initial_summary = generate_summary(raw_text)
  print("Initial Summary:", initial_summary)

  if check_summary_length(initial_summary) == "Fail":
      initial_summary = refine_summary(initial_summary)
      print("\nRefined Summary:", initial_summary)

  # New gate: spelling check before finalization
  if check_spelling(initial_summary) == "Fail":
      initial_summary = fix_spelling(initial_summary)
      print("\nSpelling-corrected Summary:", initial_summary)

  validated_summary = finalize_summary(initial_summary)
  print("\nFinal Validated Summary:", validated_summary)
%md
# Lesson 3: Integrating Autogen with Enterprise-like Systems

## 1. Introduction

In this module, you will learn how to connect your Autogen workflow to external services, such as a public OData endpoint. You can adapt this technique to work with actual enterprise OData or REST endpoints.

## 2. Example: Integrating with a Product Data API

In this example, we define a simple Python function to fetch product data from the public Northwind OData service. Then, we use the assistant agent to produce a user-friendly description of the product.
%md
#### Using the Northwind OData Demo Service

In this tutorial, we will use a public OData service to fetch product data. 
The service is the well-known Northwind dataset, available at the following endpoint:
[https://services.odata.org/V2/Northwind/Northwind.svc/](https://services.odata.org/V2/Northwind/Northwind.svc/)

This is a public, read-only service that does not require authentication, making it ideal for demonstrations.
import requests
import json
from autogen import AssistantAgent, UserProxyAgent  # Adjust import as needed

# -----------------------------------------------------------------------------
# Set up the LLM configuration (using the config from Lesson 1)
# -----------------------------------------------------------------------------
llm_config={"config_list": gpt_4o_config}

# Create the assistant agent with an instructive system message.
assistant = AssistantAgent(
    name="assistant",
    system_message=(
        "For product description tasks, only use the provided function 'fetch_product_data_tool' if "
        "product details are needed. When finished, reply with TERMINATE."
    ),
    llm_config=llm_config,
)

# Create the user proxy agent with automated execution (no human input needed).
user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False},
    max_consecutive_auto_reply=10,
)

# -----------------------------------------------------------------------------
# Northwind OData service configuration
# -----------------------------------------------------------------------------
NORTHWIND_BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"

# -----------------------------------------------------------------------------
# Register the tool: fetch_product_data_tool
# This function is decorated so that:
#  - AssistantAgent (i.e. the LLM) sees its JSON schema and description.
#  - UserProxyAgent maps the tool name to the actual function implementation.
# -----------------------------------------------------------------------------
@user_proxy.register_for_execution()
@assistant.register_for_llm(description="Fetch product data from the Northwind OData service for a given product ID. Returns key fields needed for a product description.")
def fetch_product_data_tool(product_id: int) -> str:
    """
    Calls the Northwind OData service to retrieve the given product's data.
    Returns a summary string containing key details.
    """
    endpoint = f"{NORTHWIND_BASE_URL}/Products({product_id})?$format=json"
    try:
        resp = requests.get(
            endpoint,
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"[OData Fetch Error] Could not retrieve Product {product_id}: {e}"
    
    data = resp.json().get("d", {})
    name = data.get("ProductName", "Unknown Product")
    desc = data.get("QuantityPerUnit", "No Description")
    price = data.get("UnitPrice", "N/A")
    stock = data.get("UnitsInStock", "N/A")
    
    return (
        f"Product {product_id}: {name}\n"
        f"Description: {desc}\n"
        f"Price: {price} USD\n"
        f"Stock: {stock}"
    )

# -----------------------------------------------------------------------------
# Start a conversation in which the assistant autonomously calls the tool.
# The assistant is instructed to generate a product description for product with ID 1.
# With the tool-calling feature, the LLM will determine it needs to call fetch_product_data_tool.
# Once the tool call is executed automatically, the tool’s output is provided to the LLM,
# and it then generates a final product description.
# -----------------------------------------------------------------------------
response = user_proxy.initiate_chat(
    assistant,
    message=(
        "Generate a brief, clear product description suitable for a web shop listing for the product with ID 1. "
        "If product details are needed, call the fetch_product_data_tool function automatically."
    ),
)
# The final response from the assistant is expected to incorporate the product details fetched 
# by the function call and then conclude with "TERMINATE" per the instructions.
print("Final Product Description:\n", response.chat_history[-1]['content'])
%md
### How it works

1. **Tool Registration:**  
   The `fetch_product_data_tool` function is decorated with both  
   - `@assistant.register_for_llm(...)` (which registers the function’s JSON schema and description so the assistant knows about it) and  
   - `@user_proxy.register_for_execution()` (which maps the function name to its implementation so that when the assistant issues a tool call, the user proxy automatically executes it).  
     
2. **Agent Instructions:**  
   The system message for the assistant clarifies that when product details are needed, it should use the registered tool (and then reply with TERMINATE after completing the task).

3. **Autonomous Tool Call:**  
   When the conversation is started with the prompt—asking for a product description for product ID 1—the assistant (powered by a capable LLM with tool-calling features) should detect that it needs to call the `fetch_product_data_tool` function. The UserProxyAgent automatically executes the function call and feeds its output back into the conversation. Finally, the assistant uses the product details to generate the final product description.

4. **No Manual Intervention Required:**  
   Unlike previous versions where you had to manually trigger the function call via a user message, this version makes the process fully automatic—matching the behavior in the Currency Calculator example you referenced.

%md
*Exercise:*  
Enhance this integration by binding a second tool. For instance, add a tool that converts the product price to another currency and then update the final description accordingly.

  import requests
  import json
  from autogen import AssistantAgent, UserProxyAgent
  
  llm_config = {"config_list": gpt_4o_config}

  assistant = AssistantAgent(
      name="assistant",
      system_message=(
          "For product description tasks, use 'fetch_product_data_tool' to get product details "
          "and 'convert_currency_tool' to convert the price to the requested currency. "
          "Always include the converted price in the final description. "
          "When finished, reply with TERMINATE."
      ),
      llm_config=llm_config,
  )

  user_proxy = UserProxyAgent(
      name="user_proxy",
      is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
      human_input_mode="NEVER",
      code_execution_config={"use_docker": False},
      max_consecutive_auto_reply=10,
  )

  NORTHWIND_BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"

  # Hardcoded fallback rates in case the live API is unavailable
  FALLBACK_RATES = {
      "EUR": 0.92,
      "GBP": 0.79,
      "JPY": 149.50,
      "CAD": 1.36,
      "AUD": 1.53,
  }

  @user_proxy.register_for_execution()
  @assistant.register_for_llm(
      description="Fetch product data from the Northwind OData service for a given product ID. "
                  "Returns key fields needed for a product description."
  )
  def fetch_product_data_tool(product_id: int) -> str:
      endpoint = f"{NORTHWIND_BASE_URL}/Products({product_id})?$format=json"
      try:
          resp = requests.get(endpoint, headers={"Accept": "application/json"}, timeout=15)
          resp.raise_for_status()
      except Exception as e:
          return f"[OData Fetch Error] Could not retrieve Product {product_id}: {e}"

      data = resp.json().get("d", {})
      name = data.get("ProductName", "Unknown Product")
      quantity_per_unit = data.get("QuantityPerUnit", "N/A")
      price = data.get("UnitPrice", "N/A")
      stock = data.get("UnitsInStock", "N/A")

      return (
          f"Product {product_id}: {name}\n"
          f"Quantity per unit: {quantity_per_unit}\n"
          f"Price: {price} USD\n"
          f"Stock: {stock}"
      )


  @user_proxy.register_for_execution()
  @assistant.register_for_llm(
      description="Convert a price from USD to a target currency. "
                  "Supported currencies: EUR, GBP, JPY, CAD, AUD. "
                  "Returns the converted amount as a formatted string."
  )
  def convert_currency_tool(amount_usd: float, target_currency: str) -> str:
      target_currency = target_currency.upper().strip()

      try:
          resp = requests.get(
              f"https://api.exchangerate-api.com/v4/latest/USD",
              timeout=10,
          )
          resp.raise_for_status()
          rate = resp.json()["rates"].get(target_currency)
      except Exception:
          rate = FALLBACK_RATES.get(target_currency)

      if rate is None:
          supported = ", ".join(FALLBACK_RATES.keys())
          return f"[Currency Error] Unsupported currency '{target_currency}'. Supported: {supported}"

      converted = round(amount_usd * rate, 2)
      return f"{amount_usd} USD = {converted} {target_currency} (rate: {rate})"

  result = user_proxy.initiate_chat(
      assistant,
      message="Write a product description for product ID 3, and include the price in GBP."
  )
