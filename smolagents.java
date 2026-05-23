%md

# Smolagents for Business AI: Building Agentic Workflows

This notebook demonstrates how to build agentic workflows using smolagents—a lightweight, code-first framework from Hugging Face. You will learn to:
- Install and set up smolagents and the SAP Cloud SDK for AI.
- Configure your SQL environment.
- Register tools as Python functions.
- Create a CodeAgent that dynamically generates and executes Python code to answer natural language business queries.

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
## Lesson 1: Getting Started with Smolagents

### 1.1 Install the Required Libraries

Run the following cell to install the necessary packages:
# %%capture
%pip install -qqq smolagents
%pip install -qqq "sap-ai-sdk-gen[all]==5.6.3"

dbutils.library.restartPython()
%md

### 1.2 Configure SAP GenAI Credentials

Ensure you have your SAP GenAI Experience Lab credentials configured. The following code reads your credentials from Databricks secrets:
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
### 1.3 Run a Simple CodeAgent Example
Below is a simple example where the agent prints a "Hello World" message.
from smolagents import CodeAgent, OpenAIServerModel
from gen_ai_hub.proxy.native.openai import OpenAI

# Create an OpenAIServerModel instance (update your API key accordingly)
model = OpenAIServerModel('gpt-4o', api_key='YOUR_API_KEY_HERE')
model.client = OpenAI()  # Patch the client to use SAP GenAI Experience Lab models

# Instantiate a simple CodeAgent without tools.
agent = CodeAgent(
    tools=[],
    model=model,
)

# Run the agent with a simple prompt.
result = agent.run("Print 'Hello World!'")
print("Agent output:", result)
%md
## Lesson 2: Designing and Managing Multi-Step Agentic Workflows with Smolagents

In this lesson, you will create a reusable summarization function that the agent calls as part of a multi-step workflow.

### 2.1 Create a Summarization Tool
Define a tool function that summarizes a given text into a one-line TLDR:
from smolagents import tool

@tool
def summarize_text(text: str) -> str:
    """
    Summarize the input text into one concise sentence (TLDR).
    
    Args:
        text: The text to be summarized.
        
    Returns:
        A one-line summary.
    """
    # For the purpose of this example, use a simple heuristic or static output.
    # In practice, you could write code for a complex summarization process.
    words = text.split()
    summary = " ".join(words[:10]) + "..." if len(words) > 10 else text
    return summary
%md
### 2.2 Build a Multi-Step Workflow Agent
Now create an agent that first summarizes a block of text and then refines it.
# Reuse the same model from Lesson 1.
agent = CodeAgent(
    tools=[summarize_text],
    model=model,
)
# Example raw text.
raw_text = (
    "SAP offers various enterprise solutions. However, many large organizations have complex data "
    "processes that require thorough summarization before any analytics can be derived. This often involves "
    "multiple steps of refinement and validation."
)

# In this multi-step workflow, we ask the agent to:
# 1) Summarize the text using the 'summarize_text' tool.
# 2) Then generate a refined summary.
workflow_query = (
    "Using the provided text, first call the summarize_text function to get an initial summary. "
    "Then, based on the output, generate a refined one-line TLDR."
)
result = agent.run(workflow_query + "\n\nText:\n" + raw_text)
print("Refined Summary:", result)
%md
## Lesson 3: Integrating Smolagents with Enterprise-like Systems

In this lesson, we wrap a public OData service (Northwind) as a tool and build a CodeAgent that uses it autonomously to generate a product description.

### 3.1 Create a Fetch Product Data Tool
Define a tool function that retrieves product details from the Northwind OData service:
from smolagents import tool

@tool
def fetch_product_data(product_id: int) -> str:
    """
    Retrieves product details from the Northwind OData service for a given product ID.
    Returns key product fields as a formatted string.
    
    Args:
        product_id: The integer product ID to look up (e.g., 1).
        
    Returns:
        A string with product name, unit price, and units in stock.
    """
    import requests
    
    BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"
    endpoint = f"{BASE_URL}/Products({product_id})?$format=json"
    
    try:
        resp = requests.get(
            endpoint,
            headers={"Accept": "application/json"},
            timeout=15
        )
        resp.raise_for_status()
    except Exception as e:
        return f"[OData Fetch Error] Could not retrieve Product {product_id}: {e}"
        
    data = resp.json().get("d", {})
    name = data.get("ProductName", "Unknown Product")
    price = data.get("UnitPrice", "N/A")
    stock = data.get("UnitsInStock", "N/A")
    
    return (
        f"Product {product_id}: {name}\n"
        f"Unit Price: {price}\n"
        f"Units In Stock: {stock}"
    )
%md
### 3.2 Build a CodeAgent That Uses the Tool

Create an agent that instructs the LLM to generate a product description using the tool:
# Instantiate a new CodeAgent with the fetch_product_data tool.
agent = CodeAgent(
    tools=[fetch_product_data],
    model=model,
)

# Query the agent to generate a product description for product with ID 1.
query = (
    "Generate a brief, clear product description suitable for a web shop listing for the product with ID 1. "
    "If you need more details, call the fetch_product_data function automatically."
)
result = agent.run(query)
print("Final Product Description:\n", result)
%md
## How It Works

1. **Lesson 1 – Getting Started:**  
   - **Installation & Setup:** The notebook installs smolagents and the SAP GenAI Hub SDK, then configures credentials from Databricks secrets.  
   - **Simple Agent Example:** A basic CodeAgent is created and queried to perform a simple task, such as printing "Hello World!".

2. **Lesson 2 – Designing Workflows:**  
   - **Tool Definition:** A summarization tool is created with the `@tool` decorator, including complete type hints and a docstring.  
   - **Multi-Step Workflow:** A CodeAgent uses the tool function to first produce an initial summary and then refine that summary—simulating a ReAct-style multi-turn process.

3. **Lesson 3 – Enterprise System Integration:**  
   - **Product Data Tool:** A function named `fetch_product_data` is registered as a tool, designed to call the public Northwind OData service and return formatted product details.  
   - **Autonomous Tool Usage:** The CodeAgent is instructed to generate a product description for a product with a specific ID (e.g., "1"). When additional product details are needed, the agent automatically calls `fetch_product_data` and then generates a final description that includes the fetched information.
%md
## Exercise

**Task:** Enhance the Smolagents CodeAgent by adding a new SQL tool that calculates the average product price from a simulated receipts table. Then update the agent’s prompt so that if a user asks, "What is the average product price?" the agent automatically calls this new tool and returns the calculated value.

### Steps:
1. **Create a New SQL Tool:**  
   - Define a tool function named `calculate_avg_price` that queries your database (or a simulated dataset) to return the average price.
   - Use appropriate type hints and a descriptive docstring with an **Args:** section.

2. **Update the CodeAgent:**  
   - Add the new tool to the CodeAgent’s tools list.
   - Modify the agent’s instructions to indicate that, when asked about average prices, it should call the `calculate_avg_price` function.

3. **Test Your Agent:**  
   - Run the agent with a query like "What is the average product price?" and verify that it returns the correct value.

*Hint:* If using a SQLite in-memory database, you can use SQLAlchemy and the AVG() function in your tool implementation.

  from smolagents import tool, CodeAgent, OpenAIServerModel
  from gen_ai_hub.proxy.native.openai import OpenAI
  import sqlalchemy as sa
  
  @tool
  def calculate_avg_price() -> float:
      """
      Calculates the average product price from the receipts table.
   
      Returns:
          The average product price as a float, rounded to 2 decimal places.
      """
      engine = sa.create_engine("sqlite:///:memory:")

      with engine.connect() as conn:
          # Create and populate a simulated receipts table
          conn.execute(sa.text("""
              CREATE TABLE receipts (
                  id INTEGER PRIMARY KEY,
                  product_name TEXT,
                  unit_price REAL,
                  quantity INTEGER
              )
          """))
          conn.execute(sa.text("""
              INSERT INTO receipts (product_name, unit_price, quantity) VALUES
                  ('Chai',            18.00, 10),
                  ('Chang',           19.00,  5),
                  ('Aniseed Syrup',   10.00, 13),
                  ('Chef Anton Mix',  22.00,  3),
                  ('Grandma Boysen',  25.00,  7)
          """))
          conn.commit()

          result = conn.execute(sa.text("SELECT AVG(unit_price) FROM receipts"))
          avg_price = result.scalar()

      return round(avg_price, 2)
  
   # --- Step 2: Update the CodeAgent ---

  model = OpenAIServerModel('gpt-4o', api_key='YOUR_API_KEY_HERE')
  model.client = OpenAI()

  agent = CodeAgent(
      tools=[fetch_product_data, summarize_text, calculate_avg_price],
      model=model,
  )


  # --- Step 3: Test the Agent ---

  result = agent.run("What is the average product price?")
  print("Agent output:", result)

     verbosity_level=2,
                                                                                                                    │
│ What is the average product price?                                                                              │
│                                                                                                                 │
╰─ OpenAIModel - gpt-4o ──────────────────────────────────────────────────────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output message of the LLM: ────────────────────────────────────────────────────────────────────────────────────────
Thought: To solve this task, I will directly use the tool `calculate_avg_price`, which is specifically designed to 
calculate the average product price from the receipts table. Let me call this tool to find the answer.             
                                                                                                                   
<code>                                                                                                             
average_price = calculate_avg_price()                                                                              
final_answer(average_price)                                                                                        
                                                                                                                   
 ─ Executing parsed code: ──────────────────────────────────────────────────────────────────────────────────────── 
  average_price = calculate_avg_price()                                                                            
  final_answer(average_price)                                                                                      
 ───────────────────────────────────────────────
