how did the LLM know which tool to use                                                                                                                                            

● Great question. When you called llm.bind_tools([fetch_country_data, convert_currency]), LangChain automatically converted your tool definitions into a description that gets sent
  to the LLM as part of the prompt. It looks something like this:

  You have access to the following tools:

  fetch_country_data(country_name: str)
    - Uses restcountries.com to look up capital and population of a country.

  convert_currency(amount: float, from_currency: str, to_currency: str)
    - Converts an amount between currencies using live exchange rates from open.er-api.com.

  The LLM then reads the user query:

  ▎ "What is the population of Germany and how much is 100 USD in EUR?"

  And matches it against those descriptions using its language understanding:

  - "population of Germany" → sounds like fetch_country_data
  - "100 USD in EUR" → sounds like convert_currency

  The key is the docstring. That one-line description on each tool is literally what the LLM reads to decide. Try changing it to something misleading and the LLM will call the
  wrong tool.

  This is also why in your previous assignment ToolInfo.from_callable extracted the docstring — same idea, just done manually instead of by LangChain.

  So the "intelligence" here isn't magic — it's the LLM doing text matching between the query and the tool descriptions you wrote.

  %md
## Lesson 1: Getting Started with LangGraph for Business

### 1. Introduction

In this first module, we’ll:
1. Install and configure **SAP Cloud AI SDK** and **LangGraph** within a Databricks (or similar) environment.
2. Initialize an **LLM** via the SAP Generative AI Hub.
3. Bind a **real tool** (external API function) that the LLM can call autonomously.
4. Show how the agent decides whether to use that tool in response to user prompts.

### 2. Key Concepts

- **LLM Augmentation**: We wrap the base LLM from SAP Generative AI Hub, providing structured outputs or tool-calling ability.
- **Tools**: Python functions that can perform tasks an LLM can’t do in-text alone.  
- **Agent/Tool Workflow**: The LLM chooses whether and how to call the tool, then uses the tool’s output to craft a final answer.

  
%md
## Lesson 1: Getting Started with LangGraph for Business

### 1. Introduction

In this first module, we’ll:
1. Install and configure **SAP Cloud AI SDK** and **LangGraph** within a Databricks (or similar) environment.
2. Initialize an **LLM** via the SAP Generative AI Hub.
3. Bind a **real tool** (external API function) that the LLM can call autonomously.
4. Show how the agent decides whether to use that tool in response to user prompts.

### 2. Key Concepts

- **LLM Augmentation**: We wrap the base LLM from SAP Generative AI Hub, providing structured outputs or tool-calling ability.
- **Tools**: Python functions that can perform tasks an LLM can’t do in-text alone.  
- **Agent/Tool Workflow**: The LLM chooses whether and how to call the tool, then uses the tool’s output to craft a final answer.
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
%%capture

%pip install -q "sap-ai-sdk-gen[all]==5.6.3" 
%pip install langchain_openai langchain langgraph databricks-langchain
dbutils.library.restartPython()

import json
import os
import requests

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

# ---------------------------------------------------------------------------
#  STEP 1:  Setup - reading credentials from AI Core secrets (example)
# ---------------------------------------------------------------------------

# Load your service key from a Secret Store
secret = dbutils.secrets.get(scope="PROD_XGTP_SCOPE", key="LEARNING_GENAIXL")
svcKey = json.loads(secret) 
# Set environment variables so that SAP Cloud SDK can use them for authentication
os.environ["AICORE_AUTH_URL"] = svcKey["url"]
os.environ["AICORE_CLIENT_ID"] = svcKey["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = svcKey["clientsecret"]
os.environ["AICORE_RESOURCE_GROUP"] = "default"
os.environ["AICORE_BASE_URL"] = svcKey["serviceurls"]["AI_API_URL"]

# ---------------------------------------------------------------------------
#  STEP 2:  Initialize the LLM from SAP Gen AI Hub
# ---------------------------------------------------------------------------
from gen_ai_hub.proxy.langchain import init_llm
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

# For example, if you have a model named 'gpt-4o', you can do:
llm = ChatOpenAI(proxy_model_name='gpt-4o')
# ---------------------------------------------------------------------------
#  STEP 3:  Define and bind a real tool
# ---------------------------------------------------------------------------
from langchain.tools import tool
class CountryName(BaseModel):
    country_name: str = Field(description="should be the country name")

@tool("fetch_country_data", args_schema=CountryName, return_direct=True)
def fetch_country_data(country_name: str) -> str:
    """
    Uses a real external API (restcountries.com) to look up capital & population.
    """
    try:
        resp = requests.get(f"https://restcountries.com/v3.1/name/{country_name}", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"[Tool Error] Could not fetch data for '{country_name}': {e}"

    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        capital = data[0].get("capital", ["Unknown"])[0]
        population = data[0].get("population", "Unknown")
        return f"Capital of {country_name}: {capital}, Population: {population}"
    return f"No valid data found for {country_name}"

# Bind the tool to the LLM
llm_with_tools = llm.bind_tools([fetch_country_data])

# ---------------------------------------------------------------------------
#  STEP 4:  Invoke the LLM and see if it calls the tool
# ---------------------------------------------------------------------------
user_prompt = (
    "I want to know the capital city and approximate population of France. "
    "Call the 'fetch_country_data' tool if needed."
)

# The agent LLM may or may not decide to call the tool. Let's see:
msg = llm_with_tools.invoke(user_prompt)

# Log the tool calls
print("Tool calls observed:")
for call in msg.tool_calls or []:
    print(call)
print(fetch_country_data.description)
print(fetch_country_data.name)
print(fetch_country_data.return_direct)
%md
### How it works

1. setup_gen_ai_hub()
Populates environment variables and credentials so that init_llm(...) can authenticate with SAP Generative AI Hub.

2. init_llm('gpt-4o')
Creates an LLM object referencing a model name (“gpt-4o” as an example). Adjust to match an available model in your system.

3. fetch_country_data(country_name)
A real “tool” function that calls restcountries.com to return capital and population data.

4. llm.bind_tools([fetch_country_data])
Tells the LLM that it can call fetch_country_data(...) if it decides to do so.

5. llm_with_tools.invoke(...)
We pass a user prompt that explicitly hints the agent to use the tool. The LLM’s final message is in msg.content. The tool call details (if any) are in msg.tool_calls.
%md
Observe that the approach above does not actually calls the tools. We can instead build a minimal ReAct agent using Langgraph as shown below.
print(msg.content)
# This code depends on pip install langchain[anthropic]
from langgraph.prebuilt import create_react_agent


agent = create_react_agent(llm, tools=[fetch_country_data])
agent.invoke(
    {"messages": [{"role": "user", "content": "I want to know the capital city and approximate population of France."}]}
)

%md
### 4. Exercise
1. Add a Second Tool

 - Create another function that calls an open weather or currency exchange API.

 - Bind it as llm.bind_tools([fetch_country_data, second_tool]).

2. Composite Query

- Prompt: “What is the population of Germany and what is the weather in Berlin? Please use the tools as needed.”

- Check if it calls both tools, or see how it sequences calls.

3. Logging

- Print the tool calls to observe the LLM’s chain-of-thought process.
  import json
  import os
  import requests

  from langchain.pydantic_v1 import BaseModel, Field
  from langchain.tools import tool
  from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

  # ── Credentials ───────────────────────────────────────────────────────────────

  secret = dbutils.secrets.get(scope="PROD_XGTP_SCOPE", key="LEARNING_GENAIXL")
  svcKey = json.loads(secret)
  os.environ["AICORE_AUTH_URL"]          = svcKey["url"]
  os.environ["AICORE_CLIENT_ID"]         = svcKey["clientid"]
  os.environ["AICORE_CLIENT_SECRET"]     = svcKey["clientsecret"]
  os.environ["AICORE_RESOURCE_GROUP"]    = "default"
  os.environ["AICORE_BASE_URL"]          = svcKey["serviceurls"]["AI_API_URL"]

  # ── LLM ───────────────────────────────────────────────────────────────────────

  llm = ChatOpenAI(proxy_model_name='gpt-4o')

  # ── Tool 1: Country data ──────────────────────────────────────────────────────

  class CountryName(BaseModel):
      country_name: str = Field(description="should be the country name")

  @tool("fetch_country_data", args_schema=CountryName, return_direct=False)
  def fetch_country_data(country_name: str) -> str:
      """Uses restcountries.com to look up capital and population of a country."""
      try:
          resp = requests.get(f"https://restcountries.com/v3.1/name/{country_name}", timeout=10)
          resp.raise_for_status()
          data = resp.json()
          if isinstance(data, list) and len(data) > 0:
              capital    = data[0].get("capital", ["Unknown"])[0]
              population = data[0].get("population", "Unknown")
              return f"Capital of {country_name}: {capital}, Population: {population}"
          return f"No valid data found for {country_name}"
      except Exception as e:
          return f"[Tool Error] Could not fetch data for '{country_name}': {e}"

  # ── Tool 2: Currency exchange ─────────────────────────────────────────────────
  # Uses open.er-api.com — no API key required

  class CurrencyConversion(BaseModel):
      amount:        float = Field(description="amount to convert")
      from_currency: str   = Field(description="source currency code e.g. USD")
      to_currency:   str   = Field(description="target currency code e.g. EUR")

  @tool("convert_currency", args_schema=CurrencyConversion, return_direct=False)
  def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
      """Converts an amount between currencies using live exchange rates from open.er-api.com."""
      try:
          resp = requests.get(f"https://open.er-api.com/v6/latest/{from_currency.upper()}", timeout=10)
          resp.raise_for_status()
          data = resp.json()
          if data.get("result") != "success":
              return f"Could not fetch rates for {from_currency}"
          rate      = data["rates"].get(to_currency.upper())
          if rate is None:
              return f"Currency {to_currency} not found"
          converted = round(amount * rate, 2)
          return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()} (rate: {rate})"
      except Exception as e:
          return f"[Tool Error] {e}"

  # ── Bind both tools to the LLM ────────────────────────────────────────────────
  # The LLM now knows about both tools and can decide which to call

  llm_with_tools = llm.bind_tools([fetch_country_data, convert_currency])

  # ── Composite query ───────────────────────────────────────────────────────────
  # This is the key test — one prompt that requires BOTH tools

  user_prompt = (
      "What is the population of Germany and how much is 100 USD in EUR? "
      "Please use the tools as needed."
  )

  msg = llm_with_tools.invoke(user_prompt)

  # ── Log tool calls ────────────────────────────────────────────────────────────

  print("=" * 60)
  print("USER PROMPT:", user_prompt)
  print("=" * 60)
  print("\nTOOL CALLS OBSERVED:")
  if msg.tool_calls:
      for i, call in enumerate(msg.tool_calls):
          print(f"\n  Call {i+1}:")
          print(f"    Tool:  {call['name']}")
          print(f"    Input: {call['args']}")
  else:
      print("  No tool calls made — LLM answered from memory")

  print("\nLLM RESPONSE CONTENT:")
  print(msg.content)

%md
## Lesson 2: Designing Multi-Step Agentic Workflows

### 1. Introduction

Welcome to **Module 2** of our **LangGraph for Business AI** course! Building on the basics from Module 1, we’ll explore **multi-step workflows** that allow Large Language Models (LLMs) to:
- Decompose complex tasks into smaller, more manageable steps.
- Conditionally branch based on intermediate outputs (i.e., “gate” logic).
- Merge or finalize outputs to deliver robust results.

This approach is particularly relevant to **enterprise scenarios**, where tasks often require multiple stages (e.g., validating data, summarizing, and then making a decision). By using **LangGraph** plus the **SAP Generative AI Hub**, you can orchestrate these steps in a structured, maintainable way.

---

### 2. Key Concepts

1. **Prompt Chaining**  
   Breaking a task into multiple LLM calls, each processing the output of the previous one. You can gate or check intermediate results before moving on.

2. **Graph-Based Workflows**  
   - **Nodes**: Encapsulate specific logic or an LLM call.  
   - **Edges**: Link nodes in sequence or through conditional branches.  
   - **State**: A shared data structure capturing input, intermediate results, and final outputs.

3. **Conditional Gating**  
   Before proceeding to the next step, use programmatic checks to confirm correctness or completeness. If the check fails, reroute or fix the data.

4. **Enterprise Use Case**  
   Example: Summarizing user input, validating correctness, then generating a final refined output. In SAP contexts, such gating might ensure compliance with business rules before final actions (e.g., updating records).
%md
### 3. Hands-On Code Example

Below is a **multi-step** workflow demonstration using **LangGraph** and an **LLM** from **SAP Generative AI Hub**. It expands on the single-call approach from Module 1 by introducing:

- A shared **State** typed dictionary.
- **Multiple nodes** for generating and refining text.
- A **gate function** to conditionally direct flow.

> **Prerequisite**:  
> - You have already installed `langgraph` and `sap-ai-sdk-gen[all]==5.6.3`.  
> - You ran the environment setup from Module 1 (credentials and `init_llm`).
import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# We'll reuse the same LLM reference from Module 1:
#   from gen_ai_hub.proxy.langchain import init_llm
#   llm = init_llm('gpt-4o', max_tokens=512)
# For brevity, we assume `llm` is already defined.

# ---------------------------------------------------------------------------
# STEP 1: Define our shared state for the workflow
# ---------------------------------------------------------------------------
class SummarizeState(TypedDict):
    raw_text: str
    summary: str
    validated_summary: str

# ---------------------------------------------------------------------------
# STEP 2: Define the node functions
# ---------------------------------------------------------------------------
def generate_summary(state: SummarizeState):
    """
    First node: create an initial summary of raw_text using the LLM.
    """
    response = llm.invoke(f"Summarize this text in one or two sentences:\n{state['raw_text']}")
    return {"summary": response.content}

def check_summary_length(state: SummarizeState):
    """
    Gate function to decide if the summary is short enough.
    Return "Pass" if short enough, else "Fail".
    """
    if len(state["summary"].split()) <= 30:
        return "Pass"
    return "Fail"

def refine_summary(state: SummarizeState):
    """
    If the summary is too long, let's refine or compress it further.
    """
    response = llm.invoke(f"Your previous summary was too long. Please shorten it further:\n{state['summary']}")
    return {"summary": response.content}

def finalize_summary(state: SummarizeState):
    """
    Second LLM call to finalize or 'validate' the summary 
    (you could do style checks, compliance checks, etc.).
    """
    response = llm.invoke(
        f"Review this summary for clarity and correctness. Provide the final validated version:\n{state['summary']}"
    )
    return {"validated_summary": response.content}

# ---------------------------------------------------------------------------
# STEP 3: Build and compile the workflow graph
# ---------------------------------------------------------------------------
workflow = StateGraph(SummarizeState)

# Add nodes
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("refine_summary", refine_summary)
workflow.add_node("finalize_summary", finalize_summary)

# Connect them via edges
workflow.add_edge(START, "generate_summary")

# A conditional edge for "check_summary_length"
def check_length(state: SummarizeState):
    # We'll just reuse the gate function here
    return check_summary_length(state)

workflow.add_conditional_edges(
    "generate_summary", check_length,
    {"Pass": "finalize_summary", "Fail": "refine_summary"}
)

# After refining, proceed to finalize
workflow.add_edge("refine_summary", "finalize_summary")
workflow.add_edge("finalize_summary", END)

chain = workflow.compile()

# ---------------------------------------------------------------------------
# STEP 4: Run the multi-step chain
# ---------------------------------------------------------------------------
example_text = """SAP offers various enterprise solutions. However, many large organizations
have complex data processes that require thorough summarization before any analytics 
can be derived. This often involves multiple steps of refinement and validation."""

initial_state = {"raw_text": example_text, "summary": "", "validated_summary": ""}
final_state = chain.invoke(initial_state)

print("Raw text:")
print(example_text)
print("\n--- Final Workflow Output ---")
print("Initial Summary:", final_state.get("summary"))
print("Validated Summary:", final_state.get("validated_summary"))




from IPython.display import Image, display
# Show workflow
try:    display(Image(chain.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Error displaying graph: {e}")
%md
**How it Works**  
1. **generate_summary**: The LLM produces a short summary.  
2. We **check_summary_length**.  
   - If “Pass,” the flow goes straight to **finalize_summary**.  
   - If “Fail,” it calls **refine_summary** first, then proceeds to finalize.  
3. **finalize_summary**: A final LLM pass ensuring the summary is “approved.”  
4. The **workflow** returns the updated `validated_summary` to your code.
%md
### 4. Exercise

1. **Add a Second Gate**  
   - Expand the pipeline to include a second gate that checks if the final summary is within a certain word limit. If it exceeds, go back to **refine_summary**.

2. **Add a Custom Tool**  
   - Suppose you want to run a spell-check or domain-check. Implement a Python function (tool) and call it from a node or gate.  
   - For instance, “Spell-check the summary and if there are errors, route to a refine node.”

3. **Observe Multi-Path Flow**  
   - Provide text that definitely triggers the “Fail” path.  
   - Confirm the workflow transitions to **refine_summary** before reaching the final answer.
  import json
  from typing_extensions import TypedDict
  from langgraph.graph import StateGraph, START, END
  
  # ── State ─────────────────────────────────────────────────────────────────────
  # All fields flow through every node — nodes read what they need and write back

  class SummarizeState(TypedDict):
      raw_text:          str
      summary:           str
      validated_summary: str
      spell_check_notes: str  # new: stores spell check findings

  # ── Custom tool: spell/domain checker ────────────────────────────────────────
  # This is a plain Python function called from inside a node.
  # In a real scenario this could call a spell-check API or an LLM.
  # Here we check for a list of known domain-relevant terms that should appear.

  def spell_and_domain_check(text: str) -> str:
      """
      Checks for common misspellings and whether key SAP domain terms are present.
      Returns a string describing any issues found, or 'OK' if clean.
      """
      issues = []

      # simple misspelling check
      misspellings = {
          "teh": "the",
          "sumary": "summary",
          "analitics": "analytics",
          "organzation": "organization",
          "solutons": "solutions",
      }
      for wrong, correct in misspellings.items():
          if wrong in text.lower():
              issues.append(f"Misspelling: '{wrong}' should be '{correct}'")

      # domain check — summary should mention SAP or enterprise
      domain_terms = ["sap", "enterprise", "data", "analytics", "solution"]
      found = [t for t in domain_terms if t in text.lower()]
      if not found:
          issues.append("Domain check: summary does not mention any SAP/enterprise terms")

      return "OK" if not issues else " | ".join(issues)


  # ── Node functions ────────────────────────────────────────────────────────────

  def generate_summary(state: SummarizeState):
      """Node 1: ask the LLM for an initial summary."""
      response = llm.invoke(
          f"Summarize this text in one or two sentences:\n{state['raw_text']}"
      )
      print(f"[generate_summary] produced {len(response.content.split())} words")
      return {"summary": response.content}


  def refine_summary(state: SummarizeState):
      """Node 2: ask the LLM to shorten or fix the summary."""
      notes = state.get("spell_check_notes", "")
      prompt = f"Your previous summary needs improvement.\n"
      if notes and notes != "OK":
          prompt += f"Issues found: {notes}\n"
      prompt += f"Please shorten and fix it (max 25 words):\n{state['summary']}"
      response = llm.invoke(prompt)
      print(f"[refine_summary] refined to {len(response.content.split())} words")
      return {"summary": response.content, "spell_check_notes": ""}


  def run_spell_check(state: SummarizeState):
      """Node 3: run the spell/domain check tool and store the result in state."""
      notes = spell_and_domain_check(state["summary"])
      print(f"[run_spell_check] result: {notes}")
      return {"spell_check_notes": notes}


  def finalize_summary(state: SummarizeState):
      """Node 4: final LLM call to validate and polish."""
      response = llm.invoke(
          f"Review this summary for clarity and correctness. "
          f"Provide the final validated version:\n{state['summary']}"
      )
      print(f"[finalize_summary] final word count: {len(response.content.split())}")
      return {"validated_summary": response.content}


  # ── Gate functions ────────────────────────────────────────────────────────────

  def check_initial_length(state: SummarizeState):
      """Gate 1: after generate_summary — is it under 30 words?"""
      word_count = len(state["summary"].split())
      result = "Pass" if word_count <= 30 else "Fail"
      print(f"[gate 1 - initial length] {word_count} words → {result}")
      return result


  def check_spell_result(state: SummarizeState):
      """Gate 2: after spell check — are there any issues?"""
      result = "Pass" if state.get("spell_check_notes", "") == "OK" else "Fail"
      print(f"[gate 2 - spell check] → {result}")
      return result


  def check_final_length(state: SummarizeState):
      """Gate 3: after finalize — is validated summary under 35 words?"""
      word_count = len(state["validated_summary"].split())
      result = "Pass" if word_count <= 35 else "Fail"
      print(f"[gate 3 - final length] {word_count} words → {result}")
      return result


  # ── Build the graph ───────────────────────────────────────────────────────────

  workflow = StateGraph(SummarizeState)

  workflow.add_node("generate_summary", generate_summary)
  workflow.add_node("refine_summary",   refine_summary)
  workflow.add_node("run_spell_check",  run_spell_check)
  workflow.add_node("finalize_summary", finalize_summary)

  # START → generate
  workflow.add_edge(START, "generate_summary")

  # Gate 1: initial length check
  workflow.add_conditional_edges(
      "generate_summary", check_initial_length,
      {"Pass": "run_spell_check", "Fail": "refine_summary"}
  )

  # After refine → spell check
  workflow.add_edge("refine_summary", "run_spell_check")

  # Gate 2: spell check result
  workflow.add_conditional_edges(
      "run_spell_check", check_spell_result,
      {"Pass": "finalize_summary", "Fail": "refine_summary"}
  )

  # Gate 3: final length check — if too long, go back to refine
  workflow.add_conditional_edges(
      "finalize_summary", check_final_length,
      {"Pass": END, "Fail": "refine_summary"}
  )

  chain = workflow.compile()


  # ── Test 1: long text that triggers the Fail path ────────────────────────────
  # This text is deliberately verbose to force refine_summary to be visited

  print("=" * 60)
  print("TEST 1: Long text — expect Fail path on gate 1")
  print("=" * 60)

  long_text = """
  SAP offers a comprehensive suite of enterprise resource planning solutions
  that span across finance, logistics, human resources, supply chain management,
  procurement, and customer relationship management. Many large-scale multinational
  organizations rely heavily on SAP's deeply integrated analytics and data processing
  capabilities to derive meaningful business intelligence from enormous volumes of
  transactional and operational data generated daily across their global operations.
  This often involves numerous complex steps of iterative refinement, validation,
  cleansing, transformation, and final presentation before any coherent analytics
  can be meaningfully derived from the raw underlying data sources.
  """

  state1 = {"raw_text": long_text, "summary": "", "validated_summary": "", "spell_check_notes": ""}
  result1 = chain.invoke(state1)
  print("\nFinal summary:   ", result1["validated_summary"])


  # ── Test 2: short text — expect Pass path directly ───────────────────────────

  print("\n" + "=" * 60)
  print("TEST 2: Short text — expect Pass path on gate 1")
  print("=" * 60)

  short_text = "SAP provides enterprise software for data analytics and business solutions."

  state2 = {"raw_text": short_text, "summary": "", "validated_summary": "", "spell_check_notes": ""}
  result2 = chain.invoke(state2)
  print("\nFinal summary:   ", result2["validated_summary"])

%md
## Module 3: Integrating LangGraph with Enterprise-like Systems

### 1. Introduction

In **Module 3**, you’ll learn how to connect your **LangGraph workflows** and **agentic LLM** logic to **public OData endpoints**, enabling intelligent business processes. We will use the well-known **Northwind OData service** as our example, showing how to incorporate **real data** and actions into your multi-step AI flows.

---

### 2. Key Concepts

1. **OData and REST Endpoints**  
   Many enterprise systems expose OData (or REST) services for business objects (e.g., products, sales orders, or suppliers). You can call these services like any other HTTP/JSON API.

2. **API Authentication**  
   - While our example uses a public, unauthenticated API, real-world enterprise APIs require credentials. These are typically managed via environment variables, secret managers, or platform-specific constructs like SAP BTP Destinations.

3. **LLM Tooling + External APIs**  
   - Treat external API endpoints as “tools” in your **LangGraph** or LLM agent environment.  
   - The LLM decides if/when to call these tools, based on your prompt instructions.

4. **Example Use Case**  
   - Retrieving product data to produce summary reports.  
   - Checking product stock levels, then sending an AI-driven summary to an end user.

---

### 3. Hands-On Code Example

We demonstrate **how to call a public OData endpoint** to retrieve product information from the **Northwind OData service**. We’ll integrate this product-fetching “tool” into a **LangGraph** workflow, letting an LLM from **SAP Generative AI Hub** decide when and how to use the tool for summarizing product data.

The public endpoint we will be using is: `https://services.odata.org/V2/Northwind/Northwind.svc/`

> **Important**  
> This code calls a real, public API endpoint. Make sure you have stable internet access. The service may have usage limits.
import requests
import json
from typing import Optional

# For building a LangGraph node
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool


# We'll reuse the same LLM from previous modules
#from gen_ai_hub.proxy.langchain import init_llm
#llm = init_llm('gpt-4o', max_tokens=512)

# --------------------------------------------------------------------------
# STEP 1: Define the OData service endpoint
# --------------------------------------------------------------------------
NORTHWIND_BASE_URL = "https://services.odata.org/V2/Northwind/Northwind.svc"

# ---------------------------------------------------------------------------
# STEP 2: Define the "fetch_product_data" tool
# ---------------------------------------------------------------------------
def fetch_product_data(product_id: int) -> str:
    """
    Calls the public Northwind OData service to retrieve the given product's data. 
    Returns a short text summary with key fields. The Product ID must be an integer.
    """
    endpoint = f"{NORTHWIND_BASE_URL}/Products({product_id})?$format=json"

    try:
        resp = requests.get(
            endpoint,
            headers={"Accept": "application/json"},
            timeout=15
        )
        resp.raise_for_status()
    except Exception as e:
        return f"[Fetch Error] Could not retrieve Product {product_id}: {e}"

    # The Northwind service returns a different structure
    data = resp.json().get("d", {})
    if not data:
        return f"No data found for Product ID {product_id}."

    name = data.get("ProductName", "Unknown Product")
    quantity_per_unit = data.get("QuantityPerUnit", "N/A")
    price = data.get("UnitPrice", "N/A")
    stock = data.get("UnitsInStock", "N/A")

    return (
        f"Product {product_id}: {name}\n"
        f"Details: {quantity_per_unit}\n"
        f"Price: {price} USD\n" # Assuming USD as currency is not specified
        f"Stock: {stock} units"
    )



# ---------------------------------------------------------------------------
# STEP 3: Define a typed dictionary for workflow state
# ---------------------------------------------------------------------------
class UserState(TypedDict):
    user_query: str       # The raw user prompt
    product_data: str     # Data from the OData service
    final_answer: str     # Summarized final answer

# ---------------------------------------------------------------------------
# STEP 4: Create nodes for the workflow
# ---------------------------------------------------------------------------
from gen_ai_hub.proxy.native.openai import chat 
def extract_and_fetch_node(state: UserState):
    """
    The LLM sees the user's query (which might contain a product ID).
    It has access to the tool 'fetch_product_data'. 
    We'll store the tool's output in 'product_data'.
    """
     # Define the function for the model
    my_tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_product_data",
                "description": "Calls the Northwind OData service to retrieve data for a given product ID. Returns a short text summary with key fields.",
                "parameters": { 
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "integer",
                            "description": "The Product ID, e.g., 1, 2, or 3.",
                        },
                    },
                    "required": ["product_id"],
                },
            }
        }
    ]
    prompt = (
        "You are an intelligent agent. The user said:\n"
        f"\"{state['user_query']}\"\n\n"
        "If the user is asking about a specific product, "
        "call the 'fetch_product_data' tool with the integer product ID given by the user. "
        "Then summarize the product data you find. "
        "If no product is found, explain that you couldn't retrieve details."
    )
    messages=[{"role": "user", "content":prompt}]
    kwargs = dict(model_name='gpt-4o', messages=messages, tools=my_tools, tool_choice="auto")
    response = chat.completions.create(**kwargs)
    response_message = response.choices[0].message

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "fetch_product_data":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")  
                prod_response = fetch_product_data(
                    product_id=function_args.get("product_id")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "fetch_product_data",
                    "content": prod_response,
                })
                return {"product_data": prod_response}
    else:
        # If the LLM decides not to call a tool, return its direct answer.
        return {"product_data": response_message.content or "The model did not call a tool or provide a direct answer."}

def refine_answer_node(state: UserState):
    """
    Once we have the product data from the previous node,
    let's produce a concise, final answer for the user.
    """
    prompt = (
        "You have the following product data:\n"
        f"{state['product_data']}\n\n"
        "Please provide a brief, user-friendly description suitable for a web shop listing."
    )
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# ---------------------------------------------------------------------------
# STEP 5: Assemble the LangGraph workflow
# ---------------------------------------------------------------------------
workflow = StateGraph(UserState)

workflow.add_node("extract_and_fetch", extract_and_fetch_node)
workflow.add_node("refine_answer", refine_answer_node)

workflow.add_edge(START, "extract_and_fetch")
workflow.add_edge("extract_and_fetch", "refine_answer")
workflow.add_edge("refine_answer", END)

chain = workflow.compile()

# ---------------------------------------------------------------------------
# STEP 6: Invoke the chain with a user query
# ---------------------------------------------------------------------------

# Example user prompt referencing a product ID from Northwind (e.g., 1 for "Chai")
user_input_text = "Could you tell me about product 1?"
init_state = {
    "user_query": user_input_text,
    "product_data": "",
    "final_answer": "",
}

final_state = chain.invoke(init_state)
print("=== Workflow Completed ===")
print("Product Data:\n", final_state["product_data"])
print("\nFinal Answer:\n", final_state["final_answer"])
# Example user prompt referencing multiple product IDs
user_input_text = "Could you compare products 1 and 2?"
init_state = {
    "user_query": user_input_text,
    "product_data": "",
    "final_answer": "",
}

final_state = chain.invoke(init_state)
print("=== Workflow Completed ===")
print("Product Data:\n", final_state["product_data"])
print("\nFinal Answer:\n", final_state["final_answer"])
%md
#### How it works

1. **fetch_product_data**  
   - Calls the public Northwind OData service for a specific product ID (e.g. `1`).  
   - Parses JSON fields like **ProductName**, **UnitPrice**, **UnitsInStock**, and returns them in a short text summary.

2. **extract_and_fetch_node**  
   - A LangGraph node that prompts the LLM to analyze the user's query.  
   - The LLM is given the `fetch_product_data` tool and decides whether to call it based on the user's intent.  
   - The raw data returned by the tool is stored in the `product_data` state field.

3. **refine_answer_node**  
   - Another LLM prompt that takes the raw product data and transforms it into a more concise “web shop” style listing.

4. **Workflow Execution**  
   - We invoke the chain with a query like `"Could you tell me about product 1?"`.  
   - The current graph structure is simple and does not support looping to handle multiple products in one query (e.g., comparing two products). This would require a more complex graph.  
   - Inspect the final output to see the data fetched from the Northwind service and the LLM's refined answer.
