%md
# MCP Hands-on - Wikipedia and Web Search Integration

### Welcome to the Advanced MCP Demonstration

In this hands-on tutorial, you'll learn how to build a powerful AI agent that can search both Wikipedia for encyclopedic knowledge and the web for current events - all powered by the Model Context Protocol (MCP)!

**Getting Started:**
- Select the **"Serverless"** option on the Connect button in the top right corner
- Run the installation cell below to set up all required packages
- If you haven't started the course yet, [begin here on Success Map Learning](TBD)
- **Note:** If you pause work for several hours, you may need to restart the kernel (cell 6) and re-run cells in order

**Understanding the Color-Coded Structure:**

Each code cell is color-categorized to help you understand its purpose:

- ⚪ **INFRASTRUCTURE** - Environment setup and package installation
- 🟠 **AI/LLM SETUP** - Configuring your language model connection
- 🟡 **SUPPORTING CODE** - Helper functions and utilities
- 🔵 **MCP CORE** - Core Model Context Protocol concepts
- 🟢 **MCP INTEGRATION** - Connecting MCP with LangChain and SAP AI Core

%md
### ⚪ INFRASTRUCTURE | Setting Up the Development Environment

Before we dive into MCP, we need to install several Python packages that form the foundation of our application:

**Key Packages:**
- **`sap-ai-sdk-gen[all]`** - SAP's AI SDK for connecting to AI Core services
- **`wikipedia`** - Python library for accessing Wikipedia's API
- **`fastmcp`** - The FastMCP framework for building MCP servers quickly
- **`httpx`** - Modern HTTP client for async operations
- **`ddgs`** - DuckDuckGo search library for web searches
- **`langchain-mcp-adapters`** - Bridges MCP servers with LangChain agents
- **`langgraph`** - Advanced orchestration for LangChain workflows

**Important:** After running this cell, you'll see some dependency warnings - this is normal! We'll restart the Python kernel in the next step to ensure everything loads correctly.

# Install SAP AI SDK with all optional features for generative AI
%pip install "sap-ai-sdk-gen[all]==5.10.0" wikipedia fastmcp httpx ddgs
# Install LangChain integration packages for MCP support
# These allow us to use MCP servers seamlessly with LangChain agents
%pip install langchain-mcp-adapters langgraph "langchain[core]"
%md
### ⚪ INFRASTRUCTURE | Restarting the Python Kernel

**Why do we need to restart?**

When you install new Python packages, they aren't immediately available to your running notebook. Restarting the kernel ensures that:
- All newly installed libraries are properly loaded into memory
- There are no conflicts between old and new package versions
- Import statements in subsequent cells will work correctly

This is a **critical step** in Databricks notebooks - don't skip it!

# Restart the Python runtime to load newly installed packages
dbutils.library.restartPython()
%md
### 🟠 AI/LLM SETUP | Creating the Model Selection Widget

Before we start coding, let's set up a convenient dropdown menu that allows you to **switch between different AI models** without modifying code!

**Available Models:**
- **gpt-4o** - OpenAI's most capable model (slower, more expensive)
- **gpt-4o-mini** - Faster, cost-effective version (recommended for this tutorial)
- **mistralai--mistral-large-instruct** - Mistral AI's flagship model
- **gemini-2.5-flash** - Google's fast multimodal model
- **anthropic--claude-4.5-sonnet** - Anthropic's Claude 4.5

The dropdown will appear at the top of your notebook. Try experimenting with different models to see how they perform!

# Create a dropdown widget for easy model selection
# Default is set to gpt-4o-mini for optimal cost/performance balance
dbutils.widgets.dropdown(
    "llm_model_choice",
    "gpt-4o-mini",
    [
        "gpt-5-mini",
        "gpt-4o-mini",
        "mistralai--mistral-large-instruct"
    ]
)
%md
### ⚪ INFRASTRUCTURE | Importing Core Python Libraries

Now that our environment is ready, let's import all the Python libraries we'll need. This cell brings in:

**Standard Library Tools:**
- `json` - For parsing configuration and API responses
- `logging` - For tracking what our application is doing (helps with debugging!)
- `os` - For managing environment variables and file paths
- `asyncio` - For running asynchronous operations
- `tempfile` - For creating temporary files for our MCP servers

**LangChain Components:**
- `ChatPromptTemplate` - For structuring prompts to the LLM
- Message types (`HumanMessage`, `AIMessage`, `ToolMessage`) - For conversation tracking
- `MultiServerMCPClient` - **The star of the show!** This manages multiple MCP servers

**SAP AI Core Integration:**
- `ChatOpenAI` from `gen_ai_hub` - Connects LangChain to SAP AI Core's models

# Standard Python library imports
import json
import logging
import os
import time
from typing import Literal, Type, Union, Dict, Any, List, Callable
import asyncio
import tempfile

# LangChain imports for building AI agents with MCP support
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

# SAP AI Core LLM integration
# This is the LangChain-compatible interface for SAP's Generative AI Hub
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI as GenAIHubChatOpenAI
%md
### ⚪ INFRASTRUCTURE | Configuring Logging and Global Settings

Let's set up our logging system so we can see what's happening behind the scenes!

**Logging Configuration:**
- **Level:** INFO - We'll see informational messages about what's happening
- **Format:** Includes timestamps and severity levels for easy debugging
- **Output:** Directed to stderr (standard error stream) which Databricks captures



# Configure logging to help us see what's happening during execution
logging.basicConfig(
    level=logging.INFO,  # Show INFO level and above (INFO, WARNING, ERROR)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(os.sys.stderr)]  # Databricks captures stderr
)

print("✅ Standard libraries and MCP components imported successfully")
%md
### 🟠 AI/LLM SETUP | Configuring SAP AI Core Authentication

**⚠️ CRITICAL STEP:** This cell sets up authentication with SAP AI Core - without it, we can't access any AI models!

**How SAP AI Core Authentication Works:**

1. **Service Key Retrieval:** We securely fetch credentials from Databricks Secrets
2. **Environment Variables:** The SDK reads authentication from environment variables
3. **OAuth Flow:** Behind the scenes, these credentials enable OAuth2 authentication

**Required Configuration:**
- `AICORE_AUTH_URL` - OAuth authentication endpoint
- `AICORE_CLIENT_ID` - Your application's client identifier
- `AICORE_CLIENT_SECRET` - Secret key (stored securely in Databricks)
- `AICORE_BASE_URL` - API endpoint for AI Core services
- `AICORE_RESOURCE_GROUP` - Organizational unit (usually "default")

**Important:** These environment variables MUST be set BEFORE importing any `gen_ai_hub` modules, as those modules read configuration during initialization.

# ============================================================================
# SAP AI Core Authentication Setup
# ============================================================================
# Retrieve the SAP AI Core Service Key from Databricks Secrets
# Replace scope and key names with your actual Databricks secret configuration
# ============================================================================

try:
    # Fetch the service key JSON from Databricks Secrets
    secret = dbutils.secrets.get(scope="PROD_XGTP_SCOPE", key="LEARNING_GENAIXL")
    svcKey = json.loads(secret)  # Parse JSON string to dictionary

    # Configure environment variables for SAP AI Core SDK
    os.environ["AICORE_AUTH_URL"] = svcKey["url"]
    os.environ["AICORE_CLIENT_ID"] = svcKey["clientid"]
    os.environ["AICORE_CLIENT_SECRET"] = svcKey["clientsecret"]
    os.environ["AICORE_RESOURCE_GROUP"] = "default"  # Or your specific resource group
    os.environ["AICORE_BASE_URL"] = svcKey["serviceurls"]["AI_API_URL"]

    # Confirm successful configuration
    print("✅ SAP AI Core environment variables configured")
    print(f"   • Auth URL: {os.environ['AICORE_AUTH_URL']}")
    print(f"   • Base URL: {os.environ['AICORE_BASE_URL']}")
    print(f"   • Resource Group: {os.environ['AICORE_RESOURCE_GROUP']}")

except Exception as e:
    print(f"❌ CRITICAL: SAP AI Core Authentication Setup Failed: {e}")
    print("   Environment variables MUST be set BEFORE importing gen_ai_hub modules")
    raise  # Stop execution if authentication fails
%md
### 🔵 MCP CORE | Creating the Wikipedia Search MCP Server

Now we're getting to the heart of MCP! This cell creates a **temporary Python script** that defines a complete MCP server for Wikipedia searches.

**Why a separate script file?**

The `langchain-mcp-adapters` library with `stdio` transport expects MCP servers to run as **independent processes**. This mimics real-world production deployments where:
- Servers can be maintained and updated independently
- Multiple applications can connect to the same server
- Servers can be written in any language (not just Python)
- Each server has isolated resources and error handling

**How the Wikipedia Server Works:**

1. **FastMCP Framework:** Uses `FastMCP` to quickly define an MCP-compliant server
2. **Tool Definition:** The `@mcp.tool()` decorator exposes `wiki_search` as an MCP tool
3. **Wikipedia API:** Uses the `wikipedia` library to fetch article summaries
4. **Error Handling:** Gracefully handles disambiguation pages and missing articles
5. **Stdio Transport:** Communicates via standard input/output streams

**Tool Parameters:**
- `query` - What to search for (e.g., "Albert Einstein")
- `top_k_results` - Number of results to return (default: 3)
- `lang` - Wikipedia language code (default: "en")
- `doc_content_chars_max` - Maximum characters per summary (default: 1000)

def create_wiki_search_server_script():
    """
    Creates a temporary Python file containing a FastMCP server for Wikipedia searches.
    
    This function writes a complete, runnable MCP server script to a temporary file.
    The server will be launched as a separate process and communicate via stdio.
    
    Returns:
        str: Path to the temporary Python script file
    """
    # Define the complete MCP server script as a string
    script_content = f"""
import wikipedia
from fastmcp import FastMCP
import asyncio
import os
import logging

# Configure logging for the server process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(os.sys.stderr)]
)

# Initialize the MCP server with a descriptive name
mcp = FastMCP("WikipediaSearcher")

@mcp.tool()
async def wiki_search(
    query: str,
    top_k_results: int = 3,
    lang: str = "en",
    doc_content_chars_max: int = 1000
) -> dict:
    \"\"\"Perform a Wikipedia search and retrieve page summaries.
    
    This tool is best suited for searching specific topics, people, places, or concepts
    that have dedicated Wikipedia articles. It works well for encyclopedic knowledge
    but not for current events or questions.
    
    Args:
        query (str): The search query (e.g., "Albert Einstein", "Python programming")
        top_k_results (int): Number of top results to return (default: 3)
        lang (str): Wikipedia language code (default: "en" for English)
        doc_content_chars_max (int): Maximum characters for each summary (default: 1000)
    
    Returns:
        dict: Dictionary with "summaries" key containing formatted results,
              or "error" key if the search fails
    \"\"\"
    try:
        logging.info(f"Performing Wikipedia search for: '{{query}}' in language: '{{lang}}'")
        
        # Set the Wikipedia language for this search
        wikipedia.set_lang(lang)
        
        # Truncate query to prevent API errors with very long searches
        truncated_query = query[:300]
        
        # Get a list of matching page titles from Wikipedia
        page_titles = wikipedia.search(truncated_query, results=top_k_results)

        if not page_titles:
            logging.info("No Wikipedia pages found for query.")
            return {{"summaries": "No good Wikipedia Search Result was found"}}

        # Fetch and format summaries for each matching page
        summaries = []
        for page_title in page_titles[:top_k_results]:
            try:
                # Fetch the full page object
                wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
                
                # Format the summary with title and truncated content
                summary = f"Page: {{page_title}}\\nSummary: {{wiki_page.summary[:doc_content_chars_max]}}"
                summaries.append(summary)
                
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
                # Skip pages that don't exist or are disambiguation pages
                logging.warning(f"Skipping page '{{page_title}}' due to error: {{e}}")
                continue
            except Exception as e:
                logging.error(f"Error processing Wikipedia page '{{page_title}}': {{e}}")
                continue

        if not summaries:
            logging.info("No valid summaries could be extracted.")
            return {{"summaries": "No good Wikipedia Search Result was found"}}

        # Combine all summaries and enforce maximum length
        combined_summaries = "\\n\\n".join(summaries)[:doc_content_chars_max]
        logging.info(f"Successfully retrieved {{len(summaries)}} summaries.")
        return {{"summaries": combined_summaries}}
        
    except Exception as e:
        logging.error(f"General error during Wikipedia search for '{{query}}': {{e}}")
        return {{"error": f"Wikipedia search failed: {{str(e)}}"}}

# Entry point: Start the MCP server using stdio transport
if __name__ == "__main__":
    asyncio.run(mcp.run(transport="stdio"))
"""
    
    # Write the script to a temporary file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "wiki_search_server.py")
    
    with open(file_path, "w") as f:
        f.write(script_content)
    
    return file_path
%md
### 🔵 MCP CORE | Creating the Web Search MCP Server

Our second MCP server provides web search capabilities using DuckDuckGo!

**Why DuckDuckGo?**

Unlike Wikipedia (which is best for encyclopedic knowledge), DuckDuckGo search helps us find:
- **Current events and news** - Things happening right now
- **Recent developments** - Updates and changes in any field
- **Diverse sources** - Multiple perspectives from across the web
- **No API key required** - Free to use without registration

**How the Web Search Server Works:**

1. **DDGS Library:** Uses DuckDuckGo's search API through the `ddgs` package
2. **Async Wrapper:** The `asyncio.to_thread()` call makes the blocking search async-compatible notice that MCP also allows for HTTP transport
3. **Result Formatting:** Returns title, snippet, and link for each result
4. **Configurable Results:** You can specify how many results to return

**When to Use Web Search vs Wikipedia:**
- **Web Search:** Current events, news, recent developments, diverse opinions
- **Wikipedia:** Historical facts, established concepts, biographical information

The LLM will learn to choose the appropriate tool based on your question!

def create_web_search_server_script():
    """
    Creates a temporary Python file containing a FastMCP server for web searches.
    
    This server uses DuckDuckGo to search the web, complementing Wikipedia searches
    with current information and diverse sources.
    
    Returns:
        str: Path to the temporary Python script file
    """
    script_content = f"""
import asyncio
import os
import logging
from ddgs import DDGS
from fastmcp import FastMCP

# Configure logging for the web search server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(os.sys.stderr)]
)

# Initialize the MCP server
mcp = FastMCP("WebSearcher")

@mcp.tool()
async def web_search(query: str, num_results: int = 4) -> dict:
    \"\"\"Performs a web search using DuckDuckGo to find recent or general information.
    
    This tool is ideal for questions about current events, recent developments,
    or any topic that needs up-to-date information from diverse web sources.
    Unlike Wikipedia, this searches the entire web for recent content.
    
    Args:
        query (str): The search query (e.g., "latest SpaceX launch")
        num_results (int): Number of search results to return (default: 4)
    
    Returns:
        dict: Dictionary with "results" key containing formatted search results,
              or "error" key if the search fails
    \"\"\"
    logging.info(f"Performing DuckDuckGo search for: '{{query}}'")
    
    try:
        # Use asyncio.to_thread to run the synchronous (blocking) DDGS search
        # in a separate thread, making it compatible with async operations
        results = await asyncio.to_thread(
            DDGS().text,  # The text search method
            query=query,
            region='us-en',  # Search from US English region
            safesearch='moderate',  # Filter inappropriate content
            max_results=num_results
        )

        if not results:
            logging.warning(f"No results found for query: '{{query}}'")
            return {{"results": "No results found."}}

        # Format each result with title, snippet, and link
        formatted_results = [
            f"Title: {{res.get('title')}}\\nSnippet: {{res.get('body')}}\\nLink: {{res.get('href')}}"
            for res in results
        ]
        
        logging.info(f"Found {{len(results)}} results for '{{query}}'.")
        return {{"results": "\\n\\n".join(formatted_results)}}

    except Exception as e:
        logging.error(f"Error during DuckDuckGo search for '{{query}}': {{e}}", exc_info=True)
        return {{"error": f"An unexpected error occurred during the web search: {{str(e)}}"}}

# Entry point: Start the MCP server using stdio transport
if __name__ == "__main__":
    asyncio.run(mcp.run(transport="stdio"))
"""
    
    # Write the script to a temporary file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "web_search_server.py")
    
    with open(file_path, "w") as f:
        f.write(script_content)
    
    return file_path
%md
### 🔵 MCP CORE | Generating MCP Server Script Files

Now let's actually create those temporary server files!

This cell calls the two functions we just defined to generate:
1. **`wiki_search_server.py`** - Wikipedia search server
2. **`web_search_server.py`** - Web search server

Both files are saved to your system's temporary directory (usually `/tmp` on Linux/Mac or `C:\Temp` on Windows).

**What happens next?**

These script files will be launched as separate Python processes by the `MultiServerMCPClient` in the next section. Each server will:
- Run independently with its own resources
- Communicate via standard input/output (stdio)
- Be discoverable through the MCP protocol
- Automatically expose their tools to the LangChain agent

# Generate the MCP server script files
web_search_script_path = create_web_search_server_script()
wiki_search_script_path = create_wiki_search_server_script()

# Display the file paths for reference
print(f"Web Search server script saved to: {web_search_script_path}")
print(f"Wikipedia Search server script saved to: {wiki_search_script_path}")
print("\n✅ FastMCP servers defined and written to temporary files.")
print("   These servers will be launched as separate processes in the next step.")
%md
### 🟢 MCP INTEGRATION | Connecting MCP Servers with LangChain

This is where the magic happens! We're now going to connect our MCP servers to LangChain using the `MultiServerMCPClient`.

**Understanding MultiServerMCPClient:**

The `MultiServerMCPClient` is a powerful component that:
- **Manages multiple MCP servers** - Each server runs as a separate process
- **Handles communication** - Uses stdio to send requests and receive responses, for other use cases you can consider [HTTP transport](https://docs.langchain.com/oss/python/langchain/mcp#http)
- **Discovers tools automatically** - Each server exposes its tools via MCP protocol
- **Converts to LangChain format** - Makes MCP tools work seamlessly with LangChain agents

**Configuration Structure:**

For each server, we specify:
- **Server name** - A unique identifier (e.g., "web_search")
- **transport** - How to communicate ("stdio" = standard input/output)
- **command** - What program to run ("python")
- **args** - Arguments to pass (the path to our server script)

**What happens when we call `get_tools()`?**

1. The client launches each server as a subprocess
2. It sends an MCP "list_tools" request to each server
3. Each server responds with its available tools and their schemas
4. The client converts these into LangChain-compatible tool objects
5. These tools can now be used by any LangChain agent!

# ============================================================================
# Initialize MultiServerMCPClient with our two MCP servers
# ============================================================================

client = MultiServerMCPClient(
    {
        # Web Search Server Configuration
        "web_search": {
            "transport": "stdio",  # Communicate via standard input/output
            "command": "python",   # Use Python to run the script
            "args": [web_search_script_path]  # Path to our web search server
        },
        
        # Wikipedia Search Server Configuration
        "wikipedia_search": {
            "transport": "stdio",
            "command": "python",
            "args": [wiki_search_script_path]  # Path to our Wikipedia server
        }
    }
)

# ============================================================================
# Discover and retrieve all tools from both MCP servers
# ============================================================================
# This async call:
# 1. Launches both server processes
# 2. Queries each for their available tools
# 3. Converts MCP tools to LangChain-compatible format
# 4. Returns a list of tools ready to use with our agent

tools = await client.get_tools()

print(f"\n✅ Successfully connected to MCP servers!")
print(f"   Found {len(tools)} tools ready for use:")
for tool in tools:
    print(f"   • {tool.name}: {tool.description[:80]}...")
%md
### 🟡 SUPPORTING CODE | Inspecting Available Tools

Let's take a closer look at what tools we now have available!

This cell prints the full tool list, showing you:
- **Tool names** - The identifier the LLM will use to call them
- **Descriptions** - How the LLM understands what each tool does
- **Parameter schemas** - What arguments each tool accepts
- **Metadata** - Additional information about the tool

Understanding these details helps you see how the LLM will "think" about when to use each tool!

# Display detailed information about each available tool
print("\n" + "="*70)
print("📋 DETAILED TOOL INFORMATION")
print("="*70)

for i, tool in enumerate(tools, 1):
    print(f"\n{i}. Tool Name: {tool.name}")
    print(f"   Description: {tool.description}")
    print(f"   Parameters: {tool.args_schema}")
    print("-" * 70)
%md
### 🟢 MCP INTEGRATION | Creating the LangChain Agent

Now that we have our tools ready, let's create an AI agent that can use them!

**What is a LangChain Agent?**

A LangChain agent is an AI system that can:
- **Reason about problems** - Understand what needs to be done
- **Choose tools dynamically** - Decide which tool(s) to use
- **Execute actions** - Call the selected tools with appropriate parameters
- **Process results** - Interpret tool outputs and continue reasoning
- **Provide answers** - Synthesize information into helpful responses

**Agent Components:**

1. **LLM (Language Model):** The "brain" that does the reasoning
   - We initialize it from SAP AI Core using your selected model
   - Temperature=0.0 for consistent, deterministic responses
   - Max 1024 tokens for reasonably sized responses

2. **Tools:** The "hands" that can interact with external systems
   - Our two MCP tools (web_search and wiki_search)
   - Automatically discovered and integrated

3. **Agent Framework:** The "nervous system" that coordinates everything
   - Created with `create_agent()` from LangChain
   - Handles the reasoning loop automatically

# ============================================================================
# Initialize the Language Model from SAP AI Core
# ============================================================================

from langchain.agents import create_agent
from gen_ai_hub.proxy.langchain.init_models import init_llm

# Get the selected model name from the widget
llm_model_name = dbutils.widgets.get("llm_model_choice")

# Initialize the LLM with our configuration
llm = init_llm(
    llm_model_name,
    temperature=0.0,  # Deterministic responses (no randomness)
    max_tokens=1024   # Maximum length of responses
)

print(f"✅ Initialized LLM: {llm_model_name}")
print(f"   Configuration: temperature=0.0, max_tokens=1024")

# ============================================================================
# Create the LangChain Agent with MCP Tools
# ============================================================================

agent = create_agent(
    llm,    # The language model that will do the reasoning
    tools   # The MCP tools it can use (web_search and wiki_search)
)

print("\n✅ Agent created successfully!")
print(f"   The agent can now use {len(tools)} tools to answer your questions.")
print("\n🎉 Ready to start querying!")
%md
### 🟢 MCP INTEGRATION | Testing Web Search Functionality

Let's test our agent with a question that requires web search!

**Why this question is perfect for web search:**
- "Starship flights" refers to current/recent events
- SpaceX's Starship program is actively developing
- Web search will find recent news articles and updates
- Wikipedia might be outdated for such rapidly evolving topics

**What to observe:**
1. The agent receives your question
2. It recognizes this needs current information
3. It chooses to call the `web_search` tool
4. The MCP server performs the search via DuckDuckGo
5. Results come back with titles, snippets, and links
6. The agent synthesizes this into a coherent answer

Watch the logs to see the HTTP requests to SAP AI Core!

# ============================================================================
# Test the Agent with a Web Search Query
# ============================================================================

# Invoke the agent with a question about recent events
response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What are the web search results for Starship flights?"}]}
)

# Display the complete response for analysis
print("\n" + "="*70)
print("🔍 COMPLETE AGENT RESPONSE")
print("="*70)
print(response)
print("="*70)
%md
### 🟡 SUPPORTING CODE | Agent Trace Visualization Function

Looking at raw agent responses can be overwhelming! Let's create a helper function that makes the agent's decision-making process crystal clear.

**What This Function Does:**

The `display_agent_trace()` function parses the complex response dictionary and presents it in a human-readable format showing:

1. **👤 User Input:** Your original question
2. **🤖 Agent Decision:** Which tool(s) the agent decided to call and why
3. **🛠️ Tool Results:** The raw data returned by each MCP server
4. **✅ Final Answer:** The agent's synthesized response

**Message Types Explained:**
- `HumanMessage` - Your input to the agent
- `AIMessage` - The agent's reasoning and tool calls
- `ToolMessage` - Results returned by MCP tools

This is incredibly valuable for learning and debugging!

import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def display_agent_trace(response: dict):
    """
    Parses and displays the agent's decision-making process in a readable format.
    
    This function helps you understand:
    - What question was asked
    - How the agent decided to respond
    - What tools were called
    - What results were returned
    - How the final answer was formulated
    
    Args:
        response: The dictionary returned by agent.ainvoke()
    """
    print("\n" + "="*70)
    print("🕵️ AGENT INTERACTION TRACE")
    print("="*70)
    
    # Extract the message history from the response
    messages = response.get('messages', [])
    if not messages:
        print("❌ No messages found in the response.")
        return

    # Process each message in the conversation
    for i, message in enumerate(messages):
        
        # 1. Display User Input
        if isinstance(message, HumanMessage):
            print(f"\n👤 Step 1: User Input")
            print(f"   Question: '{message.content}'")
            print("-" * 70)

        # 2. Display Agent's Decision
        elif isinstance(message, AIMessage):
            # Check if this is a tool call or final answer
            if message.tool_calls:
                print("\n🤖 Step 2: Agent Decision - Tool Call(s)")
                print("   The agent has decided to use the following tool(s):")
                
                for j, tool_call in enumerate(message.tool_calls, 1):
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    print(f"\n   Tool Call #{j}:")
                    print(f"      • Tool: `{tool_name}`")
                    print(f"      • Arguments: {json.dumps(tool_args, indent=8)}")
            
            # This is the final answer
            elif i == len(messages) - 1:
                print("\n✅ Step 4: Agent's Final Answer")
                print("-" * 70)
                # Use display() for rich rendering in Databricks
                try:
                    display(message.content)
                except NameError:
                    # Fallback for non-Databricks environments
                    print(message.content)
                print("-" * 70)

        # 3. Display Tool Results
        elif isinstance(message, ToolMessage):
            print(f"\n🛠️ Step 3: Tool Result from `{message.name}`")
            
            # The actual data is usually in the artifact
            if message.artifact and 'structured_content' in message.artifact:
                tool_output = message.artifact['structured_content']
                pretty_output = json.dumps(tool_output, indent=2)
                print(f"   Structured Output:")
                print(f"{pretty_output}")
            else:
                # Fallback if structure is different
                print(f"   Raw Content: {message.content}")
            print("-" * 70)

    print("\n" + "="*70)
    print("END OF TRACE")
    print("="*70)
%md
### 🟡 SUPPORTING CODE | Visualizing the Web Search Trace

Now let's use our visualization function to see exactly what happened during the Starship flights query!

**What You'll See:**

1. **Your Question:** "What are the web search results for Starship flights?"
2. **Agent's Tool Choice:** It decided to use `web_search` with the query parameter
3. **Search Results:** Titles, snippets, and links from DuckDuckGo
4. **Final Answer:** The agent's formatted presentation of the findings

This trace shows the complete reasoning cycle - from question to answer!

# Display the detailed trace of the web search interaction
display_agent_trace(response)
%md
### 🟢 MCP INTEGRATION | Testing Wikipedia Search Functionality

Now let's test with a question that's perfect for Wikipedia!

**Why This Question Suits Wikipedia:**
- Asks for comparison of established organizations
- Requires encyclopedic, factual information
- Both SpaceX and Blue Origin have comprehensive Wikipedia articles
- Historical context is more important than breaking news

**Expected Agent Behavior:**
1. Agent recognizes this needs encyclopedic comparison
2. Calls `wiki_search` tool twice (once for each company)
3. Retrieves summaries from their respective Wikipedia pages
4. Synthesizes a comparison based on the retrieved information

**Watch for:** The agent might call `wiki_search` multiple times in parallel or sequentially!

# ============================================================================
# Test the Agent with a Wikipedia Query
# ============================================================================

wiki_response = await agent.ainvoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Compare SpaceX and Blue Origin based on their Wikipedia entries?"
            }
        ]
    }
)

print("\n" + "="*70)
print("🔍 WIKIPEDIA SEARCH RESPONSE")
print("="*70)
print(wiki_response)
print("="*70)
%md
### 🟡 SUPPORTING CODE | Visualizing the Wikipedia Search Trace

Let's see how the agent handled the Wikipedia comparison!

**What to Look For:**
- Did the agent call `wiki_search` once or twice?
- How did it formulate the queries for each company?
- What information did Wikipedia return?
- How did the agent synthesize the comparison?

# Display the detailed trace of the Wikipedia search interaction
display_agent_trace(wiki_response)
%md
### 🟢 MCP INTEGRATION | Testing Multi-Tool Usage

Now for the ultimate test - a question that requires BOTH tools!

**Why This Question Needs Both Tools:**

"Explain what CRISPR gene-editing technology is, and find some recent news about its use in clinical trials."

This question has two distinct parts:
1. **"What is CRISPR?"** → Perfect for Wikipedia (established concept)
2. **"Recent news about clinical trials"** → Perfect for web search (current events)

**Expected Agent Behavior:**
1. Recognizes the dual nature of the question
2. Calls `wiki_search` to get CRISPR explanation
3. Calls `web_search` to find recent clinical trial news
4. Combines information from both sources into a cohesive answer

This demonstrates the agent's ability to use multiple tools intelligently!

# ============================================================================
# Test the Agent with a Multi-Tool Query
# ============================================================================

web_and_wiki_response = await agent.ainvoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain what CRISPR gene-editing technology is, and find some recent news about its use in clinical trials."
            }
        ]
    }
)

print("\n" + "="*70)
print("🔍 MULTI-TOOL RESPONSE (Wikipedia + Web Search)")
print("="*70)
print(web_and_wiki_response)
print("="*70)
%md
### 🟡 SUPPORTING CODE | Visualizing the Multi-Tool Trace

This trace will be the most interesting yet!

**What to Observe:**
- The agent's decision to use both tools
- The order in which tools are called
- How results from each tool are used
- The final synthesis combining both information sources

This showcases the true power of MCP - seamless integration of multiple specialized tools!

# Display the detailed trace of the multi-tool interaction
display_agent_trace(web_and_wiki_response)
%md
## 🎓 Exercise: Experiment with Different Query Types

Now that you've seen how the agent uses different tools, it's your turn to explore!

### Your Challenge:

Test the agent with **3 different types of questions** and observe which tool(s) it chooses to use:

**1. Wikipedia-Only Query**
- Create a question that should ONLY use Wikipedia
- Example topics: Historical figures, scientific concepts, geographical locations
- Example: "Tell me about Marie Curie's contributions to science"

**2. Web-Search-Only Query**
- Create a question that should ONLY use web search
- Example topics: Recent news, current events, today's information
- Example: "What are the latest developments in artificial intelligence this week?"

**3. Mixed Query (Both Tools)**
- Create a question that requires BOTH Wikipedia and web search
- Combine a factual explanation with current information
- Example: "What is quantum computing and what are the recent breakthroughs?"

### Steps to Complete:

1. Write your three questions in the cell below
2. Run each question through the agent using `agent.ainvoke()`
3. Use `display_agent_trace()` to see which tools were used
4. Answer the reflection questions

### What to Learn:

- How does the agent **decide** which tool to use?
- What **keywords** trigger web search vs. Wikipedia?
- Can you "trick" the agent into using the wrong tool?
- How does the agent handle **ambiguous** questions?

**Ready? Start experimenting below!** 🔬

%md
## 📚 Complete the Knowledge Check and Assignment

### Next Steps:

1. **Knowledge Check on Success Map Learning**
   - Complete the [Knowledge Check quiz](TBD)
   - This is required to record your course completion

2. **Hands-on Assignment on ML Workbench**
   - Submit your completed exercise (your custom MCP server)
   - This qualifies you for the **Credly Badge for AI Developer Skills**

### What You've Learned:
- ✅ How to set up MCP servers as independent processes
- ✅ The difference between Wikipedia search and web search
- ✅ How to integrate MCP with LangChain agents
- ✅ How agents decide which tools to use
- ✅ How to trace and debug agent decision-making
- ✅ How to build production-ready MCP architectures

### Thank You!

You've completed an advanced tutorial on the Model Context Protocol. You now understand how to build scalable, modular AI systems that can integrate multiple specialized tools!

**Questions?** Join our [*Teams channel*](https://teams.microsoft.com/l/channel/19%3AKZsG47NTYJN60sptRZuDdxXKUMVqACHP9oUGDG_sGTo1%40thread.tacv2/Questions%20and%20Answers?groupId=d9db4a55-e955-451f-8970-d36645e0bad6&tenantId=42f7676c-f455-423c-82f6-dc2d99791af7) to ask any questions you may have.

%md

      # ============================================================================
  # Use Case Description:
  # An entertainment agent combining TV Maze (show details) and Quotable API                                                                    
  # (famous quotes). Ask about any TV show and the agent returns cast/plot info
  # paired with a thematic quote matching the show's mood.
  # ============================================================================

  import os, json, logging, tempfile, sys, asyncio
  from langchain_core.messages import HumanMessage, ToolMessage
  from langchain_mcp_adapters.client import MultiServerMCPClient
  from gen_ai_hub.proxy.langchain.init_models import init_llm

  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  secret = dbutils.secrets.get(scope="PROD_XGTP_SCOPE", key="LEARNING_GENAIXL")
  svcKey = json.loads(secret)
  os.environ["AICORE_AUTH_URL"]       = svcKey["url"]
  os.environ["AICORE_CLIENT_ID"]      = svcKey["clientid"]
  os.environ["AICORE_CLIENT_SECRET"]  = svcKey["clientsecret"]
  os.environ["AICORE_RESOURCE_GROUP"] = "default"
  os.environ["AICORE_BASE_URL"]       = svcKey["serviceurls"]["AI_API_URL"]
  print("✅  SAP AI Core configured")


  def create_tvmaze_server_script():
      script = "\n".join([
          "import asyncio, os, re, logging, httpx",
          "from fastmcp import FastMCP",
          "",
          "logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(os.sys.stderr)])",
          "mcp = FastMCP('TVMazeServer')",
          "",
          "@mcp.tool()",
          "async def get_tv_show(show_name: str) -> dict:",
          '    """Fetches TV show details from TV Maze: summary, genres, status, and cast.',
          "    Use this for any question about a TV show, its plot, cast, or status.",
          "    Args:",
          "        show_name: Name of the TV show e.g. Friends, The Office, Seinfeld",
          "    Returns:",
          "        Dict with name, genres, status, premiered, rating, cast, summary.",
          '    """',
          "    try:",
          "        async with httpx.AsyncClient(timeout=15) as client:",
          "            resp = await client.get(",
          "                'https://api.tvmaze.com/singlesearch/shows',",
          "                params={'q': show_name, 'embed': 'cast'}",
          "            )",
          "            resp.raise_for_status()",
          "            data = resp.json()",
          "        summary = re.sub(r'<[^>]+>', '', data.get('summary') or 'No summary available.')",
          "        cast_list = data.get('_embedded', {}).get('cast', [])[:5]",
          "        cast = ', '.join(",
          "            f\"{c['person']['name']} as {c['character']['name']}\"",
          "            for c in cast_list if c.get('person') and c.get('character')",
          "        ) or 'Cast not available'",
          "        return {",
          "            'name': data.get('name', 'Unknown'),",
          "            'genres': ', '.join(data.get('genres', [])) or 'Unknown',",
          "            'status': data.get('status', 'Unknown'),",
          "            'premiered': data.get('premiered', 'Unknown'),",
          "            'rating': data.get('rating', {}).get('average', 'N/A'),",
          "            'cast': cast,",
          "            'summary': summary[:600]",
          "        }",
          "    except Exception as e:",
          "        return {'error': f'TV Maze lookup failed: {str(e)}'}",
          "",
          "if __name__ == '__main__':",
          "    asyncio.run(mcp.run(transport='stdio'))",
      ])
      path = os.path.join(tempfile.gettempdir(), "tvmaze_server.py")
      with open(path, "w") as f:
          f.write(script)
      return path


  def create_quotes_server_script():
      script = "\n".join([
          "import asyncio, os, logging, httpx",
          "from fastmcp import FastMCP",
          "",
          "logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(os.sys.stderr)])",
          "mcp = FastMCP('QuotesServer')",
          "",
          "@mcp.tool()",
          "async def get_quote(tag: str = 'friendship') -> dict:",
          '    """Fetches a famous quote filtered by topic from the Quotable API.',
          "    Use this to find a thematic quote matching the mood of a TV show.",
          "    Good tags: friendship, humor, life, love, wisdom, happiness, family.",
          "    Args:",
          "        tag: Topic keyword to filter quotes, default is friendship",
          "    Returns:",
          "        Dict with quote content and author name.",
          '    """',
          "    urls = [",
          "        ('https://api.quotable.io/random', {'tags': tag}),",
          "        ('https://api.quotable.io/random', {}),",
          "    ]",
          "    for url, params in urls:",
          "        for verify in [True, False]:",
          "            try:",
          "                async with httpx.AsyncClient(timeout=10, verify=verify) as client:",
          "                    resp = await client.get(url, params=params)",
          "                    resp.raise_for_status()",
          "                    data = resp.json()",
          "                    if isinstance(data, list):",
          "                        data = data[0] if data else {}",
          "                    if data.get('content'):",
          "                        logging.info(f'Quote fetched successfully (verify={verify})')",
          "                        return {",
          "                            'quote': data.get('content', 'No quote found.'),",
          "                            'author': data.get('author', 'Unknown')",
          "                        }",
          "            except Exception as e:",
          "                logging.warning(f'Quote attempt failed (url={url}, verify={verify}): {e}')",
          "                continue",
          "    return {'quote': 'Every day is a new beginning.', 'author': 'Unknown'}",
          "",
          "if __name__ == '__main__':",
          "    asyncio.run(mcp.run(transport='stdio'))",
      ])
      path = os.path.join(tempfile.gettempdir(), "quotes_server.py")
      with open(path, "w") as f:
          f.write(script)
      return path


  tvmaze_path = create_tvmaze_server_script()
  quotes_path = create_quotes_server_script()
  print(f"✅  TV Maze server written to: {tvmaze_path}")
  print(f"✅  Quotes server written to:  {quotes_path}")


  async def main():
      client = MultiServerMCPClient({
          "tvmaze": {
              "transport": "stdio",
              "command": sys.executable,
              "args": ["-u", tvmaze_path]
          },
          "quotes": {
              "transport": "stdio",
              "command": sys.executable,
              "args": ["-u", quotes_path]
          }
      })

      tools = await client.get_tools()
      print(f"\n✅  Connected via MultiServerMCPClient — {len(tools)} tools discovered:")
      for t in tools:
          print(f"   • {t.name}: {t.description[:80]}...")

      tool_map = {t.name: t for t in tools}
      llm = init_llm("gpt-4o-mini", temperature=0.0, max_tokens=1024)
      llm_with_tools = llm.bind_tools(tools)

      async def run_agent(question: str) -> str:
          messages = [HumanMessage(content=question)]
          response = await llm_with_tools.ainvoke(messages)
          if response.tool_calls:
              messages.append(response)
              for tool_call in response.tool_calls:
                  print(f"   → Calling tool: {tool_call['name']} with args: {tool_call['args']}")
                  result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
                  print(f"   → Tool result: {str(result)[:200]}")
                  messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
              final = await llm_with_tools.ainvoke(messages)
              return final.content
          return response.content

      # Test 1a: TV Maze individually
      print("\n--- Test 1a: TV Maze (Friends) ---")
      result = await tool_map["get_tv_show"].ainvoke({"show_name": "Friends"})
      print(result)

      # Test 1b: Quotable individually
      print("\n--- Test 1b: Quotable API (friendship quote) ---")
      result = await tool_map["get_quote"].ainvoke({"tag": "friendship"})
      print(result)

      # Test 2: Single-tool agent query
      print("\n--- Test 2: Single Tool (TV Maze — Seinfeld) ---")
      print(await run_agent("Tell me about Seinfeld — what genres does it cover and is it still airing?"))

      # Test 3: Combined query — both tools
      print("\n--- Test 3: Combined Query (TV Maze + Quotable) ---")
      print(await run_agent(
          "Tell me about the TV show Friends — its plot, cast, and how long it ran. "
          "Then find a famous quote about friendship that captures the spirit of the show."
      ))

  await main()
