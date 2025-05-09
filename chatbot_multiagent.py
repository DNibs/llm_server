import os
from dotenv import load_dotenv
import datetime
import logging
import io
import json
import gradio as gr
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains.summarize import load_summarize_chain
from langchain.globals import set_verbose, set_debug
from transformers import AutoTokenizer
from typing import List, Any



# Prepare global variables and environemnt ==================================
# ===========================================================================================
load_dotenv()  # Load environment variables from .env file
MODEL_NAME = os.getenv('MODEL_NAME') # Local model name
MODEL_TOKENIZER_PATH = os.getenv('MODEL_TOKENIZER_PATH') # Local tokenizer path
SET_PROMPT = os.getenv('SET_PROMPT') # Local prompt template
MAX_TOKENS = int(os.getenv('MAX_TOKENS')) # Local max tokens
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') # Local TAVILY API key
os.environ["OPENAI_API_KEY"] = "lm-studio"  # dummy key; LM Studio doesn't require real auth
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"  # LM Studio local server


# Set up logging ========================================================
# ===========================================================================================
# creates a stream handler to capture logs in a string buffer
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Set up root logger to capture all logs
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


# Set up Tokeinzer and model ========================================================
# ===========================================================================================
# Tokenizer from file for counting tokens in messages
tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER_PATH)

# Set up model (ChatOpenAI uses OpenAI-compatible endpoints)
LLM = ChatOpenAI(
    temperature=0.7,
    model_name=MODEL_NAME,  # name is arbitrary for local usage
)


# Message and Prompt Functions ===================================================
# ===========================================================================================

# LangGraph prompt template; setting prompt as system message avoids trimming
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SET_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])


def custom_token_counter(messages: List[Any]) -> int:
    """Token counter function compatible with langgraph_core.messages.trim_messages"""
    total_tokens = 0
    if not messages:
        return 0
    for msg in messages:
        total_tokens += len(tokenizer.encode(msg.content, add_special_tokens=False))
    return total_tokens


# Avoids going over in context tokens
trimmer = trim_messages(
    max_tokens=MAX_TOKENS,
    strategy="last",
    token_counter=custom_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


# Save and load state functions =============================================================
# ===========================================================================================
def serialize_state(state):
    """Convert all LangChain BaseMessage objects to serializable dicts."""
    def convert(obj):
        if isinstance(obj, BaseMessage):
            return obj.model_dump(mode="json")  # Ensure JSON-safe format
        elif isinstance(obj, dict):
            return {convert(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(i) for i in obj)
        elif isinstance(obj, set):
            return list(convert(i) for i in obj)  # JSON doesn't support sets
        else:
            return obj
    return convert(state)


def save_state(state, filepath):
    """Saves chatbot message history to JSON file. serialize_state converts custom classes to json-serializable dicts."""
    serializable_state = serialize_state(state)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_state, f, indent=2)
    return f"State saved to {filepath}"


def rehydrate_messages(state):
    """Convert all serializable dicts back to LangChain BaseMessage objects."""
    def convert(item):
        if isinstance(item, dict) and "type" in item and "content" in item:
            if item["type"] == "human":
                return HumanMessage(content=item["content"])
            elif item["type"] == "ai":
                return AIMessage(content=item["content"])
            elif item["type"] == "system":
                return SystemMessage(content=item["content"])
            elif item["type"] == "tool":
                if "tool_call_id" not in item:
                    raise ValueError("Missing 'tool_call_id' when trying to load ToolMessage.")
                return ToolMessage(
                    content=item["content"],
                    tool_call_id=item["tool_call_id"]
                )
        elif isinstance(item, list):
            return [convert(i) for i in item]
        elif isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        else:
            return item
    return convert(state)


def convert_messages_openai_to_gradio(state):
    """Converts messages from OpenAI BaseMessage format to Gradio chat_history format."""
    chat_history = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage): 
            chat_history.append({"role": "user", "content": message.content})
        if isinstance(message, AIMessage): 
            chat_history.append({"role": "assistant", "content": message.content})
    return chat_history


def load_state(filepath_list, config):
    """Loads json and converts to LangChain BaseMessage objects for chatbot state. Also passes chat_history to gradio."""
    if filepath_list is None or len(filepath_list) == 0:
        return
    if isinstance(filepath_list, list):
        fp = filepath_list[0]
    elif isinstance(filepath_list, str):
        fp = filepath_list
    else:
        raise ValueError("Invalid input type for filepath_list. Expected list or str.")
    config["configurable"]["thread_id"] = str(int(config["configurable"]["thread_id"]) + 1) 
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
        new_state = rehydrate_messages(data)[0]
        app.invoke(new_state, config)  # Rehydrate the state in the graph
    chat_history = convert_messages_openai_to_gradio(new_state)
    return chat_history, config


# Web Search Agent / Tool ===================================================================
# ===========================================================================================
custom_map_prompt = PromptTemplate.from_template("""
You are a research assistant. Summarize the following documents into a single coherent response.
Please be as verbose, and share specific details from the documents whenever relevant. 
Reference the sources in your response.
                                                 
{text}"""
                                                 )
summarizer = load_summarize_chain(LLM, chain_type="map_reduce", map_prompt=custom_map_prompt)
@tool
def websearch(query: str) -> dict:
    """
    Performs a web search and returns both a summary and a list of sources.
    """
    # Run the search
    search_tool = TavilySearchResults(max_results=3)
    search_results = search_tool.invoke({"query": query})

    # Convert to Document format for summarizer
    documents = [
        Document(page_content=result["content"], metadata={"source": result["url"]})
        for result in search_results
    ]
    
    # Summarize results
    summarizer = load_summarize_chain(LLM, chain_type="stuff")
    summary = summarizer.run(documents)

    # Extract URLs (or titles + URLs)
    sources = [
        {"title": result.get("title", "Untitled"), "url": result.get("url")}
        for result in search_results
    ]

    return {
        "summary": summary,
        "sources": sources
    }
    
tools = [websearch]
tool_node = ToolNode(tools)

def should_continue(state: MessagesState):
    """Manages the flow of the graph based on tool calls based on last message."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END
    
LLM_WITH_TOOLS = LLM.bind_tools(tools)


# Initiate the graph ===============================================================
# ===========================================================================================

# Node 1: Chatbot Agent (calls tools as necessary)
def chatbot_agent(state: MessagesState):
    """Prepares messages for and invokes LLM"""
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({'messages': trimmed_messages})
    response = LLM_WITH_TOOLS.invoke(prompt)
    return {'messages': response}


# Node 2: Research Agent (handles websearch tool call and returns summary)
def research_agent(state: MessagesState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    results = []
    for call in tool_calls:
        if call['name'] == 'websearch':
            query = call['args']['query']
            output = websearch(query)  # returns dict with 'summary' and 'sources'
            content = f"**Summary:**\n{output['summary']}\n\n"
            content += "**Sources:**\n" + "\n".join(
                f"- [{src['title']}]({src['url']})" for src in output["sources"]
            )
            
            results.append(ToolMessage(tool_call_id=call['id'], content=content))

    return {'messages': results}


# Define the single node in graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node('chatbot_agent', chatbot_agent)
workflow.add_node('research_agent', research_agent)

workflow.add_edge(START, 'chatbot_agent')
workflow.add_conditional_edges('chatbot_agent', should_continue, {
    'tools': 'research_agent',
    END: END
})
workflow.add_edge('research_agent', 'chatbot_agent')  # Send back summary for response

# Add memory to graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Functions gradio wil execute ======================================================
# ===========================================================================================
def query_response(chat_history, query, config):
    """Defines the i/o between langgraph and gradio with streaming enabled."""
    input_messages = [HumanMessage(query)]  # wraps in langgraph format
    chat_history.append({"role": "user", "content": query})

    # Initiate empty variables to be filled in during streaming
    assistant_message = {"role": "assistant", "content": ''}  
    chat_history.append(assistant_message)  # streaming will fill in pointer here
    response = ''  

    # Get state mem to calculate token count
    state_values = app.get_state(config).values # Might be empty hence condition below
    if 'messages' in state_values:
        token_count = custom_token_counter(state_values['messages']) 
        token_count += custom_token_counter(input_messages)
    else:
        token_count = custom_token_counter(input_messages)
    
    # chunks/metadata is streamed per-token output only
    for chunk, metadata in app.stream(
        {"messages": input_messages}, 
        config, 
        stream_mode='messages',
        ): 
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            response += chunk.content
            assistant_message["content"] = response

        yield chat_history, '', f'Thread_ID: {metadata["thread_id"]}; Token Count: {token_count} of {MAX_TOKENS}', app.get_state(config) 


def clear_fn(config):
    """Iterates thread id to clear state memory and returns empty values to gradio"""
    config["configurable"]["thread_id"] = str(int(config["configurable"]["thread_id"]) + 1) 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return [], '', config, f'Thread ID: {config["configurable"]["thread_id"]}', timestamp


def print_logger():
    """Prints the log stream to log window."""
    log_contents = log_stream.getvalue()
    if log_contents:
        return log_contents
    else:
        return "No logs available."


# Sets time and unique value for each thread
config_id = {"configurable": {"thread_id": "1"}}
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('logs', exist_ok=True)
filepath = os.path.join(os.getcwd(), 'logs', f'state_{timestamp}.json')


# Gradio UI ============================================================================
# ===========================================================================================
with gr.Blocks() as demo:
    gr.Markdown("### 🤖 Local LLM Chatbot via LangGraph + LM Studio") 
    chat_history = gr.Chatbot(type="messages")
    config_state = gr.State(config_id)
    msg_state = gr.State(app.get_state(config_id))
    filepath_state = gr.State(filepath)
    msg = gr.Textbox(placeholder='Type your message here!')

    send = gr.Button("Send")
    with gr.Row():
        clear = gr.Button("Clear History", scale=1)
        save = gr.Button("Save State", scale=1)
        load = gr.Button("Load State", scale=1)
    
    info = gr.Markdown('Thread ID: 1')
    
    file_explorer = gr.FileExplorer(label='Choose log file to load', file_count='single', root_dir='logs', height=200)
  
    log_window = gr.Textbox(label='Log Window', interactive=False, lines=10, max_lines=50)
    print_log = gr.Button('Print Logs')

    # Works on 'enter' key press
    msg.submit(
        query_response, 
        inputs=[chat_history, msg, config_state], 
        outputs=[chat_history, msg, info, msg_state],
        )
    
    send.click(
        query_response, 
        inputs=[chat_history, msg, config_state], 
        outputs=[chat_history, msg, info, msg_state],
        )

    clear.click(
        clear_fn,
        inputs=[config_state],
        outputs=[chat_history, msg, config_state, info, filepath_state],
        )

    save.click(
        save_state,
        inputs=[msg_state, filepath_state],
        outputs=[info],
        )
    
    load.click(
        load_state,
        inputs=[file_explorer, config_state],
        outputs=[chat_history, config_state],
        )

    print_log.click(
        print_logger,
        inputs=[],
        outputs=[log_window],
    )


# Launches App; close with ctrl+c ======================================================
# ===========================================================================================
demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
