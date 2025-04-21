import os
import gradio as gr
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_google_community import GoogleSearchAPIWrapper
from langchain.globals import set_verbose, set_debug
from transformers import AutoTokenizer
from typing import List, Any
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

#MODEL_NAME = "meta-llama/Meta-Llama-3-1B-Instruct"  # Local model name
MODEL_NAME = os.getenv('MODEL_NAME') # Local model name
MODEL_TOKENIZER_PATH = os.getenv('MODEL_TOKENIZER_PATH') # Local tokenizer path
SET_PROMPT = os.getenv('SET_PROMPT') # Local prompt template
MAX_TOKENS = int(os.getenv('MAX_TOKENS')) # Local max tokens
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') # Local TAVILY API key
set_verbose(False)
set_debug(True)

# Tokenizer for counting tokens in messages
tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER_PATH)


# Set environment for OpenAI-compatible LM Studio endpoint
os.environ["OPENAI_API_KEY"] = "lm-studio"  # dummy key; LM Studio doesn't require real auth
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"  # LM Studio local server

# Set up LangChain model (ChatOpenAI uses OpenAI-compatible endpoints)
LLM = ChatOpenAI(
    temperature=0.7,
    model_name=MODEL_NAME,  # name is arbitrary for local usage
)


# Set up Tavily search tool for web interactions
#search = TavilySearchResults(max_results=2)
#search = GoogleSearchAPIWrapper()
#tools = [search]
#LLM_WITH_TOOLS = LLM.bind_tools(tools)

# LangGraph prompt template; setting prompt as system message avoids trimming
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SET_PROMPT,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Token counter function compatible with langgraph_core.messages.trim_messages
def custom_token_counter(messages: List[Any]) -> int:
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


# EXPERIMENTAL!!!!!!! =============================================================
@tool
def execute_deez(trap: str):
    """
    Returns a DEEZ NUTZ joke if the assistant gets the user to say 'bofa' or 'suggem', otherwise returns a generic message.
    """
    if trap.lower in ['bofa', 'suggem']:
        return 'DEEZ NUTZZZ!'
    else:
        return 'awwwww shucks'
    
tools = [execute_deez]
tool_node = ToolNode(tools)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END
    
LLM_WITH_TOOLS = LLM.bind_tools(tools)
# Initiate the graph ===============================================================
workflow = StateGraph(state_schema=MessagesState)

# Define what happens in graph
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({'messages': trimmed_messages})
    response = LLM_WITH_TOOLS.invoke(prompt)
    #response = LLM.invoke(prompt)
    return {'messages': response}

# Define the single node in graph
workflow.add_node('model', call_model)
workflow.add_node('tools', tool_node)

workflow.add_edge(START, 'model')
workflow.add_conditional_edges('model', should_continue, ['tools', END])
workflow.add_edge('tools', 'model')

# Add memory to graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Functions gradio wil execute ======================================================
# Define the i/o between the graph and gradio - STREAMING 
def query_response(chat_history, query, config):
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

        yield chat_history, '', f'Thread_ID: {metadata["thread_id"]}; Token Count: {token_count} of {MAX_TOKENS}' 


# Iterates thread id to clear state memory and returns empty values to gradio
def clear_fn(config):
    config["configurable"]["thread_id"] = str(int(config["configurable"]["thread_id"]) + 1) 
    return [], '', config, f'Thread ID: {config["configurable"]["thread_id"]}'


# Sets unique value for each thread
config_id = {"configurable": {"thread_id": "1"}}

# Gradio UI ============================================================================
with gr.Blocks() as demo:
    gr.Markdown("### 🤖 Local LLM Chatbot via LangGraph + LM Studio") 
    chat_history = gr.Chatbot(type="messages")
    config_state = gr.State(config_id)
    msg = gr.Textbox(placeholder='Type your message here!')

    with gr.Row():
        send = gr.Button("Send", scale=1)
        clear = gr.Button("Clear History", scale=1)

    info = gr.Markdown('Thread ID: 1')

    # Works on 'enter' key press
    msg.submit(
        query_response, 
        inputs=[chat_history, msg, config_state], 
        outputs=[chat_history, msg, info],
        )
    
    send.click(
        query_response, 
        inputs=[chat_history, msg, config_state], 
        outputs=[chat_history, msg, info],
        )

    clear.click(
        clear_fn,
        inputs=[config_state],
        outputs=[chat_history, msg, config_state, info],
        )


# Launches App; close with ctrl+c ======================================================
demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
