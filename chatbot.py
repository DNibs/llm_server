import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

# Set environment for OpenAI-compatible LM Studio endpoint
os.environ["OPENAI_API_KEY"] = "lm-studio"  # dummy key; LM Studio doesn't require real auth
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"  # LM Studio local server

# Set up LangChain model (ChatOpenAI uses OpenAI-compatible endpoints)
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gemma-3-27b-it",  # name is arbitrary for local usage
)

# Initiate the graph
workflow = StateGraph(state_schema=MessagesState)

# Define what happens in graph
def call_model(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': response}

# Define the single node in graph
workflow.add_edge(START, 'model')
workflow.add_node('model', call_model)

# Add memory to graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Define the i/o between the graph and gradio
def query_response(query, chat_history, config):
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    response = output["messages"][-1].content
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response})   
    return chat_history, '' # clears input box


def clear_fn(config):
    config["configurable"]["thread_id"] = str(int(config["configurable"]["thread_id"]) + 1)  # increment thread id
    return [], '', config  # return updated state


# Sets unique value for each thread; change to clear history
config_id = {"configurable": {"thread_id": "1"}}

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🤖 Local LLM Chatbot via LangChain + LM Studio") 
    chat_history = gr.Chatbot(type="messages")
    config_state = gr.State(config_id)

    with gr.Row():
        msg = gr.Textbox('Type your message here!')
        send = gr.Button("Send", scale=1)

    clear = gr.Button("Clear History and Starts New Thread", scale=1)

    msg.submit(
        query_response, 
        inputs=[msg, chat_history, config_state], 
        outputs=[chat_history, msg],
        )
    
    send.click(
        query_response, 
        inputs=[msg, chat_history, config_state], 
        outputs=[chat_history, msg],
        )

    clear.click(
        clear_fn,
        inputs=[config_state],
        outputs=[chat_history, msg, config_state],
        )


demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
