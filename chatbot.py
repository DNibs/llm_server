import os
import gradio as gr
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# Set environment for OpenAI-compatible LM Studio endpoint
os.environ["OPENAI_API_KEY"] = "lm-studio"  # dummy key; LM Studio doesn't require real auth
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"  # LM Studio local server

# Set up LangChain model (ChatOpenAI uses OpenAI-compatible endpoints)
LLM = ChatOpenAI(
    temperature=0.7,
    model_name="arbitrary_value",  # name is arbitrary for local usage
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant. Answer all questions to the best of your ability. "
            "Do not make up facts. If asked for code, do not make up functions, classes, or libraries. "
            "If you are unsure about something, state that you are unsure.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Initiate the graph
workflow = StateGraph(state_schema=MessagesState)

# Define what happens in graph
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = LLM.invoke(prompt)
    return {'messages': response}

# Define the single node in graph
workflow.add_edge(START, 'model')
workflow.add_node('model', call_model)

# Add memory to graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Define the i/o between the graph and gradio
def query_response(chat_history, query, config):
    input_messages = [HumanMessage(query)]  # wraps in langgraph format
    output = app.invoke({"messages": input_messages}, config)  # run graph
    response = output["messages"][-1].content  # extract LLM response
    chat_history.append({"role": "user", "content": query})  # wraps in gradio format
    chat_history.append({"role": "assistant", "content": response})

    # Access metadata from previous message to update chat info
    model_type = output['messages'][-1].response_metadata["model_name"]
    thread_id = config["configurable"]["thread_id"]
    token_usage = output['messages'][-1].response_metadata["token_usage"]
    prompt_tokens = token_usage["prompt_tokens"]
    completion_tokens = token_usage["completion_tokens"]
    total_tokens = token_usage["total_tokens"]
    chat_info = f"Model: {model_type}, Thread ID: {thread_id}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}"

    print(output)  # REMOVE FOR PRODUCTION
    return chat_history, '', chat_info 


# Iterates thread id to clear state memory and returns empty values to gradio
def clear_fn(config):
    config["configurable"]["thread_id"] = str(int(config["configurable"]["thread_id"]) + 1)  # increment thread id
    return [], '', config, f'Thread ID: {config["configurable"]["thread_id"]}'


# Sets unique value for each thread
config_id = {"configurable": {"thread_id": "1"}}

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🤖 Local LLM Chatbot via LangGraph + LM Studio") 
    chat_history = gr.Chatbot(type="messages")
    config_state = gr.State(config_id)
    msg = gr.Textbox(placeholder='Type your message here!')

    with gr.Row():
        send = gr.Button("Send", scale=1)
        clear = gr.Button("Clear History", scale=1)

    info = gr.Markdown('Thread ID: 1')

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


demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
