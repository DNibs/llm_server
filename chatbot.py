import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# Set environment for OpenAI-compatible LM Studio endpoint
os.environ["OPENAI_API_KEY"] = "lm-studio"  # dummy key; LM Studio doesn't require real auth
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"  # LM Studio local server

# Set up LangChain model (ChatOpenAI uses OpenAI-compatible endpoints)
# TODO: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import ChatOpenAI``.
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gemma-3-27b-it",  # name is arbitrary for local usage
)

# Add simple memory
# TODO: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
memory = ConversationBufferMemory()

# Wrap in a ConversationChain
# TODO: LangChainDeprecationWarning: The class `ConversationChain` was deprecated in LangChain 0.2.7 and will be removed in 1.0. Use :meth:`~RunnableWithMessageHistory: https:ce/core/runnabce/core/ce/corce/core/ce/core/ce/corce/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html` instead.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Chat function
def chat_fn(history, message):
    response = conversation.predict(input=message)
    history.append((message, response))
    return history, ""

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🤖 Local LLM Chatbot via LangChain + LM Studio")
    #chatbot = gr.Chatbot(type='messages')
    #TODO: deprecated, update to line above, but currently gives errors 
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type your message...",
            lines=1,
            scale=8
        )
        send = gr.Button("Send", scale=1)

    clear = gr.Button("Clear")

    # Submit with Enter
    msg.submit(chat_fn, [state, msg], [chatbot, msg])

    # Submit with Send button
    send.click(chat_fn, [state, msg], [chatbot, msg])

    # Clear chat
    # TODO: Make this clear chat history. Currently, it clears only visual.
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
