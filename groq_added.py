import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

# Initialize LLM (Groq)
llm = ChatGroq(
    groq_api_key="Your_groq_api", 
    model_name="llama-3.3-70b-versatile"
)

# Initialize ConversationSummaryMemory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(llm=llm)

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=True
    )

# 🎨 Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
    }
    .chat-container {
        border-radius: 15px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 800px;
        margin: auto;
        max-height: 400px;
        overflow-y: auto;
    }
    .chat-message {
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 12px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0px 3px 6px rgba(0,0,0,0.15);
    }
    .user-msg {
        background: linear-gradient(to right, #0084ff, #00c6ff);
        color: white;
        text-align: right;
        margin-left: auto;
    }
    .bot-msg {
        background-color: #f1f0f0;
        color: black;
        text-align: left;
        margin-right: auto;
    }
    .title {
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }
    input[type="text"] {
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #ccc;
        width: 100%;
    }
    button[kind="primary"] {
        background-color: #0084ff !important;
        border-radius: 12px !important;
        padding: 8px 20px !important;
        font-weight: bold !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="title">📝 Chatbot with Memory</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">This bot remembers and summarizes past conversations ✨</div>', unsafe_allow_html=True)

# Chat input
user_input = st.text_input("Type your message here...")

if st.button("Send") and user_input:
    response = st.session_state.conversation.predict(input=user_input)
    st.session_state.chat_history = st.session_state.get("chat_history", [])
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history in styled bubbles
if "chat_history" in st.session_state:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f'<div class="chat-message user-msg">🙋 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Debugging: Show summary
if st.checkbox("Show Memory Summary"):
    st.write(st.session_state.memory.buffer)

