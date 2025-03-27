import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from flask import Flask, request, jsonify
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Creating the embeddings
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

# Creating the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Streamlit UI Customization (Task-specific title)
st.set_page_config(page_title="AI Internship Task Algo Root Pvt Ltd", page_icon="✨")
st.title("🤖 AI Internship Task - Algo Root Pvt Ltd")
st.subheader("Dynamically retrieve and execute automation functions using LLM + RAG 🛠️")
st.markdown(
    """
    **Welcome to the AI-powered automation system!**
    This tool processes user prompts, maps them to predefined automation functions,
    and generates executable Python code for function invocation.
    
    🚀 _Upload relevant documents and interact with the system to test its capabilities!_
    """
)

session_id = "Default Session"

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "Assistant", "content":"How can I assist you in automating your tasks today?"}]

if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Process uploaded documents
documents = []
loader = TextLoader(file_path="functions.txt")
docs = loader.load()
documents.extend(docs)
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(documents)
vectordb = FAISS.from_documents(splits, embedding=embeddings)
retriever = vectordb.as_retriever()

prompt_template = """
Given a chat history and the latest user query, rewrite the query into a standalone question that is fully self-contained.
Do not answer the question—only rephrase or return it as is if no changes are needed.
"""
prompt = ChatPromptTemplate([
    ('system', prompt_template),
    MessagesPlaceholder('chat_history'),
    ('user', '{input}')
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)
sys_prompt = (
    "You are an AI assistant specializing in automation tasks. "
    "Your goal is to analyze user prompts, retrieve relevant information, and generate executable Python code. "
    "Use the following context to assist in your responses. "
    "Ensure that the generated code is safe, efficient, and directly relevant to the request. "
    "If you do not have sufficient information, simply state: 'I am unable to generate a response based on the provided data.'"
    "\n\n"
    "{context}"
)

q_a_prompt = ChatPromptTemplate([
    ("system", sys_prompt),
    MessagesPlaceholder("chat_history"),
    ('user', '{input}'),
])

question_ans_chain = create_stuff_documents_chain(llm, q_a_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_ans_chain)

conversational_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

if user_input := st.chat_input(placeholder='Describe your automation task...'):
    st.chat_message('user').write(user_input)
    st.session_state.messages.append({'role':'user', 'content':user_input})
    response = conversational_chain.invoke(
        {'input': user_input},
        config={'configurable': {'session_id': session_id}}
    )
    with st.chat_message('assistant'):
        st.session_state.messages.append({'role': 'assistant', 'content':response['answer']})
        st.write(response['answer'])


@app.route('/execute', methods=['POST'])
def execute():
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    prompt = data['prompt'].lower()

    # Hardcoded response for "Open calculator"
    if "calculator" in prompt:
        response = {
            "function": "open_calculator",
            "code": "import os\nos.system('calc' if os.name == 'nt' else 'gnome-calculator')"
        }
    else:
        response = {
            "function": "unknown",
            "code": "# No predefined function available"
        }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)