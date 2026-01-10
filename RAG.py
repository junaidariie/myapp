from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import os
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


#===========================================
# Load FAISS DB & Reload Logic [FEATURE ADDED]
#===========================================

FAISS_DB_PATH = "vectorstore/db_faiss"
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Global variable for the database
db = None

def reload_vector_store():
    """
    Reloads the FAISS index from disk. 
    Call this function after a new file is ingested.
    """
    global db
    if os.path.exists(FAISS_DB_PATH):
        print(f"Loading FAISS from {FAISS_DB_PATH}...")
        try:
            db = FAISS.load_local(
                FAISS_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            db = None
    else:
        print("Warning: No Vector DB found. Please run ingestion first.")
        db = None 

# Initial Load
reload_vector_store()


#===========================================
# Class Schema
#===========================================

class Ragbot_State(TypedDict):
    query       :   str
    context     :   list[str]
    metadata    :   list[dict]
    RAG         :   bool
    web_search  :   bool
    model_name  :   str
    web_context :   str
    response    :   Annotated[list[BaseMessage], add_messages]

#===========================================
# LLM'S
#===========================================


llm_kimi2       =   ChatGroq(model='moonshotai/kimi-k2-instruct-0905', streaming=True, temperature=0.4)
llm_gpt         =   ChatOpenAI(model='gpt-4.1-nano', streaming=True, temperature=0.2)
llm_gpt_oss     =   ChatGroq(model='openai/gpt-oss-120b', streaming=True, temperature=0.3)
llm_lamma4      =   ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', streaming=True, temperature=0.5)
llm_qwen3       =   ChatGroq(model='qwen/qwen3-32b', streaming=True, temperature=0.5)

def get_llm(model_name: str):
    if model_name == "kimi2":
        return llm_kimi2
    elif model_name == "gpt":
        return llm_gpt
    elif model_name == "gpt_oss":
        return llm_gpt_oss
    elif model_name == "lamma4":
        return llm_lamma4
    elif model_name == "qwen3":
        return llm_qwen3
    else:
        return llm_gpt  

#===========================================
# Search tool
#===========================================

@tool
def tavily_search(query: str) -> dict:
    """
    Perform a real-time web search using Tavily.
    """
    try:
        search = TavilySearchResults(max_results=2)
        results = search.run(query)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}
    
#===========================================
# fetching web context
#===========================================

def fetch_web_context(state: Ragbot_State):
    user_query = state["query"]

    enriched_query = f"""
Fetch the latest, accurate, and up-to-date information about:
{user_query}

Focus on:
- recent news
- official announcements
- verified sources
- factual data
"""

    web_result = tavily_search.run(enriched_query)

    return {
        "web_context": str(web_result)
    }

#===========================================
# db search
#===========================================

@tool
def faiss_search(query: str) -> str:
    """Search the FAISS vectorstore and return relevant documents."""
    # Check global db variable
    if db is None:
        return "No documents have been uploaded yet.", []

    try:
        results = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        metadata = [doc.metadata for doc in results]
        return context, metadata
    except Exception as e:
        return f"Error searching vector store: {str(e)}", []

#===========================================
# router
#===========================================


def router(state: Ragbot_State):
    if state["RAG"]:
        return "fetch_context"

    if state["web_search"]:
        return "fetch_web_context"

    return "chat"

#===========================================
# fetching context
#===========================================

def fetch_context(state: Ragbot_State):
    query = state["query"]
    context, metadata = faiss_search.invoke({"query": query})
    return {"context": [context], "metadata": [metadata]}


#===========================================
# system prompt
#===========================================


SYSTEM_PROMPT = SystemMessage(
    content="""
You are an intelligent conversational assistant and retrieval-augmented AI system built by Junaid.

Your role is to:
- Engage naturally in conversation like a friendly, helpful chatbot.
- Answer general questions using your own knowledge when no external context is provided.
- When relevant context is provided, use it accurately to answer user questions.
- Seamlessly switch between casual conversation and knowledge-based answering.

Guidelines:
- If context is provided and relevant, use it as the primary source of truth.
- If context is not provided or not relevant, respond using your general knowledge.
- Do not hallucinate or invent information.
- If you are unsure or the information is not available, clearly state that.
- Be clear, concise, and helpful in all responses.
- Maintain a natural, human-like conversational tone.
- Never mention internal implementation details such as embeddings, vector databases, or system architecture.

You are designed to provide reliable, accurate, and engaging assistance.
"""
)

#===========================================
# Chat function
#===========================================

def chat(state:Ragbot_State):
    query = state['query']
    context = state['context']
    metadata = state['metadata']
    web_context = state['web_context']
    model_name = state.get('model_name', 'gpt')

    history = state.get("response", [])

    # [CHANGED] Updated Prompt to include History so it remembers your name
    prompt = f"""
You are an expert assistant designed to answer user questions using multiple information sources.

Source Priority Rules (STRICT):
1. **Conversation History**: Check if the answer was provided in previous messages (e.g., user's name, previous topics).
2. If the provided Context contains the answer, use ONLY the Context.
3. If the Context does not contain the answer and Web Context is available, use the Web Context.
4. If neither Context nor Web Context contains the answer, use your general knowledge.
5. Do NOT invent or hallucinate facts.
6. If the answer cannot be determined, clearly say so.

User Question:
{query}

Retrieved Context (Vector Database):
{context}

Metadata:
{metadata}

Web Context (Real-time Search):
{web_context}

Final Answer:
"""

    selected_llm = get_llm(model_name)
    messages = [SYSTEM_PROMPT] + history + [HumanMessage(content=prompt)]
    response = selected_llm.invoke(messages)
    return {
        'response': [
            HumanMessage(content=query), 
            response
        ]
    }

#===========================================
# Graph Declaration
#===========================================

memory = MemorySaver()
graph = StateGraph(Ragbot_State)

graph.add_node("fetch_context", fetch_context)
graph.add_node("fetch_web_context", fetch_web_context)
graph.add_node("chat", chat)

graph.add_conditional_edges(
    START,
    router,
    {
        "fetch_context": "fetch_context",
        "fetch_web_context": "fetch_web_context",
        "chat": "chat"
    }
)

graph.add_edge("fetch_context", "chat")
graph.add_edge("fetch_web_context", "chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=memory)


#===========================================
# Helper Function
#===========================================

def ask_bot(query: str, use_rag: bool = False, use_web: bool = False, thread_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "query": query,
        "RAG": use_rag,
        "web_search": use_web,
        "context": [],
        "metadata": [],
        "web_context": "",
    }
    
    result = app.invoke(inputs, config=config)
    last_message = result['response'][-1]
    
    return last_message.content


"""print("--- Conversation 1 ---")
# User says hello and gives name
response = ask_bot("Hi, my name is Junaid", thread_id="session_A")
print(f"Bot: {response}")

# User asks for name (RAG and Web are OFF)
response = ask_bot("What is my name?", thread_id="session_A")

print(f"Bot: {response}")"""
