import os
import json 
import re
from dotenv import load_dotenv
from pydantic import ValidationError

# --- Core LangChain/LangGraph Imports ---
from operator import add 
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

# --- LLM and Tool Imports ---
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch

# --- Re-Ranker Imports ---
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


load_dotenv()

# ==============================================================================
#  1. SETUP LLMS, CONNECTIONS & TOOLS
# ==============================================================================
def get_llm(model_name: str):
    """Factory function to get an initialized LLM instance by name."""
    print(f"Model Brain Initializing.... ({model_name})!")
    if "llama" in model_name:
        return ChatGroq(model=model_name, temperature=0)
    elif "gemini" in model_name:
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unknown model provider for model: {model_name}")

# # --- Groq Models ---
# llm_pro = ChatGroq(model="llama3-70b-8192", temperature=0)
# print("Planner Brain Initialized (Groq Llama 3 70B).")
# llm_flash = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
# print("Worker Brain Initialized (Groq Llama 3 8B).")
# # --- Google Gemini Models ---
# gemini_pro = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# print("Planner Brain Initialized (Gemini 2.5 Flash).")
# gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# print("Worker Brain Initialized (Gemini 2.0 Flash).")
# gemini_flash_lite = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
# print("Worker Brain Initialized (Gemini 2.0 Flash Lite).")


reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3) # Top 3 most relevant chunks
print("Reranker Initialized.")

# --- Tool Definitions ---
web_search = TavilySearch(max_results=5, name="web_search")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.load_local(
    folder_path="./db_chroma",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10}) 


@tool
def document_search(query: str) -> list:
    """
    Searches private documents for specific details using a retrieve-then-rerank strategy.
    """
    initial_docs = base_retriever.invoke(query)
    
    if not initial_docs:
        print("--- No initial documents found. ---")
        return []
        
    print(f"--- Reranking {len(initial_docs)} documents... ---")
    
    query_doc_pairs = [[query, doc.page_content] for doc in initial_docs]

    scores = reranker_model.score(query_doc_pairs)
    doc_with_scores = list(zip(initial_docs, scores))
    sorted_docs = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)
    final_docs = []
    top_n = 3
    for doc, score in sorted_docs[:top_n]:
        doc.metadata['relevance_score'] = score
        final_docs.append(doc)
    

    results_as_dict = []
    for doc in final_docs:
        results_as_dict.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": doc.metadata.get('relevance_score', 'N/A')
        })
        
    return results_as_dict


graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))
graph.refresh_schema()

@tool
def graph_search(query: str, asd: dict) -> str:
    """Searches the knowledge graph for relationships."""
    worker_model_name = asd.get("worker_model", "llama-3.1-8b-instant") # Default Model
    print(f"---Graph Search using worker model: {worker_model_name}---")
    
    worker_llm = get_llm(worker_model_name)
    dynamic_graph_qa_chain = GraphCypherQAChain.from_llm(
        cypher_llm=worker_llm, 
        qa_llm=worker_llm, 
        graph=graph, 
        verbose=True, 
        allow_dangerous_requests=True
    )
    return dynamic_graph_qa_chain.invoke({"query": query})["result"]

tools = [web_search, document_search, graph_search]

# ==============================================================================
#  2. THE "MASTER PLANNER" AGENT (Big Brain)
# ==============================================================================
class ExecutionPlan(BaseModel): 
    """A step-by-step plan to answer the user's query."""
    steps: List[str] = Field(description="A list of tool calls to execute, in order.")


class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: ExecutionPlan
    past_plans: Annotated[list, add]
    collected_data: list
    failed_attempt_data: Annotated[list, add] 
    step: int
    reflection: Optional[str]
    config: dict
    final_answer: Optional[str]
    reference_list: Optional[str]

tool_definitions = ""
for tool_instance in tools:
    tool_definitions += f"- `{tool_instance.name}(query: str)`: {tool_instance.description}\n"

INITIAL_PLAN_PROMPT  = f"""
You are an expert planner for an AI agent. Your task is to create a step-by-step plan of tool calls to answer a user's query.
Prioritize using the tools available to you, and ensure each step is clear and actionable.

You must follow a strict prioritization strategy for using your tools.

**Tool Prioritization Strategy:**
1.  **Primary Tools (Internal Knowledge):** Always start by using `document_search` and `graph_search`. These tools query a private, trusted knowledge base. They are fast and their information is highly relevant.
2.  **Secondary Tool (External Knowledge):** Use `web_search` as a last resort under the following conditions:
    - The user explicitly asks for recent news, events, or real-time data (e.g., "latest updates," "current stock price").
    - The query is very general and is unlikely to be in the private documents.
    - The primary tools have been tried and failed to yield a useful result.

**Available Tools:**
{tool_definitions}


Based on the user's query and the information above, create a plan.
**Your response MUST be a single, valid JSON object and nothing else.** 
Do not add any explanatory text, acknowledgements, or any other words outside of the JSON structure.
The JSON object must adhere to the Output schema.


**Output Schema Instructions:**

You must provide your response as a single, valid JSON object. Do not add any explanatory text or any words outside of the JSON structure.
The JSON object must have a single key, "steps", which contains a list of strings. Each string must be a valid tool call.

**Example of a perfect response:**
{
  [
    "document_search(query='who founded NVIDIA')",
    "web_search(query='current stock price of NVIDIA')",
    "document_search(query='who founded Apple')",
    "web_search(query='current stock price of Apple')",
    "web_search(query='NVIDIA AI strategy from internal memos')",
    "web_search(query='Apple AI strategy from internal memos')"
  ]
}
The 'steps' should be a list of strings, where each string is a clear tool call.

**User's Query:** {{query}}
"""

REPLAN_PROMPT = """
You are an expert planner for an AI agent, and you are correcting a previous mistake. (Your last plan failed)
Your last plan failed because the tools you chose returned irrelevant or no information. You must create a new plan to find the correct answer.

**1. Context of Failure:**
- **Original User Query:** {query}
- **Your Previous Failed Plan:** {past_plan}

**Critique:**
The previous plan was ineffective. It did not yield the required information.

**Your New Task:**
Analyze the original query and your failed plan. Create a new, different plan that avoids the previous mistakes.
Given the failure of the last plan, you **must prioritize using `web_search`**. Do not repeat the failed tool calls if they are not relevant.

**Available Tools:**
{tool_definitions}

**Output Requirements (MANDATORY):**
Your output MUST be a single, valid JSON object.
- The JSON object must have one key: "steps".
- The value of "steps" must be a list of strings.
- Each string in the list must be a valid tool call.
- **Do not add any other text, explanations, or words outside of the JSON object.**

"""


def planner_node(state: PlannerState):
    config = state.get("config", {})
    planner_model_name = config.get("planner_model", "llama3-70b-8192") # Default Model
    print(f"---PLANNER: Using model {planner_model_name}---")

    is_replan = bool(state.get("past_plans"))
    past_plans = state.get("past_plans", [])
    tool_definitions = "\n".join([f"- `{t.name}(query: str)`: {t.description}" for t in tools])
    query = state["messages"][0].content
    past_plan_str = ""
    prompt_template = ""

    if is_replan:
        print("---PLANNER: RE-PLANNING with memory of past failure...---")
        prompt_template = REPLAN_PROMPT
        last_failed_plan = past_plans[-1]
        past_plan_str = "\n".join(f"- {step}" for step in last_failed_plan.steps)
    else:
        print("---PLANNER: Creating plan...---")
        prompt_template = INITIAL_PLAN_PROMPT

    planner_llm = get_llm(planner_model_name)
    # planner_llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    planner_prompt = ChatPromptTemplate.from_template(prompt_template)
    planner_chain = planner_prompt | planner_llm

    response = planner_chain.invoke({
        "query": query,
        "tool_definitions": tool_definitions,
        "past_plan": past_plan_str
    })
    
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in the response.", response.content, 0)
            
        json_string = json_match.group(0)
        plan_object = ExecutionPlan.parse_raw(json_string)

    except (ValidationError, json.JSONDecodeError) as e:
        print(f"---PLANNER: FAILED to parse LLM response into JSON. Error: {e} ---")
        print(f"---PLANNER: Executing safe fallback plan.---")
        plan_object = ExecutionPlan(steps=[f"web_search(query='{query}')"])
    
    print(f"---PLANNER: Generated Plan ---\n{plan_object.steps}")
    
    return {"plan": plan_object, "step": 1, "reflection": None, "config": config}

reflection_prompt_template = """
You are a critic. Your task is to evaluate if the collected data is sufficient and relevant to answer the original user query.

Original Query: {query}
Collected Data:
---
{collected_data}
---

Based on the data, can you confidently answer the query? 
Respond with a single word: 'proceed' if the data is sufficient, or 'replan' if the data is irrelevant, incomplete, or insufficient.
"""
reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)


def reflection_node(state: PlannerState):
    query = state["messages"][0].content 
    current_data = state.get("collected_data", [])
    verdict = "proceed" 

    if not current_data or all(not item.get('results') for item in current_data):
        verdict = "replan"
    else:
        critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0) # fast and efficient
        reflection_chain = reflection_prompt | critic_llm
        
        data_str = json.dumps(current_data, indent=2, default=str)
        
        decision_response = reflection_chain.invoke({"query": query, "collected_data": data_str})
        verdict = decision_response.content.strip().lower()

    if verdict == "replan":
        print("---CRITIC: Verdict is to 'replan'. Archiving failed data.---")
        failed_plan = state.get("plan")
        past_plans = state.get("past_plans", [])
        failed_data = state.get("failed_attempt_data", [])
        
        return {
            "reflection": "replan",
            "past_plans": past_plans + [failed_plan],
            "failed_attempt_data": failed_data + current_data,
            "collected_data": [] 
        }
    else:
        # print(f"---CRITIC: Verdict is to 'proceed'---")
        return {"reflection": "proceed"}

def reflection_router(state: PlannerState):
    """The central router that directs the agent's next action."""

    if state.get("reflection") == "replan":
        print("---ROUTER: Re-planning required. Looping back to planner.---")
        return "planner"
    
    if state["step"] > len(state["plan"].steps):
        print("---ROUTER: Plan complete. Proceeding to Synthesizer.---")
        return "synthesize"
    
    else:
        print(f"---ROUTER: Plan not complete. Proceeding to next step ({state['step']}).---")
        return "execute_tool"



def tool_executor_node(state: PlannerState):
    """
    A node that executes the tools called in the plan.
    This version correctly invokes the @tool-decorated functions.
    """
    plan = state["plan"]
    step = state["step"]
    config = state.get("config", {})
    # print(config)
    
    if step > len(plan.steps):
        return {}
    
    tool_call_str = plan.steps[step - 1]
    print(f"---EXECUTOR (Step {step}): Running `{tool_call_str}`---")

    try:
        tool_name = tool_call_str.split("(")[0]
        query_match = re.search(r"query=['\"](.*?)['\"]", tool_call_str)
        if not query_match:
            raise ValueError("Could not parse query from tool call.")
        query = query_match.group(1)

    except (IndexError, AttributeError, ValueError) as e:
        error_msg = f"Error parsing tool call: '{tool_call_str}'. Details: {e}"
        print(f"   ERROR: {error_msg}")
        tool_output = {
            "tool_name": "parser_error",
            "tool_query": tool_call_str,
            "results": error_msg
        }
        collected_data = state.get("collected_data", [])
        collected_data.append(tool_output)
        return {"collected_data": collected_data, "step": step + 1}

    tool_to_use = None
    for t in tools: 
        if t.name == tool_name:
            tool_to_use = t
            break

    if not tool_to_use:
        result = f"Error: Tool '{tool_name}' not found."
    else:
        if tool_name == "graph_search":
            # print(type(config))
            # print(graph_search.args_schema.schema())
            result = graph_search.invoke({"query": query, "asd": config})
        else:
            result = tool_to_use.invoke(query)
        


    tool_output = {
        "tool_name": tool_name,
        "tool_query": query,
        "results": result  
    }
    
    collected_data = state.get("collected_data", [])
    collected_data.append(tool_output)
    
    return {"collected_data": collected_data, "step": step + 1}

synthesizer_prompt = ChatPromptTemplate.from_template(
    """Synthesize a comprehensive answer using the user's query and the collected data. 
        It should be to the point with all the sources explitily mentioned.
        It should not be repetetive and should be concise.
        It should be well structured and formatted with each section clearly defined.
        The answer should be strictly related to the user's query and should not include any irrelevant information the details should be mentioned in reference list.
        The relevancy percentage of data should be higher than 65% to be considered it should be at the very end of the answer recieved.
    Dont hallucinate or add any extra information.
    The refrence list should only include the URLs found in the final answer.
    Query: {query}
    Data: {collected_data}"""
)


# synthesizer_chain = synthesizer_prompt | llm_flash
def synthesizer_node(state: PlannerState):
    config = state.get("config", {})
    worker_model_name = config.get("worker_model", "llama-3.1-8b-instant")
    print(f"---SYNTHESIZER: Using model {worker_model_name}  Generating final answer...---")

    worker_llm = get_llm(worker_model_name)
    dynamic_synthesizer_chain = synthesizer_prompt | worker_llm

    data_for_prompt = []
    for item in state.get("collected_data", []):
        results = item.get('results')
        if not results: continue

        tool_name = item.get('tool_name')
        
        if tool_name == 'document_search' and isinstance(results, list):
            doc_strings = [d['page_content'] for d in results]
            data_for_prompt.append(f"Tool: {tool_name}\nResults:\n" + "\n\n".join(doc_strings))
        else:
            data_for_prompt.append(f"Tool: {tool_name}\nResults:\n" + str(results))
            
    collected_data_str = "\n---\n".join(data_for_prompt)
    
    query = state["messages"][0].content
    final_answer = dynamic_synthesizer_chain.invoke({"query": query, "collected_data": collected_data_str})
    return {"messages": [("ai", final_answer.content)]}


def post_processing_node(state: PlannerState):
    """
    Extracts only real URLs from final answer (avoids hallucinated ones).
    Replaces them with [1], [2], etc., and builds a clean reference list.
    """
    print("---POST-PROCESSOR: Formatting citations and sources...---")
    
    raw_answer = state["messages"][-1].content

    url_pattern = r'https?://[^\s\]]+(?:\([^\s\]]*\)[^\s\]]*)*'
    found_urls = re.findall(url_pattern, raw_answer)

    allowed_urls = set()
    for item in state.get("collected_data", []) + state.get("failed_attempt_data", []):
        results = item.get("results", [])
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    url = result.get("url")
                    if url:
                        allowed_urls.add(url)

    real_urls = [url for url in found_urls if url in allowed_urls]

    if not real_urls:
        return {"final_answer": raw_answer, "reference_list": ""}

    seen = set()
    unique_urls = []
    for url in real_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    url_to_citation = {url: f"[{i+1}]" for i, url in enumerate(unique_urls)}

    def replace_url(match):
        url = match.group(0)
        return url_to_citation.get(url, url)

    formatted_answer = re.sub(url_pattern, replace_url, raw_answer)

    reference_list = "\n".join(f"[{i+1}] {url}" for i, url in enumerate(unique_urls))

    return {
        "final_answer": formatted_answer,
        "reference_list": reference_list
    }

def router(state: PlannerState):
    if not state.get("plan") or state["step"] > len(state["plan"].steps): return "synthesize"
    else: return "execute_tool"



# ==============================================================================
#  3. DEFINE AND COMPILE THE WORKFLOW
# ==============================================================================
workflow = StateGraph(PlannerState)
workflow.add_node("planner", planner_node)
workflow.add_node("execute_tool", tool_executor_node)
workflow.add_node("reflection", reflection_node) 
workflow.add_node("synthesize", synthesizer_node)
workflow.add_node("post_processor", post_processing_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "execute_tool")
workflow.add_edge("execute_tool", "reflection")

workflow.add_conditional_edges(
    "reflection",
    reflection_router,
    {
        "planner": "planner",             
        "synthesize": "synthesize",        
        "execute_tool": "execute_tool"     
    }
)
workflow.add_edge("synthesize", "post_processor")

workflow.set_finish_point("post_processor")
app = workflow.compile()
print("\n--- Master Agent compiled. Ready to run. ---\n")

# ==============================================================================
#  4. RUNNING TEST CASES
# ==============================================================================
def run_test_case(name, query):
    print(f"\n\n--- RUNNING TEST CASE: {name} ---")
    config = {
        "planner_model": "gemini-2.5-flash",
        "worker_model": "llama-3.1-8b-instant"
    }
    inputs = {"messages": [("user", query)],"config": config }
    try:
        final_state = app.invoke(inputs)
        print("\n--- FINAL ANSWER ---")
        print(final_state["messages"][-1].content)
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED DURING THE RUN ---")
        print(e)
    finally:
        print("-" * 50)
        print("--- TEST CASE COMPLETE ---\n")

# ==============================================================================
#  5. TEST CASES
# ==============================================================================
# run_test_case(
#     "Complex Multi-Tool Query",
#     "Compare Meta with its competitor, Google, and NVIDIA. Who founded each, and what recent news is there about them and what are their vision in field of AI?"
# )
# run_test_case("Time Search", "What is the time in India?")
# run_test_case("Stock Price Search", "What is the current stock price of NVIDIA (NVDA) and APPLE?")
# run_test_case("Matrix-Graph","How does the transpose operation affect the rows and columns of a matrix?")
# run_test_case("Matrix-docs","What is the significance of the identity matrix?")
# run_test_case("Verilog-Graph","How do tasks and functions improve code reusability in Verilog?")
# run_test_case("Verilog-docs","What is the difference between wire and reg in Verilog?")
# run_test_case("Verilog-docs","What is wire in verilog? what is square matrix")
# In main.py, at the very end
# run_test_case(
#     "Chained Document-to-Graph Query",
#     "Based on my documents, find information about the 'Verilog' language, then use the knowledge graph to find how it improves code reusability."
# )