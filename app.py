import streamlit as st
from main import app
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from ingest import embedding_documents
import os

DOCS_FOLDER = "docs"
os.makedirs(DOCS_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="AI Strategist",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stExpander"] div[role="button"] + div p,
    [data-testid="stExpander"] div[role="button"] + div li,
    [data-testid="stExpander"] div[role="button"] + div code {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    st.subheader("Models")
    planner_model = st.selectbox(
        "Planner Model",
        (
            "llama3-70b-8192",  # Groq
            "gemini-1.5-pro-latest", # Google
            "llama-3.1-8b-instant",   
            "gemini-2.5-pro" 
        ),
        index=0,
        help="The more powerful model that creates the plan. Llama 70B or Gemini 1.5 Pro recommended."
    )
    worker_model = st.selectbox(
        "Worker/Synthesizer Model",
        (
            "llama-3.1-8b-instant",  
            "gemini-1.5-flash-latest",
            "llama3-70b-8192",  
            "gemini-2.5-flash"
        ),
        index=0,
        help="The faster model that executes tasks and synthesizes the final answer."
    )

    st.subheader("Tools")
    st.write("Enable or disable tools for the agent:")
    use_docs = st.toggle("Use Document Search", value=True)
    use_graph = st.toggle("Use Graph Search", value=True)
    use_web = st.toggle("Use Web Search", value=True)
    st.divider()

    st.subheader("Manage Documents")

    # 1. Upload files
    uploaded_files = st.file_uploader(
        "Upload documents (multiple allowed)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} new files.")

    # 2. List files currently in docs folder
    st.write("### Current Documents")
    files = os.listdir(DOCS_FOLDER)
    if files:
        files_to_delete = st.multiselect(
            "Select files to delete", 
            options=files
        )
        if st.button("Delete Selected Files"):
            for filename in files_to_delete:
                file_path = os.path.join(DOCS_FOLDER, filename)
                try:
                    os.remove(file_path)
                    st.success(f"Deleted {filename}")
                except Exception as e:
                    st.error(f"Error deleting {filename}: {e}")
    else:
        st.info("No files in docs folder.")

    st.divider()

    # 3. Button to run ingestion on all current files
    if st.button("Embed All Documents"):
        current_files = [os.path.join(DOCS_FOLDER, f) for f in os.listdir(DOCS_FOLDER)]
        if current_files:
            embedding_documents()
        else:
            st.warning("No documents found to embed.")

    if st.button("Embed Documents"):
        embedding_documents()


# --- MAIN PAGE ---
st.title("🧠 AI Research Strategist")
st.caption("A self-correcting agent for hybrid search and analysis.")


# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am your Personal Strategist. I can reason, self-correct, and use multiple tools to answer your questions. How can I help you today?")
    ]

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)


# --- CHAT INPUT ---
if prompt := st.chat_input("Ask a question..."):
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    response_box = st.empty()
    plan_expander = st.empty()
    chunks_expander = st.empty()
    sources_expander = st.empty()

    with st.spinner("Thinking... The agent is planning, executing, and reflecting..."):
        try:
            config = {
                "planner_model": planner_model,
                "worker_model": worker_model
            }
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "config": config
            }
            

            final_state = app.invoke(inputs)
            

            
            final_answer = final_state.get("final_answer", "Sorry, I couldn't generate a response.")
            reference_list = final_state.get("reference_list", "")
            
            full_response_content = final_answer
            if reference_list:
                references_html = reference_list.replace('\n', '<br>')
                full_response_content += f"\n\n**References:**\n{references_html}"

            with response_box.chat_message("ai"):
                st.markdown(full_response_content, unsafe_allow_html=True)
            
            st.session_state.messages.append(AIMessage(content=full_response_content))

            past_plans = final_state.get('past_plans', [])
            final_plan = final_state.get('plan')
            all_data = final_state.get('failed_attempt_data', []) + final_state.get('collected_data', [])

            # Expander 1: Execution Plan
            with plan_expander.expander("✅ **Execution Plan & Reasoning**"):
                if past_plans:
                    for i, plan in enumerate(past_plans):
                        st.markdown(f"#### Attempt {i+1} (Failed)")
                        for step in plan.steps:
                            st.markdown(f"- `{step}`")
                        st.markdown("> _Agent decided to re-plan as results were insufficient._")
                
                if final_plan:
                    st.markdown("#### Final Plan (Successful)")
                    for step in final_plan.steps:
                        st.markdown(f"- `{step}`")

            # Expander 2: Retrieved Chunks & Tool Results
            with chunks_expander.expander("📚 **Retrieved Chunks & Tool Results**"):
                if not all_data:
                    st.write("No data was retrieved during execution.")
                else:
                    for i, data_item in enumerate(all_data):
                        if data_item.get('results'):
                            tool_name = data_item.get('tool_name', 'Unknown Tool')
                            st.markdown(f"#### Results from `{tool_name}`")
                            
                            results = data_item.get('results', [])
                            
                            if tool_name == 'web_search':
                                if isinstance(results, str):
                                    try:
                                        parsed = json.loads(results)
                                        if isinstance(parsed, dict) and 'results' in parsed:
                                            results = parsed['results']
                                        else:
                                            st.warning("Parsed web search data has unexpected structure.")
                                            results = []
                                    except json.JSONDecodeError:
                                        st.warning("Could not parse stringified results.")
                                        results = []
                                elif isinstance(results, dict):
                                            # You're in this case now!
                                    results = results.get("results", [])

                                if isinstance(results, list):
                                    for res_dict in results:
                                        if isinstance(res_dict, dict):  # Safety check
                                            with st.container(border=True):
                                                title = res_dict.get('title', 'No Title')
                                                url = res_dict.get('url', '#')
                                                content = res_dict.get('content', 'No content available.')
                                                score = res_dict.get('score', None)

                                                st.markdown(f"🔗 [{title}]({url})", unsafe_allow_html=True)
                                                st.markdown(content)

                                                if score is not None:
                                                    st.markdown(f"**Relevance Score:** `{score:.4f}`")
                                else:
                                    # st.warning("Web search results is not a list.")
                                    st.warning(f"Web search results is not a list after parsing.\n {results}")

                            elif tool_name == 'document_search' and isinstance(results, list):
                                RELEVANCE_THRESHOLD = 0.3
                                
                                filtered_results = [
                                    doc for doc in results 
                                    if doc.get('metadata', {}).get('relevance_score', 0.0) >= RELEVANCE_THRESHOLD
                                ]
                                
                                if not filtered_results:
                                    st.info(f"No documents met the relevance threshold of {RELEVANCE_THRESHOLD}.")
                                else:
                                    for doc_dict in filtered_results:
                                        with st.container(border=True):
                                            metadata = doc_dict.get('metadata', {})
                                            score = metadata.get('relevance_score', -1.0)
                                            
                                            score_display = f"{score:.4f}" if score != -1.0 else "N/A"
                                            source = metadata.get('source', 'Unknown source')
                                            
                                            st.markdown(f"**Source:** `{source}` | **Relevance Score:** `{score_display}`")
                                            st.markdown(doc_dict.get('page_content'))
                            else:
                                st.text(str(results))


            with sources_expander.expander("📄 **Sources Summary**"):
                RELEVANCE_THRESHOLD = 0.3
                source_files_with_scores = []

                for data_item in all_data:
                    results = data_item.get('results', [])
                    
                    if data_item.get('tool_name') == 'document_search' and isinstance(results, list):
                        for doc_dict in results:
                            metadata = doc_dict.get('metadata', {})
                            score = metadata.get('relevance_score', 0.0)
                            
                            if score >= RELEVANCE_THRESHOLD:
                                source = metadata.get('source')
                                if source:
                                    source_files_with_scores.append((source, score))

                if source_files_with_scores:
                    st.write("The agent consulted the following high-relevance documents:")
                    
                    sorted_sources = sorted(source_files_with_scores, key=lambda x: x[1], reverse=True)
                    for source, score in sorted_sources:
                        st.markdown(f"- `{source}` (Relevance: **{score:.3f}**)")

                else:
                    st.write("No specific document sources met the relevance threshold for this query.")


                if reference_list:
                    st.markdown(reference_list)
                else:
                    st.write("No web sources were cited in the final answer.")

        except Exception as e:
            response_content = f"Sorry, an error occurred: {e}"
            st.error(f"An error occurred: {e}", icon="🚨")
    
    