import chromadb
import google.generativeai as genai
import re

# --- Configuration ---
# Use the same API key and model names as in ingest.py
GOOGLE_API_KEY = "API_KEY"
COLLECTION_NAME = "jus_mundi"
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash-latest" # A powerful and fast model for generation
DEFAULT_USER_QUERY = """
I’m working on a case representing Fenoscadia Limited, a mining company from Ticadia that was operating in Kronos under an 80-year concession to extract lindoro, a rare earth metal. In 2016, Kronos passed a decree that revoked Fenoscadia’s license and terminated the concession agreement, citing environmental concerns. The government had funded a study that suggested lindoro mining contaminated the Rhea River and caused health issues, although the study didn’t conclusively prove this. Kronos is now filing an environmental counterclaim in the ongoing arbitration, seeking at least USD 150 million for environmental damage, health costs, and water purification.

Can you help me analyze how to challenge Kronos’s environmental counterclaim, especially in terms of jurisdiction, admissibility, and merits?
"""

def get_strategic_analysis(query, tone, api_key):
    """Executes the full RAG pipeline and returns the analysis and follow-up questions."""
    
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY":
        return "ERROR: Google API Key not configured.", []
        
    genai.configure(api_key=api_key)

    try:
        # Note: Ensure you have a persistent client if running in a stateless environment
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        return f"ERROR: Could not connect to ChromaDB. Details: {e}", []

    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']

    retrieved_docs = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    
    context_str = "\n\n---\n\n".join(retrieved_docs['documents'][0])
    
    # New, more sophisticated prompt based on user feedback
    master_prompt = f"""
    You are a meticulous and adversarial legal strategist. Your task is to analyze a legal position, generate a core argument, and then ruthlessly stress-test that argument from the opponent's perspective. Your output must be a structured, professional memorandum in Markdown format.

    **USER'S SITUATION:**
    <SITUATION>
    {query}
    </SITUATION>

    **RELEVANT LEGAL PRECEDENTS RETRIEVED FROM JUS MUNDI DATABASE:**
    <LEGAL_PRECEDENTS>
    {context_str}
    </LEGAL_PRECEDENTS>

    **YOUR MANDATORY TASKS:**

    Based **ONLY** on the provided LEGAL PRECEDENTS, produce the following five-part analysis.

    **1. Legal Source Retrieved:**
    Identify and quote the single most impactful sentence or phrase from the retrieved LEGAL PRECEDENTS that forms the basis of your strategic analysis.

    **2. Weakness Detected:**
    Based on the retrieved source and the user's situation, identify the primary weakness in the opponent's (Kronos's) counterclaim. Structure your analysis by answering these questions:
    * Has this type of argument been rejected by tribunals before, according to the precedents?
    * Does the opponent's argument rely on flawed reasoning or unproven assumptions?
    * Does the opponent's argument misread or selectively interpret the law or treaty?

    **3. Closing Statement Generator:**
    Draft a single, powerful paragraph for a closing statement that leverages the weakness you just identified. The tone must be **{tone}**.

    **4. Why It Is Good (Strengths Analysis):**
    Provide a brief, critical analysis of the closing statement you just generated. Answer the following:
    * How does it effectively cite jurisdiction, admissibility, or merits to align with tribunal priorities?
    * How does it reinforce legal principles using reasoning from past cases found in the precedents?
    * How does it maintain a professional and authoritative tone?

    **5. Potential Counter-Arguments (Risk Analysis):**
    Now, adopt the mindset of the opposing counsel. Identify the primary risk or potential counter-argument to the closing statement you drafted. Answer the following:
    * Could the opponent cite stronger, more relevant cases or principles we haven't considered?
    * Is our argument speculative or unsupported by the core facts provided?
    * Does our argument fail to speak the language tribunals expect, or does it oversimplify a complex issue?
    """

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(master_prompt)
    
    # The new prompt format doesn't have separate "follow-up questions"
    # The entire output is now considered the "analysis"
    return response.text, []

