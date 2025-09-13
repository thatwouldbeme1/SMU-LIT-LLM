import streamlit as st
from main import get_strategic_analysis, DEFAULT_USER_QUERY

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Arbitration Co-Counsel",
    page_icon="⚖️"
)

# --- App State ---
if 'analysis' not in st.session_state:
    st.session_state.analysis = ""

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input(
        "Enter your Google API Key", 
        type="password", 
        key="google_api_key",
        help="Your key is stored securely and only for this session."
    )

    st.header("📝 Case Details")
    tone_style = st.selectbox(
        "Select Closing Statement Tone",
        ("Assertive", "Neutral", "Conciliatory"),
        help="Choose the desired tone for the generated closing statement."
    )
    user_prompt = st.text_area(
        "Enter Your Case Strategy and Prompt",
        DEFAULT_USER_QUERY,
        height=300,
        help="Provide the details of your legal scenario and what you need help with."
    )

    if st.button("Generate Strategic Analysis"):
        if not api_key_input:
            st.error("Please enter your Google API Key to proceed.")
        else:
            with st.spinner("Connecting to database and generating analysis... This may take a moment."):
                # The second return value (questions) is now ignored with a `_`
                analysis_output, _ = get_strategic_analysis(user_prompt, tone_style, api_key_input)
                st.session_state.analysis = analysis_output

# --- Main Content Area ---
st.title("⚖️ Arbitration Co-Counsel")
st.caption("An AI-powered strategic partner for dissecting legal counterclaims in international arbitration.")

if st.session_state.analysis:
    st.markdown(st.session_state.analysis)
else:
    st.info("Enter your case details in the sidebar and click 'Generate Strategic Analysis' to begin.")

