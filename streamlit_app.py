import tempfile

import pandas as pd
import streamlit as st

from agent.agent import DataPilotAgent


st.set_page_config(
    page_title="DataPilot",
    page_icon="◈",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling — instrument-panel theme
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: radial-gradient(ellipse at top left, #141B20 0%, #0F1417 55%, #0B0F12 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}

    .dp-mark {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 30px;
        color: #E8A94C;
        letter-spacing: -0.5px;
    }
    .dp-tagline {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: #6C7B84;
        margin-bottom: 22px;
    }

    .gauge-row { display: flex; gap: 14px; margin: 18px 0 24px 0; flex-wrap: wrap; }
    .gauge {
        flex: 1;
        min-width: 140px;
        background: #141B20;
        border: 1px solid #22303A;
        border-radius: 4px;
        padding: 14px 16px 12px 16px;
    }
    .gauge-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #6C7B84;
        margin-bottom: 6px;
    }
    .gauge-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 24px;
        font-weight: 600;
        color: #E8ECEE;
        line-height: 1.2;
        word-break: break-word;
    }

    .dp-section {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #6C7B84;
        margin: 18px 0 8px 2px;
    }

    .log-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #E8A94C;
        margin-bottom: 4px;
    }
    .log-body {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        line-height: 1.55;
        color: #E8ECEE;
        border-left: 2px solid #E8A94C;
        padding-left: 12px;
        margin-bottom: 10px;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #2C3D48;
        border-radius: 4px;
        padding: 6px;
        background: #10161A;
    }
    .stTextInput input {
        background: #141B20 !important;
        border: 1px solid #22303A !important;
        color: #E8ECEE !important;
    }
    .stButton button {
        background: #E8A94C;
        color: #10161A;
        border: none;
        font-weight: 600;
        border-radius: 3px;
    }
    .stButton button:hover { background: #F0BA6B; color: #10161A; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="dp-mark">◈ DataPilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dp-tagline">upload a dataset — ask it questions in plain english</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv",
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        dataset_path = temp_file.name

    try:
        agent = DataPilotAgent(dataset_path)

        st.markdown('<div class="dp-section">DATASET LOADED</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="gauge-row">
                <div class="gauge">
                    <div class="gauge-label">ROWS</div>
                    <div class="gauge-value">{len(agent.df):,}</div>
                </div>
                <div class="gauge">
                    <div class="gauge-label">COLUMNS</div>
                    <div class="gauge-value">{len(agent.df.columns)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "**Column names:** "
            + ", ".join(str(column) for column in agent.df.columns)
        )

        st.divider()

        question = st.text_input("Ask DataPilot a question:")

        if st.button("Analyze"):

            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("DataPilot is analyzing your dataset..."):
                    try:
                        answer = agent.ask(question)

                        st.markdown('<div class="dp-section">ANALYSIS LOG</div>', unsafe_allow_html=True)
                        st.markdown('<div class="log-meta">DATAPILOT</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="log-body">{answer}</div>', unsafe_allow_html=True)

                        # -------------------------------------
                        # EVIDENCE / COMPUTATION
                        # -------------------------------------
                        if agent.last_tool_results:

                            st.divider()
                            st.markdown('<div class="dp-section">EVIDENCE / COMPUTATION</div>', unsafe_allow_html=True)

                            for tool_result in agent.last_tool_results:

                                st.caption("Analysis tool: " + tool_result["tool"])
                                result = tool_result["result"]

                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result, use_container_width=True)

                                elif isinstance(result, list):
                                    if result and all(isinstance(item, dict) for item in result):
                                        st.dataframe(pd.DataFrame(result), use_container_width=True)
                                    else:
                                        st.write(result)

                                elif isinstance(result, dict):
                                    st.json(result)

                                else:
                                    st.write(result)

                    except Exception as error:
                        st.error(f"DataPilot Error: {error}")

    except Exception as error:
        st.error(f"Could not load dataset: {error}")