import tempfile

import pandas as pd
import streamlit as st

from agent.agent import DataPilotAgent


st.set_page_config(
    page_title="DataPilot",
    page_icon="📊",
    layout="wide",
)


st.title("📊 DataPilot")
st.subheader("AI Data Analysis Agent")

st.write(
    """
    Upload any CSV dataset and ask questions about it.
    DataPilot analyzes the dataset and explains the results.
    """
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

        temp_file.write(
            uploaded_file.getvalue()
        )

        dataset_path = temp_file.name

    try:

        agent = DataPilotAgent(
            dataset_path
        )

        st.success(
            "Dataset loaded successfully."
        )

        st.write(
            f"**Rows:** {len(agent.df)}"
        )

        st.write(
            f"**Columns:** {len(agent.df.columns)}"
        )

        st.write(
            "**Column names:** "
            + ", ".join(
                str(column)
                for column in agent.df.columns
            )
        )

        st.divider()

        question = st.text_input(
            "Ask DataPilot a question:"
        )

        if st.button("Analyze"):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "DataPilot is analyzing your dataset..."
                ):

                    try:

                        answer = agent.ask(
                            question
                        )

                        st.subheader(
                            "DataPilot"
                        )

                        st.write(
                            answer
                        )

                        # -------------------------------------
                        # EVIDENCE / COMPUTATION
                        # -------------------------------------

                        if agent.last_tool_results:

                            st.divider()

                            st.subheader(
                                "Evidence / Computation"
                            )

                            for tool_result in (
                                agent.last_tool_results
                            ):

                                st.caption(
                                    "Analysis tool: "
                                    + tool_result["tool"]
                                )

                                result = (
                                    tool_result["result"]
                                )

                                if isinstance(
                                    result,
                                    pd.DataFrame,
                                ):

                                    st.dataframe(
                                        result,
                                        use_container_width=True,
                                    )

                                elif isinstance(
                                    result,
                                    list,
                                ):

                                    if result and all(
                                        isinstance(
                                            item,
                                            dict,
                                        )
                                        for item in result
                                    ):

                                        st.dataframe(
                                            pd.DataFrame(
                                                result
                                            ),
                                            use_container_width=True,
                                        )

                                    else:

                                        st.write(
                                            result
                                        )

                                elif isinstance(
                                    result,
                                    dict,
                                ):

                                    st.json(
                                        result
                                    )

                                else:

                                    st.write(
                                        result
                                    )

                    except Exception as error:

                        st.error(
                            f"DataPilot Error: {error}"
                        )

    except Exception as error:

        st.error(
            f"Could not load dataset: {error}"
        )

