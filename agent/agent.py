import json

from dotenv import load_dotenv
from google import genai

from tools.data_loader import load_csv
from tools.schema import get_dataset_schema
from tools.data_analysis_engine import analyze_dataset

from tools.question_tools import (
    get_total,
    get_average,
    get_minimum,
    get_maximum,
    get_grouped_sum,
    get_filtered_data,
    get_top_values,
    get_quarterly_sum,
)


load_dotenv()


MODEL = "gemini-3.6-flash"

DATASET_PATH = "data/sales_data.csv"


SYSTEM_INSTRUCTION = """
You are DataPilot, a general-purpose AI data analysis agent.

You analyze arbitrary CSV datasets.

The dataset may contain information about:
- sales
- traffic
- stocks
- finance
- healthcare
- employees
- weather
- education
- manufacturing
- or any other tabular domain.

Your job is to understand the dataset schema, understand the
user's question, select the appropriate analysis tool, and
provide a clear answer.

Rules:

1. Never invent numerical results.
2. Use the available analysis tools for calculations.
3. Never perform numerical calculations yourself when a tool
   can perform them.
4. Base numerical answers only on actual tool results.
5. Use the dataset information provided in the input.
6. Use exact dataset column names when calling tools.
7. Choose the most appropriate analysis operation.
8. Use analyze_dataset whenever it can answer the question.
9. Clearly explain the result to the user.
10. If the available tools cannot answer the question, say so.
"""


def create_agent_client() -> genai.Client:
    """Create the Gemini client."""

    return genai.Client()


def create_tool_definitions() -> list[dict]:
    """
    Return the tools available to DataPilot.
    """

    return [

        # ---------------------------------------------------------
        # GENERIC ANALYSIS TOOL
        # ---------------------------------------------------------

        {
            "type": "function",
            "name": "analyze_dataset",
            "description": """
Perform deterministic analysis on the loaded CSV dataset.

Use this tool whenever the user asks for:
- totals
- averages
- minimums
- maximums
- medians
- counts
- standard deviation
- grouping
- filtering
- filtered aggregation
- ranking
- top N analysis

Supported operations:

1. aggregate
2. group
3. filter
4. filter_aggregate
5. rank

The result returned by this tool is the authoritative
dataset result. Never invent numerical values.
""",
            "parameters": {
                "type": "object",
                "properties": {

                    "operation": {
                        "type": "string",
                        "enum": [
                            "aggregate",
                            "group",
                            "filter",
                            "filter_aggregate",
                            "rank",
                        ],
                        "description": (
                            "The analysis operation to perform."
                        ),
                    },

                    "column": {
                        "type": "string",
                        "description": (
                            "Numeric column to analyze."
                        ),
                    },

                    "group_by": {
                        "type": "string",
                        "description": (
                            "Column used to group the dataset."
                        ),
                    },

                    "aggregation": {
                        "type": "string",
                        "enum": [
                            "sum",
                            "mean",
                            "min",
                            "max",
                            "median",
                            "count",
                            "std",
                        ],
                        "description": (
                            "Aggregation operation."
                        ),
                    },

                    "filters": {
                        "type": "object",
                        "description": (
                            "Column-value filters."
                        ),
                        "additionalProperties": True,
                    },

                    "top_n": {
                        "type": "integer",
                        "description": (
                            "Number of top rows to return."
                        ),
                    },
                },

                "required": [
                    "operation",
                ],
            },
        },

        # ---------------------------------------------------------
        # EXISTING TOOLS
        # ---------------------------------------------------------

        {
            "type": "function",
            "name": "get_total",
            "description": (
                "Calculate the total of a numeric column "
                "in the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "The numeric column whose values "
                            "should be summed."
                        ),
                    },
                },
                "required": ["column"],
            },
        },

        {
            "type": "function",
            "name": "get_average",
            "description": (
                "Calculate the average of a numeric column "
                "in the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "The numeric column whose average "
                            "should be calculated."
                        ),
                    },
                },
                "required": ["column"],
            },
        },

        {
            "type": "function",
            "name": "get_minimum",
            "description": (
                "Find the minimum value of a numeric column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "The numeric column to inspect."
                        ),
                    },
                },
                "required": ["column"],
            },
        },

        {
            "type": "function",
            "name": "get_maximum",
            "description": (
                "Find the maximum value of a numeric column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "The numeric column to inspect."
                        ),
                    },
                },
                "required": ["column"],
            },
        },

        {
            "type": "function",
            "name": "get_grouped_sum",
            "description": (
                "Group the dataset by one column and calculate "
                "the sum of another numeric column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_column": {
                        "type": "string",
                        "description": (
                            "Column used to group the data."
                        ),
                    },
                    "value_column": {
                        "type": "string",
                        "description": (
                            "Numeric column whose values "
                            "should be summed."
                        ),
                    },
                },
                "required": [
                    "group_column",
                    "value_column",
                ],
            },
        },

        {
            "type": "function",
            "name": "get_filtered_data",
            "description": (
                "Filter dataset rows where a column equals "
                "a specified value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Column used for filtering."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Value that the column should equal."
                        ),
                    },
                },
                "required": [
                    "column",
                    "value",
                ],
            },
        },

        {
            "type": "function",
            "name": "get_top_values",
            "description": (
                "Return the top rows ranked by a numeric column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Numeric column used for ranking."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": (
                            "Number of top rows to return."
                        ),
                    },
                },
                "required": ["column"],
            },
        },

        {
            "type": "function",
            "name": "get_quarterly_sum",
            "description": (
                "Calculate the sum of a numeric column "
                "for each quarter based on a date column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": (
                            "Column containing dates."
                        ),
                    },
                    "value_column": {
                        "type": "string",
                        "description": (
                            "Numeric column to aggregate."
                        ),
                    },
                },
                "required": [
                    "date_column",
                    "value_column",
                ],
            },
        },
    ]


def resolve_column_name(
    column: str,
    df,
) -> str:
    """
    Resolve a Gemini-provided column name against
    the actual dataset columns.

    Matching is case-insensitive.
    """

    if not column:
        raise ValueError("Column name cannot be empty.")

    column_lower = column.strip().lower()

    for actual_column in df.columns:

        if actual_column.lower() == column_lower:
            return actual_column

    raise ValueError(
        f"Column '{column}' does not exist. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_filters(
    filters: dict | None,
    df,
) -> dict | None:
    """
    Resolve filter column names case-insensitively.
    """

    if filters is None:
        return None

    resolved = {}

    for column, value in filters.items():

        actual_column = resolve_column_name(
            column,
            df,
        )

        resolved[actual_column] = value

    return resolved


def execute_tool(
    tool_name: str,
    arguments: dict,
    df,
):
    """
    Execute an approved DataPilot tool locally.

    Gemini chooses the tool and parameters,
    but Python controls the actual execution.
    """

    # ---------------------------------------------------------
    # GENERIC ANALYSIS TOOL
    # ---------------------------------------------------------

    if tool_name == "analyze_dataset":

        operation = arguments["operation"]

        column = arguments.get("column")

        if column:
            column = resolve_column_name(
                column,
                df,
            )

        group_by = arguments.get("group_by")

        if group_by:
            group_by = resolve_column_name(
                group_by,
                df,
            )

        filters = resolve_filters(
            arguments.get("filters"),
            df,
        )

        return analyze_dataset(
            df=df,
            operation=operation,
            column=column,
            group_by=group_by,
            aggregation=arguments.get(
                "aggregation",
                "sum",
            ),
            filters=filters,
            top_n=arguments.get("top_n"),
        )

    # ---------------------------------------------------------
    # EXISTING SIMPLE TOOLS
    # ---------------------------------------------------------

    tool_functions = {
        "get_total": get_total,
        "get_average": get_average,
        "get_minimum": get_minimum,
        "get_maximum": get_maximum,
    }

    if tool_name in tool_functions:

        function = tool_functions[tool_name]

        column = resolve_column_name(
            arguments["column"],
            df,
        )

        return function(
            df,
            column,
        )

    # ---------------------------------------------------------
    # GROUPED SUM
    # ---------------------------------------------------------

    if tool_name == "get_grouped_sum":

        group_column = resolve_column_name(
            arguments["group_column"],
            df,
        )

        value_column = resolve_column_name(
            arguments["value_column"],
            df,
        )

        return get_grouped_sum(
            df,
            group_column,
            value_column,
        )

    # ---------------------------------------------------------
    # FILTER
    # ---------------------------------------------------------

    if tool_name == "get_filtered_data":

        column = resolve_column_name(
            arguments["column"],
            df,
        )

        return get_filtered_data(
            df,
            column,
            arguments["value"],
        )

    # ---------------------------------------------------------
    # TOP VALUES
    # ---------------------------------------------------------

    if tool_name == "get_top_values":

        column = resolve_column_name(
            arguments["column"],
            df,
        )

        top_n = arguments.get(
            "top_n",
            5,
        )

        return get_top_values(
            df,
            column,
            top_n,
        )

    # ---------------------------------------------------------
    # QUARTERLY SUM
    # ---------------------------------------------------------

    if tool_name == "get_quarterly_sum":

        date_column = resolve_column_name(
            arguments["date_column"],
            df,
        )

        value_column = resolve_column_name(
            arguments["value_column"],
            df,
        )

        return get_quarterly_sum(
            df,
            date_column,
            value_column,
        )

    raise ValueError(
        f"Unknown tool requested by Gemini: {tool_name}"
    )


def serialize_tool_result(result):
    """
    Convert Pandas results into JSON-serializable data.
    """

    if hasattr(result, "to_dict"):
        return result.to_dict(
            orient="records"
        )

    return result


def create_dataset_context(df) -> str:
    """
    Create a concise description of the loaded dataset
    for Gemini.
    """

    schema = get_dataset_schema(df)

    return json.dumps(
        schema,
        indent=2,
    )


class DataPilotAgent:
    """
    Stateful DataPilot agent.

    Maintains conversation context across questions
    during the same application session.
    """

    def __init__(self):

        self.client = create_agent_client()

        self.df = load_csv(
            DATASET_PATH
        )

        self.tools = create_tool_definitions()

        self.previous_interaction_id = None

    def ask(
        self,
        question: str,
    ) -> str:
        """
        Ask DataPilot a question while maintaining
        conversation context.
        """

        dataset_context = create_dataset_context(
            self.df
        )

        agent_input = f"""
Dataset information:

{dataset_context}

User question:

{question}
"""

        # -----------------------------------------------------
        # FIRST INTERACTION
        # -----------------------------------------------------

        if self.previous_interaction_id is None:

            interaction = self.client.interactions.create(
                model=MODEL,
                input=agent_input,
                system_instruction=SYSTEM_INSTRUCTION,
                tools=self.tools,
            )

        # -----------------------------------------------------
        # CONTINUING CONVERSATION
        # -----------------------------------------------------

        else:

            interaction = self.client.interactions.create(
                model=MODEL,
                previous_interaction_id=(
                    self.previous_interaction_id
                ),
                input=agent_input,
                tools=self.tools,
            )

        # -----------------------------------------------------
        # TOOL-CALLING LOOP
        # -----------------------------------------------------

        while True:

            function_calls = [
                step
                for step in interaction.steps
                if step.type == "function_call"
            ]

            # -------------------------------------------------
            # NO MORE TOOL CALLS
            # -------------------------------------------------

            if not function_calls:

                self.previous_interaction_id = (
                    interaction.id
                )

                return interaction.output_text

            function_results = []

            # -------------------------------------------------
            # EXECUTE EACH TOOL CALL
            # -------------------------------------------------

            for step in function_calls:

                result = execute_tool(
                    step.name,
                    step.arguments,
                    self.df,
                )

                result = serialize_tool_result(
                    result
                )

                print(
                    f"\n[Tool] {step.name}"
                    f"({step.arguments})"
                )

                print(
                    f"[Result] {result}"
                )

                function_results.append(
                    {
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    result,
                                    default=str,
                                ),
                            }
                        ],
                    }
                )

            # -------------------------------------------------
            # SEND TOOL RESULTS BACK TO GEMINI
            # -------------------------------------------------

            interaction = self.client.interactions.create(
                model=MODEL,
                previous_interaction_id=interaction.id,
                tools=self.tools,
                input=function_results,
            )


def ask_datapilot(
    question: str,
) -> str:
    """
    Backward-compatible helper.

    Creates a new DataPilot session for one question.
    """

    agent = DataPilotAgent()

    return agent.ask(question)