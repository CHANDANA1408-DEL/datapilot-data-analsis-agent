import json

from dotenv import load_dotenv
from google import genai

from tools.data_loader import load_csv
from tools.schema import get_dataset_schema
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
You are DataPilot, an AI data analysis agent.

You answer questions about the loaded dataset.

Rules:

1. Never invent numerical results.
2. Use the available analysis tools for calculations.
3. Never perform numerical calculations yourself when a tool can do it.
4. Base numerical answers only on actual tool results.
5. Choose the most appropriate tool for the user's question.
6. Use the dataset information provided in the input to understand
   available columns and their data types.
7. Use the exact dataset column names when calling tools.
8. Clearly explain the result to the user.
9. If the available tools cannot answer the question, say so.
"""


def create_agent_client() -> genai.Client:
    """Create the Gemini client."""

    return genai.Client()


def create_tool_definitions() -> list[dict]:
    """Return the tools available to DataPilot."""

    return [
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
                    }
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
                    }
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
                    }
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
                    }
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

    column_lower = column.strip().lower()

    for actual_column in df.columns:

        if actual_column.lower() == column_lower:
            return actual_column

    raise ValueError(
        f"Column '{column}' does not exist. "
        f"Available columns: {list(df.columns)}"
    )


def execute_tool(
    tool_name: str,
    arguments: dict,
    df,
):
    """
    Execute an approved DataPilot tool locally.

    Gemini chooses the tool, but Python controls
    the actual execution.
    """

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

    if tool_name == "get_top_values":

        column = resolve_column_name(
            arguments["column"],
            df,
        )

        top_n = arguments.get("top_n", 5)

        return get_top_values(
            df,
            column,
            top_n,
        )

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
    """Convert Pandas results into JSON-serializable data."""

    if hasattr(result, "to_dict"):
        return result.to_dict(orient="records")

    return result


def create_dataset_context(df) -> str:
    """
    Create a concise description of the loaded dataset
    for the Gemini agent.
    """

    schema = get_dataset_schema(df)

    return json.dumps(
        schema,
        indent=2,
    )


def ask_datapilot(question: str) -> str:
    """Ask DataPilot a natural-language data question."""

    client = create_agent_client()

    df = load_csv(DATASET_PATH)

    tools = create_tool_definitions()

    dataset_context = create_dataset_context(df)

    agent_input = f"""
Dataset information:

{dataset_context}

User question:

{question}
"""

    interaction = client.interactions.create(
        model=MODEL,
        input=agent_input,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
    )

    while True:

        function_calls = [
            step
            for step in interaction.steps
            if step.type == "function_call"
        ]

        if not function_calls:
            return interaction.output_text

        function_results = []

        for step in function_calls:

            result = execute_tool(
                step.name,
                step.arguments,
                df,
            )

            result = serialize_tool_result(result)

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

        interaction = client.interactions.create(
            model=MODEL,
            previous_interaction_id=interaction.id,
            tools=tools,
            input=function_results,
        )