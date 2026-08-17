from pathlib import Path

from agent.agent import DataPilotAgent


def main():

    print("====================================")
    print("       DataPilot Data Agent")
    print("====================================")
    print()

    while True:

        dataset_path = input(
            "Enter CSV file path (or 'exit'): "
        ).strip()

        if dataset_path.lower() == "exit":
            print("Goodbye!")
            return

        if not dataset_path:
            print("Please enter a CSV file path.")
            print()
            continue

        path = Path(dataset_path)

        if not path.exists():
            print(
                f"File not found: {dataset_path}"
            )
            print()
            continue

        if path.suffix.lower() != ".csv":
            print(
                "Please provide a CSV file."
            )
            print()
            continue

        try:

            agent = DataPilotAgent(
                str(path)
            )

            print()
            print("Dataset loaded successfully.")
            print(
                f"Rows: {len(agent.df)}"
            )
            print(
                f"Columns: {len(agent.df.columns)}"
            )
            print(
                f"Column names: "
                f"{list(agent.df.columns)}"
            )
            print()

            print(
                "Ask questions about your dataset."
            )
            print(
                "Type 'new' to load another CSV."
            )
            print(
                "Type 'exit' to quit."
            )
            print()

            while True:

                question = input(
                    "You: "
                ).strip()

                if question.lower() == "exit":
                    print(
                        "DataPilot: Goodbye!"
                    )
                    return

                if question.lower() == "new":
                    print()
                    break

                if not question:
                    continue

                try:

                    answer = agent.ask(
                        question
                    )

                    print()
                    print("DataPilot:")
                    print(answer)
                    print()

                except Exception as error:

                    print()
                    print(
                        "DataPilot Error:"
                    )
                    print(error)
                    print()

        except Exception as error:

            print()
            print(
                "Could not load dataset:"
            )
            print(error)
            print()


if __name__ == "__main__":
    main()