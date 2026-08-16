from agent.agent import ask_datapilot


def main():
    question = input("Ask DataPilot: ")

    answer = ask_datapilot(question)

    print("\nDataPilot:")
    print(answer)


if __name__ == "__main__":
    main()