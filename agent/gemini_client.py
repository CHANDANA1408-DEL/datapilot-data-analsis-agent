import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


def create_gemini_client() -> genai.Client:
    """
    Create and return a Gemini API client.

    The API key is loaded from the GEMINI_API_KEY
    environment variable.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(
        api_key=api_key
    )