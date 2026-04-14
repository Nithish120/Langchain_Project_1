import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def main():
    print("Hello from langchain-course!")
    print(GEMINI_API_KEY)


if __name__ == "__main__":
    main()
