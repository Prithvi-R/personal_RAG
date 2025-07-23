import os
import sys
import argparse
import google.generativeai as genai

API_KEY_ENV_VAR = "GOOGLE_API_KEY"

def main():
    """Main function to parse arguments and run the Gemini model."""
    parser = create_parser()
    args = parser.parse_args()

    # --- Configure API Key ---
    api_key = args.api_key or os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        print(f"Error: API key not found.")
        print(f"Please set the '{API_KEY_ENV_VAR}' environment variable or use the --api-key argument.")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error: Failed to configure the Generative AI client: {e}")
        sys.exit(1)

    # --- Handle --list-models flag ---
    if args.list_models:
        list_available_models()
        sys.exit(0)

    # --- Validate prompt ---
    if not args.prompt:
        print("Error: A prompt is required unless you are listing models.")
        parser.print_help()
        sys.exit(1)

    # --- Model Selection and Content Preparation ---
    model_name = args.model
    prompt = args.prompt

    # --- Run the model ---
    print(f"\nRunning model: {model_name}...")
    print("-" * 20)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        print("\n--- Model Response ---")
        print(response.text)
        print("-" * 22)

    except Exception as e:
        print(f"\nAn error occurred while running the model: {e}")
        sys.exit(1)


def create_parser():
    """Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser(
        description="A text-only command-line tool to run Google's Gemini models.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic prompt using the fast Gemini 2.5 Flash model (default)
  python learning.py "Summarize the plot of the book 'Dune' in five sentences."

  # Use the most powerful Gemini 2.5 Pro model for a complex task
  python learning.py --model gemini-2.5-pro "Write a Python script to scrape headlines from a news website."

  # Use a different text model like the original Gemini Pro
  python learning.py --model gemini-pro "Explain the difference between 'affect' and 'effect'."

  # List all available models for your API key
  python learning.py --list-models
"""
    )
    parser.add_argument(
        "prompt",
        nargs='?',  # Make the prompt optional
        default=None,
        help="The text prompt to send to the model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-lite",
        help="The text-based model to use (e.g., 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-pro')."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Your Google AI Studio API key. (It's more secure to use the GOOGLE_API_KEY environment variable)."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all generative models available to your API key and exit."
    )
    return parser


def list_available_models():
    """Lists all models that support content generation."""
    print("Available Gemini models for your API key:")
    try:
        for m in genai.list_models():
            # Check if the model supports the 'generateContent' method (standard for text)
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"An error occurred while fetching models: {e}")


if __name__ == "__main__":
    main()