from google import genai
from google.genai import types
from config.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

def get_ai_response(prompt, context, mode="Detailed"):
    # Instruction based on mode
    mode_prompt = "Keep your answer brief and concise (max 3 sentences)." if mode == "Concise" else "Provide a detailed, structured, and comprehensive answer."
    
    full_prompt = f"""
    ROLE: Grounded Research Assistant
    MODE: {mode_prompt}
    CONTEXT FROM PDF: {context if context else "No PDF content available."}
    USER QUESTION: {prompt}
    
    INSTRUCTIONS: 
    1. If the PDF context contains the answer, prioritize it.
    2. If not, use Google Search to find the most up-to-date information.
    3. Always cite your sources.
    """

    # Setup Google Search Tool
    generate_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction="You are a professional assistant with access to PDF context and Live Web Search.",
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )

    return client.models.generate_content_stream(
        model="gemini-2.5-flash-lite", 
        contents=full_prompt,
        config=generate_config,
    )