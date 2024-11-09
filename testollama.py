import ollama
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

prompt = "Generate a personalized LinkedIn connection message for John Doe, a Software Engineer with skills in Python and Flask."

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "system",
            "content": "You are a professional networking assistant crafting personalized LinkedIn connection messages."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

logger.debug(f"Ollama response: {response}")
