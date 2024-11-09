from flask import Flask, request, jsonify
from flask_cors import CORS

import ollama
import chromadb
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Create ChromaDB directory
CHROMA_PATH = Path("./chroma_db")
CHROMA_PATH.mkdir(exist_ok=True)

# Initialize ChromaDB with persistent storage
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Load example messages
EXAMPLE_MESSAGES = [
    {
        "context": "Software Engineering and Development",
        "message": "Hi {name}, I noticed your impressive work in software engineering. Your experience with {skills} caught my attention. I'd love to connect and discuss technology trends and potential collaborations.",
    },
    {
        "context": "Data Science and Machine Learning",
        "message": "Hi {name}, your background in data science and work with {skills} is fascinating. I'd love to connect and share insights about AI/ML developments and explore potential synergies.",
    },
    {
        "context": "Product Management",
        "message": "Hi {name}, your experience as a product leader at {company} is impressive. I'd love to connect and exchange ideas about product strategy and innovation.",
    },
    {
        "context": "Marketing and Growth",
        "message": "Hi {name}, your marketing expertise and achievements at {company} are remarkable. I'd love to connect and discuss growth strategies and industry trends.",
    }
]

def initialize_collection():
    try:
        # Try to get existing collection
        collection = client.get_collection("linkedin_messages")
        logger.info("Retrieved existing collection")
    except Exception as e:
        # Create new collection if it doesn't exist
        collection = client.create_collection("linkedin_messages")
        logger.info("Created new collection")
        
        # Add example messages
        for idx, example in enumerate(EXAMPLE_MESSAGES):
            collection.add(
                documents=[example["message"]],
                metadatas=[{"context": example["context"]}],
                ids=[f"example_{idx}"]
            )
        logger.info("Added example messages to collection")
    
    return collection

def verify_ollama():
    try:
        models = ollama.list()
        logger.info(f"Available models: {models}")
        return True
    except Exception as e:
        logger.error(f"Ollama verification failed: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    try:
        collection = initialize_collection()
        ollama_status = verify_ollama()
        return jsonify({
            "status": "healthy",
            "chromadb_count": collection.count(),
            "ollama_status": ollama_status
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/generate_message', methods=['POST'])
def generate_message():
    try:
        data = request.json
        logger.debug(f"Received data: {data}")

        if not data or 'profile_data' not in data:
            logger.error("Missing profile data")
            return jsonify({
                "success": False,
                "error": "Missing profile data"
            }), 400

        profile_data = data['profile_data']
        logger.info(f"Generating message for: {profile_data.get('name')}")

        # Initialize collection
        collection = initialize_collection()

        # Create search context
        search_context = f"""
        Role: {profile_data.get('title', '')}
        Company: {profile_data.get('company', '')}
        Skills: {', '.join(profile_data.get('skills', []))}
        """

        logger.debug(f"Search context: {search_context}")

        # Query for similar messages
        logger.debug("Querying collection with search context")
        results = collection.query(
            query_texts=[search_context],
            n_results=2
        )
        logger.debug(f"Query results: {results}")

        # Extract reference messages
        reference_messages = []
        if 'documents' in results and results['documents']:
            for doc_list in results['documents']:
                reference_messages.extend(doc_list)
        reference_text = '\n'.join(reference_messages)

        # Construct improved prompt for message generation
        prompt = f"""
        Please create a professional and friendly LinkedIn connection request message for:
        Name: {profile_data.get('name')}
        Title: {profile_data.get('title')}
        Company: {profile_data.get('company')}
        Skills: {', '.join(profile_data.get('skills', []))}

        Reference Messages:
        {reference_text}

        Guidelines:
        - Limit the message to 1-2 sentences.
        - Incorporate specific details about their background or skills.
        - Maintain a professional yet conversational tone.
        - Personalize the message based on their profile information.
        - Provide only one cohesive message without numbering or additional introductory text.
        """

        logger.debug("Constructed improved prompt for Ollama:")
        logger.debug(prompt)

        # Generate message using Ollama
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional networking assistant crafting a concise and personalized LinkedIn connection message."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        logger.debug(f"Ollama response: {response}")

        if 'message' in response and 'content' in response['message']:
            generated_message = response['message']['content'].strip()
        else:
            logger.error("Invalid response format from Ollama")
            raise ValueError("Invalid response format from Ollama")

        logger.info("Successfully generated message")
        
        return jsonify({
            "success": True,
            "message": generated_message  # Returns only the message content
        })

    except Exception as e:
        logger.exception("Error generating message")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
