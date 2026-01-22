# web_app.py

import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.serving import make_server
import threading
from conversation_logger import ConversationLogger

# Global variable, used to receive RAG engine instance from the main app
rag_engine_instance = None
logger = logging.getLogger(__name__)

# Initialize conversation logger
conversation_logger = ConversationLogger()

app = Flask(__name__)


# Web Home Page
@app.route('/')
def index():
    """Render the main Q&A page"""
    return render_template('index.html')


# API endpoint for handling queries
@app.route('/api/query', methods=['POST'])
def api_query():
    """Receive query request from frontend, call RAG engine and return result"""
    global rag_engine_instance

    if not rag_engine_instance:
        return jsonify({"error": "RAG engine not initialized"}), 500

    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    try:
        # Call core RAG engine
        result = rag_engine_instance.query(query)

        # Log conversation to file
        try:
            conversation_logger.log_conversation(
                user_query=query,
                model_response=result.get("answer", ""),
                metadata={
                    "contexts_count": len(result.get("contexts", [])),
                    "has_results": len(result.get("contexts", [])) > 0
                }
            )
        except Exception as log_error:
            # Logging failure should not affect normal response
            logger.warning(f"Failed to log conversation: {log_error}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query in web API: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


# Wrapper class for running server in a background thread
class ServerThread(threading.Thread):
    def __init__(self, flask_app, host='127.0.0.1', port=5000):
        super().__init__()
        self.srv = make_server(host, port, flask_app)
        self.ctx = flask_app.app_context()
        self.ctx.push()

    def run(self):
        logger.info("Starting Flask server...")
        self.srv.serve_forever()

    def shutdown(self):
        logger.info("Shutting down Flask server...")
        self.srv.shutdown()


def run_web_server(engine, host='127.0.0.1', port=5000):
    """
    Main function to start Flask Web Server
    :param engine: RAG engine instance passed from main app
    :param host: Server host
    :param port: Server port
    """
    global rag_engine_instance
    rag_engine_instance = engine

    # Use werkzeug to start server so it can be shut down gracefully
    server_thread = ServerThread(app, host, port)
    server_thread.start()
    logger.info(f"Web UI server is running on http://{host}:{port}")
    return server_thread