"""
Conversation logging module: Responsible for saving conversation records between user and the large model.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ConversationLogger:
    def __init__(self, log_dir: str = "conversation_logs"):
        """
        Initialize conversation logger

        Args:
            log_dir: Directory to store log files, defaults to "conversation_logs"
        """
        self.log_dir = Path(log_dir)
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensure the log directory exists"""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Conversation log directory is ready: {self.log_dir}")
        except Exception as e:
            logger.error(f"Failed to create log directory: {str(e)}")
            raise

    def _get_log_file_path(self) -> Path:
        """
        Get the log file path for the current date

        Returns:
            The full path of the log file
        """
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"conversations_{today}.json"
        return self.log_dir / filename

    def log_conversation(self, user_query: str, model_response: str, metadata: Dict[str, Any] = None):
        """
        Log a conversation

        Args:
            user_query: The query sent by the user
            model_response: The response returned by the large model
            metadata: Optional extra metadata (e.g., session ID, source, etc.)
        """
        try:
            # Build conversation entry
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "model_response": model_response
            }

            # If there is extra metadata, add it
            if metadata:
                conversation_entry["metadata"] = metadata

            # Get log file path
            log_file = self._get_log_file_path()

            # Read existing logs (if file exists)
            conversations = []
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        conversations = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Log file {log_file} format error, a new file will be created")
                    conversations = []

            # Add new conversation entry
            conversations.append(conversation_entry)

            # Write to file
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)

            logger.info(f"Conversation record saved to {log_file}")

        except Exception as e:
            logger.error(f"Error saving conversation record: {str(e)}")

    def get_conversations_by_date(self, date_str: str) -> list:
        """
        Get all conversation records for a specific date

        Args:
            date_str: Date string, formatted as "YYYY-MM-DD"

        Returns:
            List of conversation records
        """
        try:
            filename = f"conversations_{date_str}.json"
            log_file = self.log_dir / filename

            if not log_file.exists():
                logger.warning(f"Log file for date {date_str} does not exist")
                return []

            with open(log_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            return conversations

        except Exception as e:
            logger.error(f"Error reading log file: {str(e)}")
            return []

    def get_all_log_files(self) -> list:
        """
        Get a list of all log files

        Returns:
            List of log file paths
        """
        try:
            log_files = sorted(self.log_dir.glob("conversations_*.json"))
            return [str(f) for f in log_files]
        except Exception as e:
            logger.error(f"Error getting log file list: {str(e)}")
            return []
