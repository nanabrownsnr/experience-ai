"""
Storage adapters for persisting interaction data.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any


class LocalStorageAdapter:
    """
    A storage adapter that persists interaction data to a local JSON file.
    
    This adapter provides simple file-based storage for interaction data,
    allowing the system to learn from past interactions.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the storage adapter with a file path.
        
        Args:
            filepath (str): Path to the JSON file for storing interactions
        """
        self.filepath = filepath
    
    def read_interactions(self) -> List[Dict[str, Any]]:
        """
        Read all stored interactions from the JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of interaction dictionaries, or empty list if file doesn't exist
        """
        if not os.path.exists(self.filepath):
            return []
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read interactions from {self.filepath}: {e}")
            return []
    
    def write_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Write a new interaction to the JSON file.
        
        Args:
            interaction_data (Dict[str, Any]): The interaction data to store
        """
        # Read existing interactions
        interactions = self.read_interactions()
        
        # Add timestamp if not present
        if 'timestamp' not in interaction_data:
            interaction_data['timestamp'] = datetime.now().isoformat()
        
        # Append new interaction
        interactions.append(interaction_data)
        
        # Write back to file
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True) if os.path.dirname(self.filepath) else None
            with open(self.filepath, 'w', encoding='utf-8') as file:
                json.dump(interactions, file, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error: Could not write to {self.filepath}: {e}")
    
    def clear_interactions(self) -> None:
        """
        Clear all stored interactions.
        """
        try:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
        except IOError as e:
            print(f"Error: Could not clear interactions from {self.filepath}: {e}")
    
    def get_interaction_count(self) -> int:
        """
        Get the total number of stored interactions.
        
        Returns:
            int: Number of interactions stored
        """
        return len(self.read_interactions())
