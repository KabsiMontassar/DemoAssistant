"""
Chitchat Handler Module
Handles greetings and off-topic queries without RAG retrieval.
"""

import logging
from typing import Dict
import random

logger = logging.getLogger(__name__)


class ChitchatHandler:
    """
    Handles chitchat queries (greetings, casual conversation) without retrieval.
    Provides direct responses for non-domain queries.
    """
    
    def __init__(self):
        """Initialize chitchat handler with pre-defined responses."""
        
        # Greeting patterns and responses
        self.greeting_responses = [
            "Hello! I'm your Material Pricing Assistant. I can help you with:\n\n"
            "• Material prices (concrete, wood, metal, stone)\n"
            "• Project specifications and details\n"
            "• Supplier information\n"
            "• Material comparisons\n\n"
            "What would you like to know?",
            
            "Hi there! I specialize in material pricing and construction project information. "
            "Ask me about material costs, project details, or supplier data!",
            
            "Welcome! I'm here to assist with material pricing queries. "
            "Feel free to ask about concrete, wood, metal, stone prices, or project specifications."
        ]
        
        # Help/capability responses
        self.help_responses = [
            "I can help you with:\n\n"
            "📊 **Material Pricing**: Get current prices for concrete, wood, metal, and stone\n"
            "🏗️ **Project Information**: Details about specific construction projects\n"
            "🏢 **Supplier Data**: Information about material suppliers and vendors\n"
            "⚖️ **Comparisons**: Compare materials across different projects or suppliers\n\n"
            "Try asking: 'What's the price of concrete in Project Acme?' or 'Compare wood prices across projects'"
        ]
        
        # Out-of-domain responses
        self.ood_responses = [
            "I'm specialized in material pricing and construction project information. "
            "I don't have information about that topic, but I can help with:\n\n"
            "• Material prices and specifications\n"
            "• Project details and costs\n"
            "• Supplier information\n"
            "• Material comparisons\n\n"
            "Is there anything related to construction materials I can help you with?",
            
            "That's outside my area of expertise. I focus on material pricing and project information. "
            "Would you like to know about material costs or project specifications instead?"
        ]
        
        logger.info("✓ Chitchat handler initialized")
    
    def handle_query(self, query: str, confidence: float) -> Dict[str, any]:
        """
        Generate appropriate chitchat response based on query type.
        
        Args:
            query: User query string
            confidence: Intent confidence score (0-100)
            
        Returns:
            Dict with response and metadata
        """
        try:
            query_lower = query.lower().strip()
            
            # Detect query type
            if self._is_greeting(query_lower):
                response_type = "greeting"
                response = random.choice(self.greeting_responses)
            elif self._is_help_request(query_lower):
                response_type = "help"
                response = random.choice(self.help_responses)
            else:
                response_type = "out_of_domain"
                response = random.choice(self.ood_responses)
            
            result = {
                "response": response,
                "sources": [],
                "web_search_used": False,
                "metadata": {
                    "response_type": response_type,
                    "confidence": confidence,
                    "retrieval_used": False,
                    "handler": "chitchat"
                }
            }
            
            logger.info(f"Chitchat response: {response_type} | Query: {query[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling chitchat: {str(e)}", exc_info=True)
            # Fallback response
            return {
                "response": (
                    "I'm here to help with material pricing and project information. "
                    "Could you please rephrase your question?"
                ),
                "sources": [],
                "web_search_used": False,
                "metadata": {
                    "response_type": "error_fallback",
                    "error": str(e)
                }
            }
    
    @staticmethod
    def _is_greeting(query: str) -> bool:
        """Check if query is a greeting."""
        greetings = [
            'hello', 'hi', 'hey', 'greetings', 'good morning',
            'good afternoon', 'good evening', 'howdy', 'sup',
            'yo', 'hola', 'bonjour', 'hallo'
        ]
        
        # Exact match or starts with greeting
        words = query.split()
        if len(words) <= 3:  # Short greeting queries
            return any(greeting in query for greeting in greetings)
        
        return False
    
    @staticmethod
    def _is_help_request(query: str) -> bool:
        """Check if query is asking for help/capabilities."""
        help_keywords = [
            'help', 'what can you do', 'capabilities', 'features',
            'how to use', 'what do you know', 'tell me about yourself',
            'what are you', 'who are you', 'your purpose'
        ]
        
        return any(keyword in query for keyword in help_keywords)
    
    def add_custom_response(self, response_type: str, response: str):
        """
        Add custom response to handler.
        
        Args:
            response_type: Type ('greeting', 'help', 'ood')
            response: Response text
        """
        if response_type == "greeting":
            self.greeting_responses.append(response)
        elif response_type == "help":
            self.help_responses.append(response)
        elif response_type in ["ood", "out_of_domain"]:
            self.ood_responses.append(response)
        
        logger.info(f"Added custom {response_type} response")
    
    def get_stats(self) -> Dict[str, any]:
        """Get handler statistics."""
        return {
            "greeting_responses": len(self.greeting_responses),
            "help_responses": len(self.help_responses),
            "ood_responses": len(self.ood_responses)
        }
