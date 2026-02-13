"""
Chitchat Handler
Handles non-domain queries (greetings, help requests, clarifications) without retrieval.
Provides efficient, direct responses for conversational interactions.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ChitchatHandler:
    """
    Handles chitchat queries with direct LLM responses (no retrieval needed).
    Optimized for fast responses to conversational interactions.
    """
    
    def __init__(self, llm_manager=None):
        """
        Initialize chitchat handler.
        
        Args:
            llm_manager: LLMManager instance for generating responses
        """
        self.llm_manager = llm_manager
        logger.info("✓ Chitchat handler initialized")
    
    def _get_system_prompt(self) -> str:
        """
        Construct system prompt for chitchat interactions.
        Different from RAG system prompt - more conversational.
        """
        return """You are Atlas, a helpful AI assistant for material pricing and construction project data.

You are responding to a general conversational query (not a specific data request).

RESPONSE GUIDELINES:
1. Be friendly, professional, and concise
2. For greetings: Respond warmly and offer to help with material pricing queries
3. For help requests: Explain your capabilities (material pricing, project data, supplier information)
4. For thank you: Acknowledge politely and offer further assistance
5. For unclear queries: Ask for clarification about what material pricing information they need
6. Keep responses brief (2-3 sentences)

CAPABILITIES TO MENTION:
- Material pricing information (wood, concrete, metal, stone)
- Project-specific pricing data
- Supplier comparisons
- Material specifications and details

Do NOT make up specific prices or data - explain you can look them up if they provide details."""
    
    def handle_chitchat(self, query: str, intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate chitchat response without retrieval.
        
        Args:
            query: User query
            intent_info: Intent classification result from IntentClassifier
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Check for common patterns and provide template responses
            query_lower = query.lower().strip()
            
            # Pattern matching for instant responses (no LLM needed)
            template_response = self._try_template_response(query_lower)
            
            if template_response:
                logger.info(f"Using template response for chitchat: {query[:50]}...")
                return {
                    "response": template_response,
                    "sources": [],
                    "web_search_used": False,
                    "intent": "chitchat",
                    "confidence": intent_info.get("confidence", 80),
                    "handler": "template"
                }
            
            # If no template match and LLM available, generate response
            if self.llm_manager:
                logger.info(f"Generating LLM response for chitchat: {query[:50]}...")
                
                response = self._generate_llm_response(query)
                
                return {
                    "response": response,
                    "sources": [],
                    "web_search_used": False,
                    "intent": "chitchat",
                    "confidence": intent_info.get("confidence", 70),
                    "handler": "llm"
                }
            else:
                # Fallback if no LLM available
                return {
                    "response": "Hello! I'm Atlas, your material pricing assistant. I can help you find pricing and specifications for construction materials. What would you like to know?",
                    "sources": [],
                    "web_search_used": False,
                    "intent": "chitchat",
                    "confidence": 60,
                    "handler": "fallback"
                }
                
        except Exception as e:
            logger.error(f"Error handling chitchat: {str(e)}", exc_info=True)
            # Graceful fallback
            return {
                "response": "Hello! How can I assist you with material pricing information today?",
                "sources": [],
                "web_search_used": False,
                "intent": "chitchat",
                "confidence": 50,
                "handler": "error_fallback",
                "error": str(e)
            }
    
    def _try_template_response(self, query_lower: str) -> str:
        """
        Try to match query to template responses for instant replies.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            Template response string, or empty string if no match
        """
        # Greeting patterns
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(greeting in query_lower for greeting in greetings):
            return "Hello! I'm Atlas, your AI assistant for material pricing and construction project data. I can help you find prices, specifications, and supplier information for various construction materials. What would you like to know?"
        
        # Help patterns
        help_patterns = ["help", "what can you do", "capabilities", "how do you work", "what are you"]
        if any(pattern in query_lower for pattern in help_patterns):
            return """I can assist you with:
• **Material Pricing**: Get current and historical prices for wood, concrete, metal, and stone
• **Project Data**: Access pricing specific to different construction projects
• **Supplier Information**: Compare prices across different suppliers
• **Specifications**: Find detailed material specifications and properties

Just ask me about any material pricing or specification you need!"""
        
        # Thank you patterns
        thanks_patterns = ["thank you", "thanks", "appreciate", "grateful"]
        if any(pattern in query_lower for pattern in thanks_patterns):
            return "You're welcome! Feel free to ask if you need any more material pricing information."
        
        # Goodbye patterns
        goodbye_patterns = ["goodbye", "bye", "see you", "farewell"]
        if any(pattern in query_lower for pattern in goodbye_patterns):
            return "Goodbye! Come back anytime you need material pricing information."
        
        # No template match
        return ""
    
    def _generate_llm_response(self, query: str) -> str:
        """
        Generate response using LLM for non-template chitchat.
        
        Args:
            query: User query
            
        Returns:
            Generated response string
        """
        try:
            system_prompt = self._get_system_prompt()
            
            # Use LLM manager's client to make API call
            headers = {
                "Authorization": f"Bearer {self.llm_manager.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Material Pricing AI Assistant"
            }
            
            payload = {
                "model": self.llm_manager.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "max_tokens": 300  # Shorter for chitchat
            }
            
            response = self.llm_manager.client.post(
                f"{self.llm_manager.API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15.0
            )
            
            if response.status_code == 200:
                response_data = response.json()
                message = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return message if message else self._get_fallback_response()
            else:
                logger.warning(f"LLM API error for chitchat: {response.status_code}")
                return self._get_fallback_response()
                
        except Exception as e:
            logger.error(f"Error generating LLM chitchat response: {str(e)}")
            return self._get_fallback_response()
    
    @staticmethod
    def _get_fallback_response() -> str:
        """Fallback response when LLM generation fails."""
        return "Hello! I'm Atlas, your material pricing assistant. I can help you find pricing information for construction materials like wood, concrete, metal, and stone. What would you like to know?"
