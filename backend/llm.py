"""
Calls OpenRouter API with Nemotron model
Constructs system prompt defining behavior
Includes retrieved context in prompt
Optional: Adds web search results from Tavily
Returns markdown-formatted response
"""

import os
import json
import logging
from typing import Optional
import httpx
from tavily import TavilyClient

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages interactions with OpenRouter API.
    Handles prompt construction, response generation, and web search integration.
    """
    
    # OpenRouter API endpoint
    API_BASE = "https://openrouter.ai/api/v1"
    
    # Model configuration
    DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    
    def __init__(self):
        """Initialize LLM manager with API credentials and configuration."""
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.model = os.getenv('LLM_MODEL', self.DEFAULT_MODEL)
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2000'))
        
        # Web search configuration
        self.web_search_enabled = os.getenv('WEB_SEARCH_ENABLED', 'true').lower() == 'true'
        tavily_key = os.getenv('TAVILY_API_KEY')
        self.tavily_client = TavilyClient(api_key=tavily_key) if tavily_key and self.web_search_enabled else None
        
        # HTTP client with timeout
        self.client = httpx.Client(timeout=30.0)
        
        logger.info(f"LLM Manager initialized with OpenRouter model: {self.model}")
        if self.tavily_client:
            logger.info("Web search enabled via Tavily API")
    
    def _search_web(self, query: str, max_results: int = 5) -> str:
        """
        Perform web search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            Formatted web search results
        """
        if not self.tavily_client:
            return ""
        
        try:
            logger.info(f"Performing web search for: {query[:100]}...")
            response = self.tavily_client.search(query=query, max_results=max_results)
            
            if not response.get('results'):
                return ""
            
            # Format search results
            web_results = "CURRENT WEB SEARCH RESULTS:\n"
            for result in response['results']:
                web_results += f"\n- Source: {result.get('title', 'Unknown')}\n"
                web_results += f"  Content: {result.get('content', '')[:300]}...\n"
            
            logger.info(f"Web search returned {len(response.get('results', []))} results")
            return web_results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return ""
    
    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt for the LLM.
        Defines behavior, constraints, and response format.
        
        Returns:
            System prompt string
        """
        return """You are Atlas, a professional and energetic AI Material Assistant. Your role is to provide clear, factual information about material pricing and project history from the provided context.

CORE PRINCIPLES:
1. FOCUS: Only provide information that exists in the provided context or web search results.
2. ACCURACY: Do NOT make up or hallucinate pricing information or project details.
3. HELPFULNESS: If a query is short or ambiguous (like "dsq"), but the context contains relevant material data, provide a concise summary of what you found instead of a generic refusal.
4. HONESTY: Only say "I don't have that information" if the provided context is truly irrelevant to the query.
5. CONCiseness: Be direct and factual. Avoid fluff.

RESPONSE FORMATTING:
- Start IMMEDIATELY with the information (NEVER use prefixes like "Answer:", "Response:", or "Results:").
- Use clean Markdown: ## for main sections, ### for subsections.
- Use **bold** for prices, dates, material names, and project names.
- Do NOT add a "Source:" or "References" section at the end (the system handles citations separately).
- Make the response look professional, energetic, and extremely readable.
"""
    
    def _construct_user_prompt(
        self,
        query: str,
        context: str,
        web_search_results: str = "",
        use_web_search: bool = False
    ) -> str:
        """
        Construct the user message prompt with context and instructions.
        
        Args:
            query: User's question
            context: Retrieved document context
            web_search_results: Web search results if available
            use_web_search: Whether web search is enabled
            
        Returns:
            Formatted user prompt
        """
        prompt = f"""
CONTEXT INFORMATION:
---
{context}
{web_search_results}
---

USER QUERY: {query}

Provide a factual answer based ONLY on the context above. If the query is just a keyword or shorthand, provide a summary of the most relevant prices or specs found in the context.
"""
        return prompt
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list,
        use_web_search: bool = False
    ) -> dict:
        """
        Generate an answer from retrieved chunks with full source attribution.
        
        Args:
            query: User's question
            retrieved_chunks: List of scored chunks with metadata
            use_web_search: Whether to include web search
            
        Returns:
            Dict with answer, confidence, and sources
        """
        try:
            # Format context from chunks
            context_lines = []
            sources = set()
            
            for chunk in retrieved_chunks:
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                source_file = metadata.get("file_path", "unknown")
                sources.add(source_file)
                
                # Include chunk text with source
                context_lines.append(f"{text}\n[Source: {source_file}]")
            
            context = "\n\n".join(context_lines)
            
            # Generate response
            response = self.generate_response(query, context, use_web_search)
            
            # Extract confidence from chunks if available
            confidences = [c.get("scores", {}).get("confidence", "low") 
                          for c in retrieved_chunks]
            confidence_level = confidences[0] if confidences else "low"
            
            return {
                "answer": response,
                "confidence": confidence_level,
                "sources": list(sources),
                "chunks_used": len(retrieved_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def generate_response(
        self,
        query: str,
        context: str,
        use_web_search: bool = False
    ) -> str:
        """
        Generate an AI response to a user query using retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document context
            use_web_search: Whether to include web search in consideration
            
        Returns:
            Generated response from the LLM
            
        Raises:
            Exception: If API call fails or response is invalid
        """
        try:
            # Perform web search if enabled
            web_search_results = ""
            if use_web_search and self.tavily_client:
                web_search_results = self._search_web(query)
            
            # Construct prompts
            system_prompt = self._construct_system_prompt()
            user_prompt = self._construct_user_prompt(query, context, web_search_results, use_web_search)
            
            logger.info("Calling OpenRouter API...")
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Material Pricing AI Assistant"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": 0.9
            }
            
            # Make API call
            response = self.client.post(
                f"{self.API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Handle API errors
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"OpenRouter API error ({response.status_code}): {error_detail}")
                
                if response.status_code == 401:
                    raise Exception("Invalid OpenRouter API key. Check OPENROUTER_API_KEY.")
                elif response.status_code == 429:
                    raise Exception("Rate limit exceeded. Please try again later.")
                elif response.status_code == 500:
                    raise Exception("OpenRouter API server error. Please try again.")
                else:
                    raise Exception(f"API error: {error_detail}")
            
            # Parse response
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from API: {response.text}")
                raise Exception("Invalid response from API")
            
            # Extract message
            if 'choices' not in response_data or not response_data['choices']:
                logger.error(f"No choices in API response: {response_data}")
                raise Exception("Unexpected API response format")
            
            message = response_data['choices'][0].get('message', {}).get('content', '')
            
            if not message:
                logger.error(f"Empty message in API response: {response_data}")
                raise Exception("Received empty response from API")
            
            # Log token usage if available
            if 'usage' in response_data:
                usage = response_data['usage']
                logger.debug(
                    f"Token usage - Input: {usage.get('prompt_tokens', 0)}, "
                    f"Output: {usage.get('completion_tokens', 0)}, "
                    f"Total: {usage.get('total_tokens', 0)}"
                )
            
            logger.info("✓ Response generated successfully")
            return message
            
        except httpx.TimeoutException:
            error_msg = "API request timed out. Please try again."
            logger.error(error_msg)
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Network error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise
    
    def __del__(self):
        """Clean up HTTP client on deletion."""
        try:
            self.client.close()
        except:
            pass
