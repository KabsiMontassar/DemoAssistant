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
    
    def _search_web(self, query: str, max_results: int = 5) -> list:
        """
        Perform web search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of search result dictionaries
        """
        if not self.tavily_client:
            return []
        
        try:
            logger.info(f"Performing web search for: {query[:100]}...")
            response = self.tavily_client.search(query=query, max_results=max_results)
            
            results = response.get('results', [])
            logger.info(f"Web search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt for the LLM.
        Defines behavior, constraints, and response format.
        """
        return """You are Atlas, an enterprise-grade AI assistant for material pricing and project data.
Your sole responsibility is to provide precise, verifiable answers strictly grounded in the provided context.

OPERATING RULES (STRICT):
1. GROUNDING:
   - Use ONLY the information explicitly present in the provided context (Project Documents AND Web Search Results).
   - NEVER infer, estimate, interpolate, or assume missing values.
   - If a value is not present, it must not appear in the answer.

2. ACCURACY:
   - Provide prices, dates, and specifications EXACTLY as they appear in the documents or search results.
   - If multiple contradictory values exist (e.g., project price vs current market price), cite both with their respective sources.

3. AMBIGUITY HANDLING:
   - If the user query is vague (e.g., "dsq"), and there is relevant context, provide a summary of the available information.
   - If NO valid context exists after checking both documents and web search, state clearly: "I could not find information about [query] in the available project documents or current market data."

4. FORMAT & STRUCTURE:
   - Use standard professional formatting.
   - Bold key figures (prices, dates).
   - Use Markdown Tables for datasets (prices/specs). Ensure correct syntax:
     - Headers must be separated from rows by a line of dashes and pipes (e.g., `| --- | --- |`).
     - Use newlines before and after tables.
   - Do NOT use emojis.
   - Do NOT use chatty introductions like "Here is the information" or prefixes like "Answer:". Start directly with the data.

RESPONSE STRUCTURE (REQUIRED):
## [Descriptive Topic Heading]
[Direct Answer / Markdown Table / List]

### Key Details
- [Detail 1]
- [Detail 2]
- [Contextual note about source types if applicable]
"""

    
    def _construct_user_prompt(
        self,
        query: str,
        context: str,
        web_search_results: str = ""
    ) -> str:
        """
        Construct the user message prompt with context and instructions.
        """
        prompt = f"""
PROJECT DOCUMENTS CONTEXT:
---
{context}
---

WEB SEARCH RESULTS (CURRENT MARKET DATA):
---
{web_search_results}
---

USER QUERY: {query}

Provide a factual answer based ONLY on the context blocks above. Prioritize Project Documents if they exist, but use Web Search Results for current market comparisons or if project data is missing.
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
            
            # Perform web search if requested
            web_search_results_raw = []
            actual_search_used = False
            if use_web_search:
                web_search_results_raw = self._search_web(query)
                if web_search_results_raw:
                    actual_search_used = True
            
            # Format web search results for prompt
            web_context = ""
            if web_search_results_raw:
                web_context_lines = []
                for res in web_search_results_raw:
                    title = res.get('title', 'Web Result')
                    url = res.get('url', 'N/A')
                    content = res.get('content', '')
                    web_context_lines.append(f"SOURCE [Web]: {title} ({url})\nCONTENT: {content}")
                web_context = "\n\n".join(web_context_lines)

            # Generate response
            response, _ = self.generate_response(query, context, web_context)
            
            # Collect web sources for main to use
            web_sources = []
            for res in web_search_results_raw:
                web_sources.append({
                    "title": res.get('title', 'Web Result'),
                    "url": res.get('url', ''),
                    "content": res.get('content', '')[:200]
                })

            # Extract confidence from chunks if available
            confidences = [c.get("scores", {}).get("confidence", "low") 
                          for c in retrieved_chunks]
            confidence_level = confidences[0] if confidences else "low"
            
            return {
                "answer": response,
                "confidence": confidence_level,
                "sources": list(sources),
                "web_sources": web_sources,
                "chunks_used": len(retrieved_chunks),
                "web_search_used": actual_search_used
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def generate_response(
        self,
        query: str,
        context: str,
        web_context: str = ""
    ) -> tuple[str, bool]:
        """
        Generate an AI response to a user query using retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document context
            web_context: Formatted web search results to include in the prompt
            
        Returns:
            Generated response from the LLM
            
        Raises:
            Exception: If API call fails or response is invalid
        """
        try:
            # Construct prompts
            system_prompt = self._construct_system_prompt()
            user_prompt = self._construct_user_prompt(query, context, web_context)
            
            actual_search_used = bool(web_context)
            logger.info(f"Calling OpenRouter API (Search context present: {actual_search_used})...")
            
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
            return message, actual_search_used
            
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
