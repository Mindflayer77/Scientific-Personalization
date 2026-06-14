import random
import time
from enum import Enum
from google import genai
from google.genai.types import ThinkingConfig, GenerateContentConfig
from api_client.gemini_parser import GeminiResponseParser

"""
    ****** gemini-3.1-flash-lite-preview ******

    Temperature: 0.0-2.0 (default 1.0)
    topP: 0.0-1.0 (default 0.95)
    topK: 64 (fixed)
    candidateCount: 1-8 (default 1)

    ****** gemini-3-flash-preview ******

    Temperature: 0.0-2.0 (default 1.0)
    topP: 0.0-1.0 (default 0.95)
    topK: 64 (fixed)
    candidateCount: 1-8 (default 1)

    ****** gemini-3.1-pro-preview ******

    Temperature: 0.0-2.0 (default 1.0)
    topP: 0.0-1.0 (default 0.95)
    topK: 64 (fixed)
    candidateCount: 1-8 (default 1)

    ****** gemini-2.5-pro ******

    Temperature: 0.0-2.0 (default 1.0)
    topP: 0.0-1.0 (default 0.95)
    topK: 64 (fixed)
    candidateCount: 1-8 (default 1)


"""

REASONING_MODELS = set([
    'gemini-3.1-flash-lite-preview',
    'gemini-3-flash-preview',
    'gemini-3.1-pro-preview',
    'gemini-2.5-pro',
])

class ThinkingLevel(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'

class GeminiClient:
    def __init__(self, project_id, location="global"):
        self.client = genai.Client(
            vertexai=True, 
            project=project_id,
            location=location
        )
        self._default_parsing_params = {}
        
    def query(self, model, system_message, user_message, thinking_level: ThinkingLevel = None, response_schema=None, **generation_params):
        if response_schema:
            generation_params['response_mime_type'] = 'application/json'
            generation_params['response_schema'] = response_schema
        if model in REASONING_MODELS:
            if thinking_level:
                generation_params['thinking_config'] = ThinkingConfig(include_thoughts=True, thinking_level=thinking_level)
            else:
                generation_params['thinking_config'] = ThinkingConfig(include_thoughts=True, thinking_budget=-1)

        max_retries = 8
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_message,
                    config=GenerateContentConfig(
                        system_instruction=system_message,
                        **generation_params,
                    ),
                )
                return response
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_last = attempt == max_retries - 1
                if is_429 and not is_last:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 3)
                    print(f"[429] Rate limited. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})…")
                    time.sleep(delay)
                else:
                    print(f"Could not process the request. {e}")
                    raise e
    
    async def async_query(
        self,
        model: str,
        system_message: str,
        user_message: str,
        thinking_level: ThinkingLevel = None,
        response_schema=None,
        **generation_params,
    ):
        if response_schema:
            generation_params['response_mime_type'] = 'application/json'
            generation_params['response_schema'] = response_schema

        if model in REASONING_MODELS:
            if thinking_level:
                generation_params['thinking_config'] = ThinkingConfig(
                    include_thoughts=True, thinking_level=thinking_level
                )
            else:
                generation_params['thinking_config'] = ThinkingConfig(include_thoughts=True)
        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=user_message,
                config=GenerateContentConfig(
                    system_instruction=system_message,
                    **generation_params,
                ),
            )
            return response
        except Exception as e:
            print(f"Could not process the request. {e}")
            raise e