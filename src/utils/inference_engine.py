from loguru import logger
from openai import OpenAI, AzureOpenAI
from typing import Dict, List
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from functools import wraps
from datetime import datetime
from typing import Callable
from asyncio import Semaphore
import google.generativeai as genai

import sys
import os
import unittest
import anthropic
import asyncio
import json
import logging
import logging.handlers
import aiohttp

# === logging setup ===
os.makedirs("logs", exist_ok=True)

# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s | %(levelname)s | %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Main logger setup
logger = logging.getLogger('default')
logger.setLevel(logging.INFO)

# File handler for default logs
file_handler = logging.handlers.RotatingFileHandler(
    "logs/default.log",
    maxBytes=500*1024*1024,  # 500 MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

# Console handler for default logs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# Inference logger setup
inference_logger = logging.getLogger('inference')
inference_logger.setLevel(logging.INFO)

inference_handler = logging.handlers.RotatingFileHandler(
    "logs/inference.log",
    maxBytes=500*1024*1024,  # 500 MB
    backupCount=5
)
inference_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
inference_logger.addHandler(inference_handler)

def log_inference(func: Callable):
    @wraps(func)
    def wrapper(self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters = Hyperparameters()):
        start_time = datetime.now()
        result = None
        usage = None
        log_entry = None
        
        try:
            # The function returns a tuple of (result, usage)
            result, usage = func(self, messages, hyperparameters)
            
            log_entry = {
                "strategy": self.strategy,
                "model": self.model_name,
                "hyperparameters": hyperparameters.kwargs,
                "messages": messages,
                "output": result,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "time_taken_seconds": (datetime.now() - start_time).total_seconds(),
                "status": "success",
                "usage": usage
            }
            return result, usage
            
        except Exception as e:
            log_entry = {
                "strategy": self.strategy,
                "model": self.model_name,
                "hyperparameters": hyperparameters.kwargs,
                "messages": messages,
                "error": str(e),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "time_taken_seconds": (datetime.now() - start_time).total_seconds(),
                "status": "error",
                "usage": None
            }
            return "", {}
        finally:
            if log_entry:
                inference_logger.info(json.dumps(log_entry))

    return wrapper

# Update async_log_inference decorator
def async_log_inference(func: Callable):
    @wraps(func)
    async def async_wrapper(self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters = Hyperparameters()):
        start_time = datetime.now()
        result = None
        usage = None
        log_entry = None
        
        try:
            # The function returns a tuple of (result, usage)
            result, usage = await func(self, messages, hyperparameters)
            
            log_entry = {
                "strategy": self.strategy,
                "model": self.model_name,
                "hyperparameters": hyperparameters.kwargs,
                "messages": messages,
                "output": result,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "time_taken_seconds": (datetime.now() - start_time).total_seconds(),
                "status": "success",
                "usage": usage
            }
            # Return both result and usage
            return result, usage
            
        except Exception as e:
            log_entry = {
                "strategy": self.strategy,
                "model": self.model_name,
                "hyperparameters": hyperparameters.kwargs,
                "messages": messages,
                "error": str(e),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "time_taken_seconds": (datetime.now() - start_time).total_seconds(),
                "status": "error",
                "usage": None
            }
            # return blank result and usage
            return "", {}
        finally:
            if log_entry:
                inference_logger.info(json.dumps(log_entry))

    return async_wrapper

# === constants ===
load_dotenv()

COMPATIBLE_STRATEGIES = ["openai", "anthropic", "azure", "gemini"]
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 10))
RETRY_WAIT_EXPONENTIAL_MULTIPLIER = int(os.getenv("RETRY_WAIT_EXPONENTIAL_MULTIPLIER", 1))
RETRY_WAIT_EXPONENTIAL_MIN = int(os.getenv("RETRY_WAIT_EXPONENTIAL_MIN", 4))
RETRY_WAIT_EXPONENTIAL_MAX = int(os.getenv("RETRY_WAIT_EXPONENTIAL_MAX", 300))
RUN_EXPENSIVE_TESTS = os.getenv("RUN_EXPENSIVE_TESTS", "false").lower() == "true"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 1024))

class Hyperparameters:
    """
    Hyperparameters for inference.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        return

class InferenceEngine:
    """
    Handles inference for any LLM
    """
    def __init__(self, connection_details: Dict[str, str], model_name: str = None):
        assert self._check_model_compatibility(connection_details["strategy"]), "Strategy not compatible"
        self.strategy = connection_details["strategy"]
        self.client = self._initialize_client(connection_details)
        self.model_name = model_name
        self.base_url = connection_details["azure_endpoint"] if connection_details["strategy"] == "azure" else connection_details["base_url"]
        self.api_version = connection_details["api_version"] if connection_details["strategy"] == "azure" else "2024-02-01"
        # Initialize session to None - will be created when needed
        self._session = None
        return
    
    def _initialize_client(self, connection_details: Dict[str, str]):
        """
        Initialize the client based on the strategy.
        """
        if connection_details["strategy"] == "openai":
            return OpenAI(
                api_key = connection_details["api_key"],
                base_url = connection_details["base_url"],
            )
        elif connection_details["strategy"] == "azure":
            return AzureOpenAI(
                api_key = connection_details["api_key"],
                azure_endpoint = connection_details["azure_endpoint"],
                api_version = connection_details["api_version"],
            )
        elif connection_details["strategy"] == "gemini":
            return genai.GenerativeModel(
                api_key = connection_details["api_key"],
            )
        elif connection_details["strategy"] == "anthropic":
            return anthropic.Anthropic(
                api_key = connection_details["api_key"],
            )
        return
    
    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_EXPONENTIAL_MULTIPLIER, min=RETRY_WAIT_EXPONENTIAL_MIN, max=RETRY_WAIT_EXPONENTIAL_MAX),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.debug(
            f"Retry attempt {retry_state.attempt_number} failed with error: {retry_state.outcome.exception()}. Retrying..."
        )
    )
    @log_inference
    def single_inference(self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters = Hyperparameters()) -> str:
        """
        Perform an inference, with a single message
        """
        if self.strategy in ["openai", "azure"]:
            return self._openai_single_inference(messages, hyperparameters)
        else:
            raise NotImplementedError(f"Single inference not implemented for strategy {self.client.strategy}")

    @async_log_inference
    async def _async_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        if self.strategy == "azure":
            return await self._async_azure_openai_single_message_inference(messages, hyperparameters)
        elif self.strategy == "openai":
            return await self._async_openai_single_message_inference(messages, hyperparameters)
        elif self.strategy == "gemini":
            return await self._async_gemini_single_message_inference(messages, hyperparameters)
        elif self.strategy == "anthropic":
            return await self._async_anthropic_single_message_inference(messages, hyperparameters)
        else:
            raise ValueError(f"Invalid inference strategy: {self.strategy}")
    
    async def _async_gemini_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        raise NotImplementedError(f"Async Gemini single message inference not implemented for strategy {self.strategy}")
    
    async def _async_anthropic_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> str:
        raise NotImplementedError(f"Async Anthropic single message inference not implemented for strategy {self.strategy}")
    
    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_EXPONENTIAL_MULTIPLIER, min=RETRY_WAIT_EXPONENTIAL_MIN, max=RETRY_WAIT_EXPONENTIAL_MAX),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.debug(
            f"Retry attempt {retry_state.attempt_number} failed with error: {retry_state.outcome.exception()}. Retrying..."
        )
    )
    async def _async_openai_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> tuple[str, dict]:
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": hyperparameters.kwargs.get("temperature", 0.0),
            "max_tokens": hyperparameters.kwargs.get("max_tokens", 4096),
        }
        
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}chat/completions",
                headers={
                    "Authorization": f"Bearer {self.client.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_payload
            ) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    logger.error(f"API request failed with status {response.status}: {response_text}")
                    raise Exception(f"API request failed with status {response.status}: {response_text}")
                    
                result = json.loads(response_text)
                
                usage = {
                    "prompt_tokens": result['usage']['prompt_tokens'],
                    "completion_tokens": result['usage']['completion_tokens'],
                    "total_tokens": result['usage']['total_tokens']
                }
                return result['choices'][0]['message']['content'], usage
                
        except asyncio.TimeoutError:
            raise Exception("Request timed out after 6000 seconds")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_EXPONENTIAL_MULTIPLIER, min=RETRY_WAIT_EXPONENTIAL_MIN, max=RETRY_WAIT_EXPONENTIAL_MAX),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.debug(
            f"Retry attempt {retry_state.attempt_number} failed with error: {retry_state.outcome.exception()}. Retrying..."
        )
    )
    async def _async_azure_openai_single_message_inference(
        self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters
    ) -> tuple[str, dict]:
        request_payload = {
            "messages": messages,
            "temperature": hyperparameters.kwargs.get("temperature", 0.0),
            "max_tokens": hyperparameters.kwargs.get("max_tokens", 4096),
        }
        
        session = await self._get_session()
        try:
            url = f"{self.client.base_url}deployments/{self.model_name}/chat/completions"
            if self.api_version:
                url += f"?api-version={self.api_version}"
            
            async with session.post(
                url,
                headers={
                    "api-key": f"{self.client.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_payload
            ) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    logger.error(f"API request failed with status {response.status}: {response_text}")
                    raise Exception(f"API request failed with status {response.status}: {response_text}")
                    
                result = json.loads(response_text)
                
                usage = {
                    "prompt_tokens": result['usage']['prompt_tokens'],
                    "completion_tokens": result['usage']['completion_tokens'],
                    "total_tokens": result['usage']['total_tokens']
                }
                return result['choices'][0]['message']['content'], usage
                
        except asyncio.TimeoutError:
            raise Exception("Request timed out after 6000 seconds")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    def _openai_single_inference(self, messages: List[Dict[str, str]], hyperparameters: Hyperparameters = None) -> tuple[str, dict]:
        """
        Perform an inference, with a single message, using OpenAI
        """
        # check the model. if its not o1-preview or o1-mini, then use the default response creation
        if self.model_name not in ["o1-preview", "o1-mini"]:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                max_tokens = hyperparameters.kwargs.get("max_tokens", 4096),
                temperature = hyperparameters.kwargs.get("temperature", 0.0),
            )
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            return response.choices[0].message.content, usage
        else:
            raise NotImplementedError(f"Default response creation not implemented for model {self.model_name}")

    async def _improved_parallel_messages_inference(
        self,
        messages: List[List[Dict[str, str]]],
        hyperparameters: Hyperparameters = Hyperparameters(),
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
    ) -> tuple[List[str], List[dict]]:
        semaphore = Semaphore(max_concurrent_requests)
        results = [None] * len(messages)
        usages = [None] * len(messages)
        
        async def process_message(index, msgs):
            async with semaphore:
                try:
                    result, usage = await self._async_single_message_inference(msgs, hyperparameters)
                    results[index] = result
                    usages[index] = usage
                except Exception as e:
                    logger.error(f"Inference task failed for index {index}: {e}")
                    results[index] = ""
                    usages[index] = {}

        try:
            tasks = [asyncio.create_task(process_message(i, msgs)) for i, msgs in enumerate(messages)]
            await asyncio.gather(*tasks)
            return results, usages
        finally:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    def parallel_inference(
        self,
        messages: List[List[Dict[str, str]]],
        hyperparameters: Hyperparameters = Hyperparameters(),
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    ) -> tuple[List[str], List[dict]]:
        async def run_inference():
            try:
                return await self._improved_parallel_messages_inference(
                    messages, hyperparameters, max_concurrent_requests
                )
            finally:
                if self._session:
                    await self._session.close()
                    self._session = None

        return asyncio.run(run_inference())

    @staticmethod
    def _check_model_compatibility(strategy_name: str) -> bool:
        """
        Check if we're able to use the given strategy for inference.
        """
        return strategy_name in COMPATIBLE_STRATEGIES

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=6000)
            connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

class TestInferenceEngine(unittest.TestCase):
    async def asyncTearDown(self):
        """Clean up any remaining sessions after each async test"""
        if hasattr(self, 'engine') and self.engine._session:
            await self.engine._session.close()
            self.engine._session = None

    def tearDown(self):
        """Run async teardown in sync context"""
        if hasattr(self, 'engine') and self.engine._session:
            asyncio.run(self.asyncTearDown())

    def test_incompatible_strategy(self):
        with self.assertRaises(AssertionError):
            InferenceEngine({"strategy": "incompatible_strategy"})

    def test_compatible_strategy(self):
        try:
            InferenceEngine({
                "strategy": "openai",
                "api_key": "dummy_api_key",
                "base_url": "https://api.openai.com/v1"
            })
        except AssertionError:
            self.fail("InferenceEngine raised AssertionError unexpectedly!")
    
    def test_azure_single_inference(self):
        if not RUN_EXPENSIVE_TESTS:
            self.skipTest("Skipping expensive test")
        try:
            engine = InferenceEngine({
                "strategy": "azure",
                "api_key": os.getenv("AZURE_API_KEY"),
                "azure_endpoint": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            }, model_name="gpt-4o-mini")
            
            result, usage = engine.single_inference(
                messages=[{"role": "user", "content": "Hello, how are you?"}]
            )

            self.assertIsInstance(result, str)
            self.assertIsInstance(usage, dict)
            self.assertTrue(len(result) > 0)
            self.assertIn('prompt_tokens', usage)
            self.assertIn('completion_tokens', usage)
            self.assertIn('total_tokens', usage)
            
        except AssertionError:
            self.fail("InferenceEngine raised AssertionError unexpectedly!")

    async def async_test_single_message_inference(self):
        if not RUN_EXPENSIVE_TESTS:
            self.skipTest("Skipping expensive test")
            
        self.engine = InferenceEngine({  # Store engine as instance variable
            "strategy": "azure",
            "api_key": os.getenv("AZURE_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        }, model_name="gpt-4o-mini")
        
        test_message = [{"role": "user", "content": "Say hello!"}]
        
        try:
            result, usage = await self.engine._async_single_message_inference(
                messages=test_message,
                hyperparameters=Hyperparameters(temperature=0.0)
            )
            
            # Validate the result
            self.assertIsInstance(result, str)
            self.assertIsInstance(usage, dict)
            self.assertTrue(len(result) > 0)
            self.assertIn('prompt_tokens', usage)
            self.assertIn('completion_tokens', usage)
            self.assertIn('total_tokens', usage)
            
        except Exception as e:
            self.fail(f"Async single message inference failed with error: {str(e)}")
        finally:
            await self.asyncTearDown()

    def test_single_message_inference(self):
        """Wrapper to run the async test"""
        if not RUN_EXPENSIVE_TESTS:
            self.skipTest("Skipping expensive test")
        asyncio.run(self.async_test_single_message_inference())

    def test_azure_parallel_inference(self):
        if not RUN_EXPENSIVE_TESTS:
            self.skipTest("Skipping expensive test")
            
        self.engine = InferenceEngine({  # Store engine as instance variable
            "strategy": "azure",
            "api_key": os.getenv("AZURE_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        }, model_name="gpt-4o-mini")
        
        # Create 5 test messages
        messages = [
            [{"role": "user", "content": f"Count to {i}"}] for i in range(1, 6)
        ]
        
        # Test parallel inference
        try:
            results, usages = self.engine.parallel_inference(
                messages=messages,
                hyperparameters=Hyperparameters(temperature=0.0),
                max_concurrent_requests=3  # Limit concurrent requests for testing
            )
            
            # Basic validation
            self.assertEqual(len(results), 5)
            self.assertTrue(all(isinstance(result, str) for result in results))
            self.assertEqual(len(usages), 5)
            self.assertTrue(all(isinstance(usage, dict) for usage in usages))
            
        except Exception as e:
            self.fail(f"Parallel inference failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main()
