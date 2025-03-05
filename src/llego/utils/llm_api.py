import asyncio
import logging
import re
from typing import List

import openai
from aiohttp import ClientSession

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class LLM_API:
    """
    Helper class to call the API and parse the responses.
    """

    def __init__(
        self,
        model: str,
        api_type: str,
        api_base: str,
        api_version: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_tokens: List,
        system_message: str,
        with_logprobs: bool,
    ) -> None:

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_tokens = list(stop_tokens)
        self.system_message = system_message
        self.with_logprobs = with_logprobs

        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.MAX_RETRIES = 4
        self.retry_backoff = [10, 30, 60]
        self.REQUEST_TIMEOUT = 30

    def _extract_retry_time(self, exception: str, attempt_num: int) -> int:
        """Calculate exact retry time from openai.error.RateLimitError exception message."""
        match = re.search(r"retry after (\d+) seconds", exception)
        if match:
            return int(match.group(1)) + 1
        else:
            return self.retry_backoff[attempt_num]

    async def _async_generate_without_logprobs(
        self, user_message: str, n_generations_per_prompt: int
    ):
        """Generate a response from the LLM async."""

        openai.api_type = self.api_type
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_key = self.api_key

        message = []
        message.append(
            {
                "role": "system",
                "content": self.system_message,
            }
        )
        message.append({"role": "user", "content": user_message})

        session = ClientSession(trust_env=True)
        openai.aiosession.set(session)

        resp = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = await openai.ChatCompletion.acreate(
                    engine=self.model,
                    messages=message,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=n_generations_per_prompt,
                    request_timeout=self.REQUEST_TIMEOUT,
                    stop=self.stop_tokens,
                )
                break
            except openai.error.RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    retry_time = self._extract_retry_time(str(e), attempt)
                    logger.info(
                        f"[LLM API] Rate Limit Error. Retrying in {retry_time} seconds"
                    )
                    await asyncio.sleep(retry_time)
            except asyncio.exceptions.TimeoutError as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.info(
                        f"[LLM API] OpenAI API timeout. Sleeping for {self.retry_backoff[attempt]} seconds"
                    )
                    await asyncio.sleep(self.retry_backoff[attempt])
            except Exception as e:
                logger.info(f"Error: {e}")
                await session.close()
                raise e

        await session.close()

        return resp

    async def _async_generate_with_logprobs(
        self, user_message: str, n_generations_per_prompt: int
    ):
        """Generate a response from the LLM async."""

        openai.api_type = self.api_type
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_key = self.api_key

        session = ClientSession(trust_env=True)
        openai.aiosession.set(session)

        resp = None

        for attempt in range(self.MAX_RETRIES):
            try:
                if self.model.startswith("gpt4"):
                    # if self.model.startswith("gpt"):

                    message = []
                    message.append(
                        {
                            "role": "system",
                            "content": self.system_message,
                        }
                    )
                    message.append({"role": "user", "content": user_message})

                    resp = await openai.ChatCompletion.acreate(
                        engine=self.model,
                        messages=message,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        n=n_generations_per_prompt,
                        request_timeout=self.REQUEST_TIMEOUT,
                        stop=self.stop_tokens,
                        logprobs=True,
                    )
                else:
                    resp = await openai.Completion.acreate(
                        engine=self.model,
                        prompt=user_message,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        n=n_generations_per_prompt,
                        request_timeout=self.REQUEST_TIMEOUT,
                        stop=self.stop_tokens,
                        logprobs=1,
                    )
                break
            except openai.error.RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    retry_time = self._extract_retry_time(str(e), attempt)
                    logger.info(
                        f"[LLM API] Rate Limit Error. Retrying in {retry_time} seconds"
                    )
                    await asyncio.sleep(retry_time)
            except asyncio.exceptions.TimeoutError as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.info(
                        f"[LLM API] OpenAI API timeout. Sleeping for {self.retry_backoff[attempt]} seconds"
                    )
                    await asyncio.sleep(self.retry_backoff[attempt])
            except Exception as e:
                await session.close()
                raise e

        await session.close()

        return resp

    async def _async_generate_concurrently(
        self, list_prompts, n_generations_per_prompt: int
    ):
        """
        Perform concurrent generation of responses from the LLM async.
        Returns a list of responses.
        """

        coroutines = []
        for prompt in list_prompts:
            if self.with_logprobs:
                logger.info("Generating with logprobs")
                coroutines.append(
                    self._async_generate_with_logprobs(prompt, n_generations_per_prompt)
                )

            else:
                logger.info("Generating without logprobs")
                coroutines.append(
                    self._async_generate_without_logprobs(
                        prompt, n_generations_per_prompt
                    )
                )

        tasks = [asyncio.create_task(c) for c in coroutines]

        results = [None] * len(coroutines)

        llm_response = await asyncio.gather(*tasks)

        for idx, response in enumerate(llm_response):
            if response is not None:
                resp = response
                results[idx] = resp

        return results
