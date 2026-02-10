"""
Multi-Provider AI Module

Supports multiple AI providers with automatic fallback:
- OpenAI GPT-4o Vision
- Anthropic Claude 3 (Sonnet/Opus)
- Google Gemini Pro Vision
- Ollama (local)
- LM Studio (local)
"""

import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import asyncio

from . import config
from .config import AIProvider, TaskType, ProviderConfig, get_provider_config, get_providers_for_task

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    """Standardized AI response"""
    content: str
    provider: AIProvider
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Any = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class VisionRequest:
    """Vision analysis request"""
    images: List[Path]
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = config.MAX_TOKENS
    temperature: float = config.TEMPERATURE
    response_format: Optional[str] = None  # "json" or None


@dataclass
class TextRequest:
    """Text analysis request"""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = config.MAX_TOKENS
    temperature: float = config.TEMPERATURE
    response_format: Optional[str] = None


class BaseAIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, provider_config: ProviderConfig):
        self.config = provider_config
        self.client = None
        self._init_client()
    
    @abstractmethod
    def _init_client(self):
        """Initialize the provider client"""
        pass
    
    @abstractmethod
    async def analyze_vision(self, request: VisionRequest) -> AIResponse:
        """Analyze images with vision model"""
        pass
    
    @abstractmethod
    async def analyze_text(self, request: TextRequest) -> AIResponse:
        """Analyze text with text model"""
        pass
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')


class OpenAIProvider(BaseAIProvider):
    """OpenAI GPT-4o Vision provider"""
    
    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info("Initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    async def analyze_vision(self, request: VisionRequest) -> AIResponse:
        if not self.client:
            return AIResponse(
                content="", 
                provider=AIProvider.OPENAI,
                model=self.config.model_vision or "gpt-4o",
                success=False,
                error="Client not initialized"
            )
        
        try:
            # Prepare image messages
            content = []
            for img_path in request.images[:8]:  # Max 8 images
                base64_image = self._encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto"
                    }
                })
            
            content.append({"type": "text", "text": request.prompt})
            
            messages = [{"role": "user", "content": content}]
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            response_format = {"type": "json_object"} if request.response_format == "json" else None
            
            response = await self.client.chat.completions.create(
                model=self.config.model_vision or "gpt-4o",
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                response_format=response_format
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider=AIProvider.OPENAI,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenAI vision analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.OPENAI,
                model=self.config.model_vision or "gpt-4o",
                success=False,
                error=str(e)
            )
    
    async def analyze_text(self, request: TextRequest) -> AIResponse:
        if not self.client:
            return AIResponse(
                content="",
                provider=AIProvider.OPENAI,
                model=self.config.model_text or "gpt-4o",
                success=False,
                error="Client not initialized"
            )
        
        try:
            messages = [{"role": "user", "content": request.prompt}]
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            response_format = {"type": "json_object"} if request.response_format == "json" else None
            
            response = await self.client.chat.completions.create(
                model=self.config.model_text or "gpt-4o",
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                response_format=response_format
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider=AIProvider.OPENAI,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenAI text analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.OPENAI,
                model=self.config.model_text or "gpt-4o",
                success=False,
                error=str(e)
            )


class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude 3 provider"""
    
    def _init_client(self):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info("Initialized Anthropic client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None
    
    async def analyze_vision(self, request: VisionRequest) -> AIResponse:
        if not self.client:
            return AIResponse(
                content="",
                provider=AIProvider.ANTHROPIC,
                model=self.config.model_vision or "claude-3-opus-20240229",
                success=False,
                error="Client not initialized"
            )
        
        try:
            # Prepare content with images
            content = []
            for img_path in request.images[:8]:
                base64_image = self._encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            
            content.append({"type": "text", "text": request.prompt})
            
            response = await self.client.messages.create(
                model=self.config.model_vision or "claude-3-opus-20240229",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt or "",
                messages=[{"role": "user", "content": content}]
            )
            
            # Extract text content
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
            
            return AIResponse(
                content=text_content,
                provider=AIProvider.ANTHROPIC,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Anthropic vision analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.ANTHROPIC,
                model=self.config.model_vision or "claude-3-opus-20240229",
                success=False,
                error=str(e)
            )
    
    async def analyze_text(self, request: TextRequest) -> AIResponse:
        if not self.client:
            return AIResponse(
                content="",
                provider=AIProvider.ANTHROPIC,
                model=self.config.model_text or "claude-3-sonnet-20240229",
                success=False,
                error="Client not initialized"
            )
        
        try:
            response = await self.client.messages.create(
                model=self.config.model_text or "claude-3-sonnet-20240229",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt or "",
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
            
            return AIResponse(
                content=text_content,
                provider=AIProvider.ANTHROPIC,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Anthropic text analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.ANTHROPIC,
                model=self.config.model_text or "claude-3-sonnet-20240229",
                success=False,
                error=str(e)
            )


class GeminiProvider(BaseAIProvider):
    """Google Gemini Pro Vision provider"""
    
    def _init_client(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self.vision_model = genai.GenerativeModel(self.config.model_vision or "gemini-pro-vision")
            self.text_model = genai.GenerativeModel(self.config.model_text or "gemini-pro")
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.vision_model = None
            self.text_model = None
    
    async def analyze_vision(self, request: VisionRequest) -> AIResponse:
        if not self.vision_model:
            return AIResponse(
                content="",
                provider=AIProvider.GEMINI,
                model=self.config.model_vision or "gemini-pro-vision",
                success=False,
                error="Client not initialized"
            )
        
        try:
            from PIL import Image
            
            # Load images
            images = []
            for img_path in request.images[:8]:
                images.append(Image.open(img_path))
            
            # Prepare content
            content = images + [request.prompt]
            
            response = self.vision_model.generate_content(content)
            
            return AIResponse(
                content=response.text,
                provider=AIProvider.GEMINI,
                model=self.config.model_vision or "gemini-pro-vision",
                usage={},  # Gemini doesn't provide usage stats
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Gemini vision analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.GEMINI,
                model=self.config.model_vision or "gemini-pro-vision",
                success=False,
                error=str(e)
            )
    
    async def analyze_text(self, request: TextRequest) -> AIResponse:
        if not self.text_model:
            return AIResponse(
                content="",
                provider=AIProvider.GEMINI,
                model=self.config.model_text or "gemini-pro",
                success=False,
                error="Client not initialized"
            )
        
        try:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}" if request.system_prompt else request.prompt
            
            response = self.text_model.generate_content(full_prompt)
            
            return AIResponse(
                content=response.text,
                provider=AIProvider.GEMINI,
                model=self.config.model_text or "gemini-pro",
                usage={},
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Gemini text analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.GEMINI,
                model=self.config.model_text or "gemini-pro",
                success=False,
                error=str(e)
            )


class OllamaProvider(BaseAIProvider):
    """Ollama local model provider"""
    
    def _init_client(self):
        try:
            import aiohttp
            self.base_url = self.config.base_url or "http://localhost:11434"
            self.model = self.config.model_vision or "llava"
            logger.info(f"Initialized Ollama client at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.base_url = None
    
    async def analyze_vision(self, request: VisionRequest) -> AIResponse:
        if not self.base_url:
            return AIResponse(
                content="",
                provider=AIProvider.OLLAMA,
                model=self.model,
                success=False,
                error="Client not initialized"
            )
        
        try:
            import aiohttp
            
            # Ollama supports one image per request for vision
            images_b64 = [self._encode_image(img) for img in request.images[:1]]
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": request.prompt,
                    "images": images_b64,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    result = await response.json()
                    
                    return AIResponse(
                        content=result.get("response", ""),
                        provider=AIProvider.OLLAMA,
                        model=self.model,
                        usage={},
                        raw_response=result
                    )
                    
        except Exception as e:
            logger.error(f"Ollama vision analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.OLLAMA,
                model=self.model,
                success=False,
                error=str(e)
            )
    
    async def analyze_text(self, request: TextRequest) -> AIResponse:
        if not self.base_url:
            return AIResponse(
                content="",
                provider=AIProvider.OLLAMA,
                model=self.model,
                success=False,
                error="Client not initialized"
            )
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}" if request.system_prompt else request.prompt
                
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    result = await response.json()
                    
                    return AIResponse(
                        content=result.get("response", ""),
                        provider=AIProvider.OLLAMA,
                        model=self.model,
                        usage={},
                        raw_response=result
                    )
                    
        except Exception as e:
            logger.error(f"Ollama text analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.OLLAMA,
                model=self.model,
                success=False,
                error=str(e)
            )


class LMStudioProvider(BaseAIProvider):
    """LM Studio local model provider"""
    
    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                base_url=self.config.base_url or "http://localhost:1234/v1",
                api_key="not-needed"
            )
            self.model = self.config.model_vision or "local-model"
            logger.info(f"Initialized LM Studio client at {self.config.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize LM Studio client: {e}")
            self.client = None
    
    async def analyze_vision(self, request: VisionRequest) -> AIResponse:
        # LM Studio vision support varies by model
        return AIResponse(
            content="LM Studio vision not supported - use text only",
            provider=AIProvider.LM_STUDIO,
            model=self.model,
            success=False,
            error="Vision not supported"
        )
    
    async def analyze_text(self, request: TextRequest) -> AIResponse:
        if not self.client:
            return AIResponse(
                content="",
                provider=AIProvider.LM_STUDIO,
                model=self.model,
                success=False,
                error="Client not initialized"
            )
        
        try:
            messages = [{"role": "user", "content": request.prompt}]
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider=AIProvider.LM_STUDIO,
                model=self.model,
                usage={},
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"LM Studio text analysis error: {e}")
            return AIResponse(
                content="",
                provider=AIProvider.LM_STUDIO,
                model=self.model,
                success=False,
                error=str(e)
            )


class MultiProviderAI:
    """Multi-provider AI manager with automatic fallback"""
    
    PROVIDER_MAP = {
        AIProvider.OPENAI: OpenAIProvider,
        AIProvider.ANTHROPIC: AnthropicProvider,
        AIProvider.GEMINI: GeminiProvider,
        AIProvider.OLLAMA: OllamaProvider,
        AIProvider.LM_STUDIO: LMStudioProvider,
    }
    
    def __init__(self):
        self._providers: Dict[AIProvider, BaseAIProvider] = {}
        self._init_providers()
    
    def _init_providers(self):
        """Initialize all configured providers"""
        for provider_type in AIProvider:
            try:
                provider_config = get_provider_config(provider_type)
                if provider_config.api_key or provider_type in (AIProvider.OLLAMA, AIProvider.LM_STUDIO):
                    provider_class = self.PROVIDER_MAP.get(provider_type)
                    if provider_class:
                        self._providers[provider_type] = provider_class(provider_config)
                        logger.info(f"Initialized provider: {provider_type.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {provider_type}: {e}")
    
    def get_provider(self, provider_type: AIProvider) -> Optional[BaseAIProvider]:
        """Get a specific provider instance"""
        return self._providers.get(provider_type)
    
    async def analyze_with_fallback(
        self,
        task_type: TaskType,
        vision_request: Optional[VisionRequest] = None,
        text_request: Optional[TextRequest] = None
    ) -> AIResponse:
        """Analyze with automatic provider fallback"""
        providers = get_providers_for_task(task_type)
        
        last_error = None
        for provider_type in providers:
            provider = self._providers.get(provider_type)
            if not provider:
                continue
            
            try:
                logger.info(f"Trying provider {provider_type.value} for task {task_type.value}")
                
                if vision_request:
                    response = await provider.analyze_vision(vision_request)
                elif text_request:
                    response = await provider.analyze_text(text_request)
                else:
                    return AIResponse(
                        content="",
                        provider=provider_type,
                        success=False,
                        error="No request provided"
                    )
                
                if response.success:
                    logger.info(f"Successfully used provider {provider_type.value}")
                    return response
                else:
                    last_error = response.error
                    logger.warning(f"Provider {provider_type.value} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Provider {provider_type.value} error: {e}")
        
        return AIResponse(
            content="",
            provider=AIProvider.OPENAI,
            success=False,
            error=f"All providers failed. Last error: {last_error}"
        )
    
    async def analyze_vision(
        self,
        images: List[Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.SCENE_ANALYSIS,
        response_format: Optional[str] = None,
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE
    ) -> AIResponse:
        """Analyze images with automatic fallback"""
        request = VisionRequest(
            images=images,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format
        )
        return await self.analyze_with_fallback(task_type, vision_request=request)
    
    async def analyze_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.SUMMARIZATION,
        response_format: Optional[str] = None,
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE
    ) -> AIResponse:
        """Analyze text with automatic fallback"""
        request = TextRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format
        )
        return await self.analyze_with_fallback(task_type, text_request=request)


# Global multi-provider AI instance
_multi_provider_ai: Optional[MultiProviderAI] = None


def get_multi_provider_ai() -> MultiProviderAI:
    """Get the global multi-provider AI instance"""
    global _multi_provider_ai
    if _multi_provider_ai is None:
        _multi_provider_ai = MultiProviderAI()
    return _multi_provider_ai
