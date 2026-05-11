import os
import sys
import json
import time
import re
from typing import List
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException



class ApiKeyManager:
    def __init__(self, required_keys):
        self.api_keys = {}
        self.required_keys = required_keys
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))


        for key in self.required_keys:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.required_keys if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class GoogleGenAIEmbeddings:
    def __init__(self, model: str, api_key: str):
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.max_batch_size = 100
        self.min_interval_seconds = 0.7

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            if embeddings:
                time.sleep(self.min_interval_seconds)
            batch = texts[start : start + self.max_batch_size]
            for attempt in range(3):
                try:
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                    embeddings.extend([embedding.values for embedding in result.embeddings])
                    break
                except Exception as e:
                    from google.genai import errors

                    if isinstance(e, errors.ClientError) and "RESOURCE_EXHAUSTED" in str(e):
                        delay_seconds = 60
                        match = re.search(r"retryDelay'?:\s*'?(\d+)s", str(e))
                        if match:
                            delay_seconds = int(match.group(1)) + 1
                        log.warning("Embedding rate limited, backing off", delay_seconds=delay_seconds)
                        time.sleep(delay_seconds)
                        continue
                    raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text],
        )
        return result.embeddings[0].values


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager(self._required_keys())
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def _required_keys(self):
        llm_block = load_config().get("llm", {})
        provider_key = os.getenv("LLM_PROVIDER") or (next(iter(llm_block), None))
        if not provider_key:
            return []
        llm_provider = llm_block.get(provider_key, {}).get("provider", provider_key)

        required = []
        if llm_provider == "google":
            required.append("GOOGLE_API_KEY")
        elif llm_provider == "groq":
            required.append("GROQ_API_KEY")
        return required


    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            model_cfg = self.config["embedding_model"]
            provider = model_cfg.get("provider", "google")
            model_name = model_cfg["model_name"]
            log.info("Loading embedding model", provider=provider, model=model_name)

            if provider == "local":
                return HuggingFaceEmbeddings(model_name=model_name)

            return GoogleGenAIEmbeddings(
                model=model_name,
                api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER") or next(iter(llm_block))

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )

        elif provider == "local":
            return Ollama(
                model=model_name,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temperature,
            )

        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")