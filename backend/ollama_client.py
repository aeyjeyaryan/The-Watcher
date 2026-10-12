import httpx


class OllamaError(Exception):
    pass


class OllamaUnavailable(OllamaError):
    pass


class OllamaClient:
    def __init__(self, host, request_timeout):
        self._host = host.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(request_timeout),
        )

    async def close(self):
        await self._client.aclose()

    async def ping(self):
        try:
            response = await self._client.get("/api/version", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self):
        try:
            response = await self._client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Cannot reach Ollama at {self._host}: {exc}") from exc
        data = response.json()
        return sorted(model.get("name", "") for model in data.get("models", []))

    async def chat(
        self,
        model,
        user,
        system=None,
        images=None,
        temperature=0.2,
        num_predict=128,
        seed=None,
        response_format=None,
        timeout=120.0,
        keep_alive="5m",
    ):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        message = {"role": "user", "content": user}
        if images:
            message["images"] = images
        messages.append(message)
        options = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format
        try:
            response = await self._client.post("/api/chat", json=payload, timeout=timeout)
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self._host}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(f"Model `{model}` timed out after {timeout:.0f}s.") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc
        if response.status_code != 200:
            detail = response.text[:300]
            raise OllamaError(f"Ollama returned {response.status_code}: {detail}")
        data = response.json()
        return (data.get("message", {}) or {}).get("content", "").strip()
