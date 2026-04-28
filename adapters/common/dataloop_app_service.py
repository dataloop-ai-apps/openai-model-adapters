"""
Resolve Dataloop app-service routes and use JWT-APP cookie auth for
OpenAI-compatible services (e.g. ollama-server on Dataloop).
Health check uses OpenAI GET /v1/models.
"""

from __future__ import annotations

import datetime
import logging
import os

import dtlpy as dl
import httpx
import jwt
import openai
import requests

SSL_VERIFY = os.environ.get("DATALOOP_SSL_VERIFY", "true").lower() not in ("0", "false", "no")

_log = logging.getLogger("openai-adapter")


def _strip_bearer(request: httpx.Request):
    """Service authenticates via JWT-APP cookie; strip the dummy Bearer token."""
    request.headers.pop("authorization", None)


def resolve_app_service_endpoint(app_id: str):
    """
    Resolve a Dataloop app-service route by following the gateway redirect
    chain to discover the real service URL and capture the JWT-APP cookie.

    Returns:
        (base_url, session): base_url ends with /v1; session is requests.Session.
    """
    app = dl.apps.get(app_id=app_id)
    route = list(app.routes.values())[0].rstrip("/")
    base_before = "/".join(route.split("/")[:-1])
    session = requests.Session()
    resp = session.get(f"{base_before}/models", headers=dl.client_api.auth, verify=SSL_VERIFY)
    _log.debug("Redirect chain resolved to: %s (cookies: %s)", resp.url, dict(session.cookies))
    base_url = resp.url.split("/models")[0]
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url, session


class DataloopAppServiceClient:
    """
    OpenAI client + session for a Dataloop app-service (cookie-only auth).
    Call check_jwt_expiration() before batched or long inference.
    Includes native warmup to pre-load models before the first /v1 call.
    """

    def __init__(self, app_id: str, model_entity, log: logging.Logger, timeout: int = 900) -> None:
        self.app_id = app_id
        self.model_entity = model_entity
        self._log = log
        self.timeout = timeout
        self.base_url = None
        self.current_session: requests.Session | None = None
        self.client: openai.OpenAI | None = None
        self._rebuild_client()

    def _post_native(self, path: str, payload: dict):
        origin = self.base_url.removesuffix("/v1").rstrip("/")
        return self.current_session.post(
            f"{origin}{path}",
            json=payload,
            verify=SSL_VERIFY,
            timeout=self.timeout,
        )

    def warmup_native_chat(self, model: str) -> None:
        """POST /api/generate before /v1/chat/completions — loads model if evicted, resets keep_alive."""
        if self.current_session is None:
            return
        keep_alive = self.model_entity.configuration.get("keep_alive", "10m")
        payload = {
            "model": model,
            "prompt": "",
            "keep_alive": keep_alive,
            "stream": False,
        }
        try:
            self._log.info(
                "Hosted warmup (chat): POST /api/generate model=%s keep_alive=%s",
                model,
                keep_alive,
            )
            self._post_native("/api/generate", payload).raise_for_status()
        except Exception as e:
            self._log.warning("Hosted native chat warmup failed (non-fatal): %s", e)

    def warmup_native_embed(self, model: str) -> None:
        """POST /api/embeddings before /v1/embeddings — loads model if evicted, resets keep_alive."""
        if self.current_session is None:
            return
        try:
            self._log.info("Hosted warmup (embed): POST /api/embeddings model=%s", model)
            self._post_native(
                "/api/embeddings",
                {"model": model, "input": "."},
            ).raise_for_status()
        except Exception as e:
            self._log.warning("Hosted native embed warmup failed (non-fatal): %s", e)

    def _rebuild_client(self) -> None:
        self.base_url, self.current_session = resolve_app_service_endpoint(
            self.app_id
        )
        cookie_header = "; ".join(
            f"{c.name}={c.value}" for c in self.current_session.cookies
        )
        self._log.info("Using app-service endpoint: %s", self.base_url)
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="unused",
            default_headers={"Cookie": cookie_header},
            http_client=httpx.Client(
                verify=SSL_VERIFY,
                timeout=httpx.Timeout(connect=60.0, read=float(self.timeout), write=60.0, pool=60.0),
                event_hooks={"request": [_strip_bearer]},
            ),
        )
        self.model_entity.configuration["base_url"] = self.base_url
        self.model_entity.update()
        try:
            self.client.models.list()
        except Exception as e:
            self._log.error("App-service health check (models.list) failed: %s", e)
            raise ValueError(
                f"Failed to reach app-service OpenAI API at {self.base_url}: {e}"
            ) from e
        self._log.info("App-service ready at %s", self.base_url)

    def check_jwt_expiration(self, margin_seconds: int = 60) -> None:
        if self.current_session is None:
            self._rebuild_client()
            return
        token = self.current_session.cookies.get("JWT-APP")
        if not token:
            self._log.warning("No JWT-APP cookie found, refreshing session")
            self._rebuild_client()
            return
        decoded = jwt.decode(token, options={"verify_signature": False})
        exp_timestamp = decoded.get("exp")
        if not exp_timestamp:
            self._log.warning("No 'exp' claim in JWT, refreshing session")
            self._rebuild_client()
            return
        exp_dt = datetime.datetime.fromtimestamp(exp_timestamp)
        now = datetime.datetime.now()
        remaining = exp_dt - now
        if now >= exp_dt - datetime.timedelta(seconds=margin_seconds):
            self._log.info(
                "JWT expired or expiring soon (remaining: %s). Refreshing session.",
                remaining,
            )
            self._rebuild_client()
        else:
            self._log.debug("JWT still valid (remaining: %s)", remaining)
