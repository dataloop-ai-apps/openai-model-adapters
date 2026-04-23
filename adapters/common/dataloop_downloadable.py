"""
Resolve Dataloop app routes and use JWT-APP cookie auth for OpenAI-compatible
downloadable services (e.g. ollama-server on Dataloop). Health check uses
OpenAI GET /v1/models (Ollama does not expose NIM /manifest).
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


def get_downloadable_endpoint_and_cookie(app_id: str):
    """
    Resolve Dataloop app route and obtain JWT-APP cookie via redirect.

    Returns:
        (base_url, session): base_url ends with /v1; session is requests.Session.
    """
    app = dl.apps.get(app_id=app_id)
    route = list(app.routes.values())[0].rstrip("/")
    base_before = "/".join(route.split("/")[:-1])
    session = requests.Session()
    resp = session.get(base_before, headers=dl.client_api.auth, verify=SSL_VERIFY)
    base_url = resp.url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url, session


class DataloopDownloadableContext:
    """
    OpenAI client + session for a Dataloop app_id (cookie-only auth).
    Call check_jwt_expiration() before batched or long inference.
    """

    def __init__(self, app_id: str, model_entity, log: logging.Logger) -> None:
        self.app_id = app_id
        self.model_entity = model_entity
        self._log = log
        self.base_url = None
        self.current_session: requests.Session | None = None
        self.client: openai.OpenAI | None = None
        self._rebuild_client()

    def _cookie_header(self) -> str:
        if self.current_session is None:
            return ""
        return "; ".join(
            f"{c.name}={c.value}" for c in self.current_session.cookies
        )

    def _rebuild_client(self) -> None:
        self.base_url, self.current_session = get_downloadable_endpoint_and_cookie(
            self.app_id
        )
        cookie_header = self._cookie_header()
        self._log.info("Using downloadable endpoint, base URL: %s", self.base_url)
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="",
            default_headers={"Cookie": cookie_header},
            http_client=httpx.Client(verify=SSL_VERIFY),
        )
        self.model_entity.configuration["base_url"] = self.base_url
        self.model_entity.update()
        try:
            self.client.models.list()
        except Exception as e:
            self._log.error("Downloadable health check (models.list) failed: %s", e)
            raise ValueError(
                f"Failed to reach downloadable OpenAI API at {self.base_url}: {e}"
            ) from e
        self._log.info("Downloadable endpoint ready at %s", self.base_url)

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
