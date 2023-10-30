import json
import logging
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any, Self

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests import PreparedRequest, Response
from requests.auth import AuthBase, HTTPBasicAuth
from requests_oauthlib import OAuth2Session

ACCESS_TOKEN_BUFFER = timedelta(seconds=30)
API_URL = "https://api-platform.cvent.com/ea"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    client_id: str
    client_secret: str
    event_id: str

    @classmethod
    def load(cls, repodir: Path) -> Self:
        with (repodir / "config.toml").open("rb") as f:
            config = tomllib.load(f)
        return cls(**config)


@dataclass(frozen=True)
class Auth(AuthBase):
    access_token: str

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        return request

    @classmethod
    def load_or_fetch_token(cls, repodir: Path, config: Config) -> Self:
        try:
            with (repodir / "token.json").open() as f:
                token = json.load(f)
        except FileNotFoundError:
            pass
        else:
            expires_in = datetime.fromtimestamp(token["expires_at"]) - datetime.now()
            if expires_in > ACCESS_TOKEN_BUFFER:
                logging.info("Cached auth token expires in %s", expires_in)
                return cls(token["access_token"])
        logging.info("Fetching a new auth token...")
        auth = HTTPBasicAuth(config.client_id, config.client_secret)
        client = BackendApplicationClient(config.client_id)
        oauth = OAuth2Session(client=client)
        token = oauth.fetch_token(f"{API_URL}/oauth2/token", auth=auth)
        if (token_type := token.pop("token_type")) != "Bearer":
            raise RuntimeError(f"Got non-'Bearer' value for token_type: {token_type}")
        del token["expires_in"]
        with (repodir / "token.json").open("w") as f:
            json.dump(token, f)
        return cls(token["access_token"])


@dataclass(frozen=True)
class Cvent:
    auth: Auth
    event_id: str

    def get(self, url: str, params: dict[str, Any]) -> Response:
        logger.info("GET %s with params %s", url, params)
        return requests.get(
            f"{API_URL}{url}",
            params=params,
            auth=self.auth,
            headers={"Accept": "application/json"},
        )


def load_all_sessions(cvent: Cvent) -> list[Any]:
    sessions = []
    token_param: dict[str, str] = {}
    while True:
        sleep(1)
        r = cvent.get(
            "/sessions",
            params={
                "filter": f"event.id eq '{cvent.event_id}'",
                "limit": 200,
                **token_param,
            },
        )
        data = r.json()
        sessions.extend(data["data"])
        if (token := data["paging"].get("nextToken")) is not None:
            token_param = {"token": token}
        else:
            break
    return sessions


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repodir = Path(__file__).parent
    config = Config.load(repodir)
    auth = Auth.load_or_fetch_token(repodir, config)
    cvent = Cvent(auth, config.event_id)

    sessions = load_all_sessions(cvent)
    json.dump(sessions, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
