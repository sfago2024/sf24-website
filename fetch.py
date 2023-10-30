import json
import logging
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session

ACCESS_TOKEN_BUFFER = timedelta(seconds=30)
API_URL = "https://api-platform.cvent.com/ea"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    client_id: str
    client_secret: str

    @classmethod
    def load(cls, repodir: Path) -> Self:
        with (repodir / "config.toml").open("rb") as f:
            config = tomllib.load(f)
        return cls(**config)


def load_or_fetch_token(repodir: Path, config: Config) -> str:
    try:
        with (repodir / "token.json").open() as f:
            token = json.load(f)
    except FileNotFoundError:
        pass
    else:
        expires_in = datetime.fromtimestamp(token["expires_at"]) - datetime.now()
        if expires_in > ACCESS_TOKEN_BUFFER:
            logging.debug("Cached auth token expires in %s", expires_in)
            return token["access_token"]
    logging.debug("Fetching a new auth token...")
    auth = HTTPBasicAuth(config.client_id, config.client_secret)
    client = BackendApplicationClient(config.client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(f"{API_URL}/oauth2/token", auth=auth)
    if (token_type := token.pop("token_type")) != "Bearer":
        raise RuntimeError(f"Got non-'Bearer' value for token_type: {token_type}")
    del token["expires_in"]
    with (repodir / "token.json").open("w") as f:
        json.dump(token, f)
    return token["access_token"]


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    repodir = Path(__file__).parent
    config = Config.load(repodir)
    token = load_or_fetch_token(repodir, config)


if __name__ == "__main__":
    main()
