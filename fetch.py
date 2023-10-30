import asyncio
import json
import logging
import re
import shutil
import sys
import tomllib
from argparse import ArgumentParser
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent
from time import sleep
from typing import Any, AsyncIterator, Self
from zoneinfo import ZoneInfo

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests import PreparedRequest, Response
from requests.auth import AuthBase, HTTPBasicAuth
from requests_oauthlib import OAuth2Session

ACCESS_TOKEN_BUFFER = timedelta(seconds=30)
API_URL = "https://api-platform.cvent.com/ea"
TIMEZONE = ZoneInfo("America/Los_Angeles")

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
        response = requests.get(
            f"{API_URL}{url}",
            params=params,
            auth=self.auth,
            headers={"Accept": "application/json"},
        )
        (
            Path.home()
            / "cvent-backups"
            / f"{url.replace('/', '')} {datetime.now().isoformat()}.json"
        ).write_text(response.text)
        return response


SLUG_REPLACE_PATTERN: re.Pattern[str] = re.compile(r"[^\w\d]+")


def slugify(s: str) -> str:
    return SLUG_REPLACE_PATTERN.sub("-", s.casefold()).strip("-")


@dataclass(frozen=True)
class Session:
    id: str
    name: str
    description: str | None
    start_time: datetime
    end_time: datetime
    full_description: str | None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            name=data["title"],
            description=data.get("description"),
            start_time=datetime.fromisoformat(data["start"]).astimezone(TIMEZONE),
            end_time=datetime.fromisoformat(data["end"]).astimezone(TIMEZONE),
            full_description=next(
                (
                    d
                    for d in data["customFields"]
                    if d["name"] == "SF24 Full Description"
                ),
                {"value": None},
            ).get("value"),
        )

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def url_relpath(self) -> str:
        return f"sessions/{self.slug}/"

    def link(self, base_url: str) -> str:
        return f'<a href="{base_url}{self.url_relpath}">{self.name}</a>'

    def page_content(self) -> str:
        page = dedent(
            """\
            <h2>Date/Time</h2>
            <p>{self.start_time:%A, %B %d, %Y}<br>
            {self.start_time:%I:%M %p} – {self.end_time:%I:%M %p}</p>
            """
        ).format(**locals())
        if desc := self.full_description or self.description:
            page += dedent(
                """\
                <h2>Description</h2>

                {desc}

                """
            ).format(**locals())
        return page


def load_all_sessions(cvent: Cvent) -> dict[str, Session]:
    sessions: dict[str, Session] = {}
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
        sessions.update({d["id"]: Session.from_json(d) for d in data["data"]})
        if (token := data["paging"].get("nextToken")) is not None:
            token_param = {"token": token}
        else:
            break
    return sessions


class SpeakerType(Enum):
    COMPOSER = auto()
    PERFORMER = auto()
    PRESENTER = auto()


@dataclass(frozen=True)
class Speaker:
    id: str
    first_name: str
    last_name: str
    bio: str | None
    photo_url: str | None
    types: set[SpeakerType]

    @classmethod
    def from_json(cls, data: dict[str, Any], sessions: dict[str, Session]) -> Self:
        for session in sessions.values():
            ...
        return cls(
            id=data["id"],
            first_name=data["firstName"],
            last_name=data["lastName"],
            bio=data.get("biography"),
            photo_url=data["links"].get("profilePicture", {"href": None})["href"],
            types=set(),
        )

    @property
    def name(self) -> str:
        if self.first_name.strip("\N{ZERO WIDTH SPACE}"):
            return f"{self.first_name} {self.last_name}"
        else:
            return self.last_name

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def url_relpath(self) -> str | None:
        if SpeakerType.PERFORMER in self.types:
            return f"performers/{self.slug}/"
        elif SpeakerType.COMPOSER in self.types:
            return f"composers/{self.slug}/"
        elif SpeakerType.PRESENTER in self.types:
            return f"presenters/{self.slug}/"
        return None

    def link(self, base_url: str) -> str:
        return f'<a href="{base_url}{self.url_relpath}">{self.name}</a>'

    def page_content(self) -> str:
        page = ""
        if self.photo_url:
            page += dedent(
                """\
                <img src="{self.photo_url}">
                """
            ).format(**locals())
        if self.bio:
            page += dedent(
                """\
                <h2>Biography</h2>
                <p>{self.bio}</p>
                """
            ).format(**locals())
        return page


def load_all_speakers(cvent: Cvent, sessions: dict[str, Session]) -> dict[str, Speaker]:
    speakers: dict[str, Speaker] = {}
    token_param: dict[str, str] = {}
    while True:
        sleep(1)
        r = cvent.get(
            "/speakers",
            params={
                "filter": f"event.id eq '{cvent.event_id}'",
                "limit": 200,
                **token_param,
            },
        )
        data = r.json()
        speakers.update({d["id"]: Speaker.from_json(d, sessions) for d in data["data"]})
        if (token := data["paging"].get("nextToken")) is not None:
            token_param = {"token": token}
        else:
            break
    return speakers


def index_page(title: str, links: list[str]) -> str:
    return "\n".join(
        [
            "<ul>",
            *(f"<li>{link}</li>" for link in links),
            "</ul>",
        ]
    )


def render_page(url: str, title: str, content: str):
    date = datetime.now()
    return dedent(
        """\
        +++
        title = '''{title}'''
        path = '''{url}'''
        template = "future.html"
        +++

        <h1>{title}</h1>

        {content}
        """
    ).format(**locals())


@asynccontextmanager
async def manage_repo(repo_dir: Path, commit: bool) -> AsyncIterator[None]:
    proc = await create_subprocess_exec(
        "git", "status", "--porcelain", cwd=repo_dir, stdout=PIPE
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"'git status' exited {proc.returncode}")
    if stdout:
        items = stdout.decode("utf-8", errors="replace").splitlines()
        if commit:
            raise RuntimeError(f"repo is not clean ({items})")
        else:
            logger.warning("repo is not clean (%s)", items)
    if commit:
        proc = await create_subprocess_exec("git", "pull", "--ff-only", cwd=repo_dir)
        if (returncode := await proc.wait()) != 0:
            raise RuntimeError(f"'git pull' exited {returncode}")
    try:
        yield
    except:
        # If we're in commit mode, clean up any mess
        if commit:
            proc = await create_subprocess_exec(
                "git", "status", "--porcelain", cwd=repo_dir, stdout=PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"'git status' exited {proc.returncode}")
            if stdout:
                logger.warn("Repo is dirty, cleaning it...")
                proc = await create_subprocess_exec("git", "stash", "-u", cwd=repo_dir)
                if (returncode := await proc.wait()) != 0:
                    raise RuntimeError(f"'git stash' exited {returncode}")
                proc = await create_subprocess_exec(
                    "git", "stash", "drop", cwd=repo_dir
                )
                await proc.wait()
        raise
    else:
        # If we're in commit mode, build and make a commit!
        if commit:
            proc = await create_subprocess_exec("zola", "build", cwd=repo_dir)
            if (returncode := await proc.wait()) != 0:
                raise RuntimeError(f"'zola build' exited {returncode}")
            proc = await create_subprocess_exec("git", "add", ".", cwd=repo_dir)
            if (returncode := await proc.wait()) != 0:
                raise RuntimeError(f"'git add' exited {returncode}")
            proc = await create_subprocess_exec(
                "git",
                "commit",
                "--author=cvent-webhook-handler <colin+cventwebhookhandler@lumeh.org>",
                "--message=cvent-webhook-handler update",
                cwd=repo_dir,
            )
            if (returncode := await proc.wait()) != 0:
                raise RuntimeError(f"'git commit' exited {returncode}")
            proc = await create_subprocess_exec("git", "push", cwd=repo_dir)
            if (returncode := await proc.wait()) != 0:
                raise RuntimeError(f"'git push' exited {returncode}")


async def generate_pages(
    base_url: str,
    repo_dir: Path,
    commit: bool,
    sessions: dict[str, Session],
    speakers: dict[str, Speaker],
) -> None:
    async with manage_repo(repo_dir, commit):
        outdir = repo_dir / "content/_generated"
        if outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir()

        (outdir / "sessions").mkdir()
        for session in sessions.values():
            path = outdir / (session.url_relpath.removesuffix("/") + ".md")
            if path.exists():
                logger.warn("Overwriting duplicate %s", session.url_relpath)
            path.write_text(
                render_page(
                    base_url + session.url_relpath, session.name, session.page_content()
                )
            )
        session_links = [s.link(base_url) for s in sessions.values()]
        (outdir / "sessions/index.md").write_text(
            render_page(
                base_url + "sessions/",
                "Sessions",
                index_page("Sessions", session_links),
            )
        )

        (outdir / "composers").mkdir()
        (outdir / "performers").mkdir()
        (outdir / "presenters").mkdir()
        for speaker in speakers.values():
            if (url_relpath := speaker.url_relpath) is not None:
                path = outdir / (url_relpath.removesuffix("/") + ".md")
                if path.exists():
                    logger.warn("Overwriting duplicate %s", url_relpath)
                path.write_text(
                    render_page(
                        base_url + url_relpath, speaker.name, speaker.page_content()
                    )
                )
        composer_links = [
            s.link(base_url)
            for s in speakers.values()
            if SpeakerType.COMPOSER in s.types
        ]
        performer_links = [
            s.link(base_url)
            for s in speakers.values()
            if SpeakerType.PERFORMER in s.types
        ]
        presenter_links = [
            s.link(base_url)
            for s in speakers.values()
            if SpeakerType.PRESENTER in s.types
        ]
        (outdir / "composers/index.md").write_text(
            render_page(
                base_url + "composers/",
                "Composers",
                index_page("Composers", composer_links),
            )
        )
        (outdir / "performers/index.md").write_text(
            render_page(
                base_url + "performers/",
                "Performers",
                index_page("Performers", performer_links),
            )
        )
        (outdir / "presenters/index.md").write_text(
            render_page(
                base_url + "presenters/",
                "Presenters",
                index_page("Presenters", presenter_links),
            )
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repodir = Path(__file__).parent
    config = Config.load(repodir)
    auth = Auth.load_or_fetch_token(repodir, config)
    cvent = Cvent(auth, config.event_id)

    sessions = load_all_sessions(cvent)
    speakers = load_all_speakers(cvent, sessions)
    asyncio.run(
        generate_pages(args.base_url, args.repo_dir, args.commit, sessions, speakers)
    )


if __name__ == "__main__":
    main()
