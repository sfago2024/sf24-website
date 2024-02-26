from __future__ import annotations

import asyncio
import functools
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
from itertools import chain
from operator import attrgetter
from pathlib import Path
from textwrap import dedent
from time import sleep
from typing import Any, AsyncIterator, Callable, Self, TypeVar, cast
from zoneinfo import ZoneInfo

import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests import PreparedRequest, Response
from requests.auth import AuthBase, HTTPBasicAuth
from requests.exceptions import HTTPError
from requests_oauthlib import OAuth2Session
from unidecode import unidecode

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


T = TypeVar("T", bound=Callable)


def log_response_errors(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            logger.debug(
                "Caught HTTPError. Response content: %s",
                e.response.content.decode("utf-8", errors="replace"),
            )
            raise

    return cast(T, wrapper)


@dataclass(frozen=True)
class Cvent:
    auth: Auth
    event_id: str

    @log_response_errors
    def get(self, url: str, params: dict[str, Any]) -> Response:
        sleep(1)
        logger.info("GET %s with params %s", url, params)
        response = requests.get(
            f"{API_URL}{url}",
            params=params,
            auth=self.auth,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        (
            Path.home()
            / "cvent-backups"
            / f"{url.replace('/', '')} {datetime.now().isoformat()}.json"
        ).write_text(response.text)
        return response

    @log_response_errors
    def put(self, url: str, data: dict[str, Any]) -> Response:
        sleep(1)
        logger.info("PUT %s with data %s", url, data)
        response = requests.put(
            f"{API_URL}{url}",
            json=data,
            auth=self.auth,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response


SLUG_REPLACE_PATTERN: re.Pattern[str] = re.compile(r"[^\w\d]+")


def slugify(s: str) -> str:
    return SLUG_REPLACE_PATTERN.sub("-", s.casefold()).strip("-")


READ_MORE_PATTERN: re.Pattern[str] = re.compile(r"<a .*Read more</a>")


def update_session_read_more_link(cvent: Cvent, session: Session) -> None:
    if session.description is None:
        logger.warning("Skipping session %s with None description", session.slug)
        return
    new_read_more = f'<a href="https://www.sfago2024.org/future/{session.url_relpath}" target="_blank">Read more</a>'
    if match := READ_MORE_PATTERN.search(session.description):
        description = READ_MORE_PATTERN.sub(new_read_more, session.description)
    else:
        description = session.description + " " + new_read_more
    if description != session.description:
        new_data = session.data.copy()
        new_data["description"] = description
        if new_data["waitlistCapacity"] == -1:
            del new_data["waitlistCapacity"]
        cvent.put(f"/sessions/{session.id}", new_data)


class SessionCategory(Enum):
    RECITAL = auto()
    WORKSHOP = auto()
    WORSHIP = auto()


@dataclass(frozen=True)
class Session:
    id: str
    name: str
    description: str | None
    start_time: datetime
    end_time: datetime
    full_description: str | None
    data: dict[str, Any]
    category: SessionCategory | None
    speakers: list[Speaker]

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        all_speakers: dict[str, Speaker],
        map: dict[str, list[str]],
    ) -> Self:
        category_name: str = data.get("category", {}).get("name", "")
        try:
            category = SessionCategory[category_name.upper()]
        except KeyError:
            logger.warning(
                "Unknown session category %r for %s", category_name, data["title"]
            )
            category = None
        speaker_ids = map.get(data["id"], [])
        speakers = [all_speakers[id] for id in speaker_ids if id in all_speakers]
        return cls(
            id=data["id"],
            name=data["title"],
            description=data.get("description"),
            start_time=datetime.fromisoformat(data["start"]).astimezone(TIMEZONE),
            end_time=datetime.fromisoformat(data["end"]).astimezone(TIMEZONE),
            full_description=next(
                (
                    d
                    for d in data.get("customFields", [])
                    if d["name"] == "SF24 Full Description"
                ),
                cast(dict[str, list[str | None]], {"value": [None]}),
            )["value"][0],
            category=category,
            speakers=speakers,
            data=data,
        )

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def url_relpath(self) -> str:
        if self.category == SessionCategory.RECITAL:
            return f"recitals/{self.slug}/"
        elif self.category == SessionCategory.WORKSHOP:
            return f"workshops/{self.slug}/"
        elif self.category == SessionCategory.WORSHIP:
            return f"worship/{self.slug}/"
        return f"sessions/{self.slug}/"

    def link(self, base_url: str) -> str:
        return f'<a href="{base_url}{self.url_relpath}">{self.name}</a>'

    def page_content(self, base_url: str) -> str:
        page = dedent(
            """\
            <h2>Date/Time</h2>
            <p>{self.start_time:%A, %B %d, %Y}<br>
            {self.start_time:%I:%M %p} – {self.end_time:%I:%M %p}</p>
            """
        ).format(**locals())
        if self.speakers:
            speaker_items = "\n".join(
                f"<li>{speaker.link(base_url)}</li>" for speaker in self.speakers
            )
            page += dedent(
                """\
                <h2>Presenters</h2>
                <ul>
                {speaker_items}
                </ul>
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


def load_all_sessions(
    cvent: Cvent, speakers: dict[str, Speaker], map: dict[str, list[str]]
) -> dict[str, Session]:
    sessions: dict[str, Session] = {}
    token_param: dict[str, str] = {}
    while True:
        r = cvent.get(
            "/sessions",
            params={
                "filter": f"event.id eq '{cvent.event_id}'",
                "limit": 200,
                **token_param,
            },
        )
        data = r.json()
        sessions.update(
            {d["id"]: Session.from_json(d, speakers, map) for d in data["data"]}
        )
        if (token := data["paging"].get("nextToken")) is not None:
            token_param = {"token": token}
        else:
            break
    return sessions


class SpeakerCategory(Enum):
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
    category: SpeakerCategory | None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        category_name = data["category"]["name"]
        if category_name in {"Composer"}:
            category = SpeakerCategory.COMPOSER
        elif category_name in {
            "Organist",
            "Performer",
            "Soprano",
            "Violinist",
            "Trombonist",
            "Conductor",
        }:
            category = SpeakerCategory.PERFORMER
        elif category_name in {"Presenter"}:
            category = SpeakerCategory.PRESENTER
        else:
            logger.warning(
                "Unknown speaker category %r for %s %s",
                category_name,
                data["firstName"],
                data["lastName"],
            )
            category = None
        return cls(
            id=data["id"],
            first_name=data["firstName"],
            last_name=data["lastName"],
            bio=data.get("biography"),
            photo_url=data["links"].get("profilePicture", {"href": None})["href"],
            category=category,
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
    def url_relpath(self) -> str:
        if self.category == SpeakerCategory.COMPOSER:
            return f"composers/{self.slug}/"
        elif self.category == SpeakerCategory.PERFORMER:
            return f"performers/{self.slug}/"
        elif self.category == SpeakerCategory.PRESENTER:
            return f"presenters/{self.slug}/"
        else:
            return f"people/{self.slug}/"

    def link(self, base_url: str) -> str:
        return f'<a href="{base_url}{self.url_relpath}">{self.name}</a>'

    def page_content(self) -> str:
        page = ""
        if self.photo_url:
            page += dedent(
                """\
                <img class="speaker-photo" src="{self.photo_url}">
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


def load_all_speakers(cvent: Cvent) -> dict[str, Speaker]:
    speakers: dict[str, Speaker] = {}
    token_param: dict[str, str] = {}
    while True:
        r = cvent.get(
            "/speakers",
            params={
                "filter": f"event.id eq '{cvent.event_id}'",
                "limit": 200,
                **token_param,
            },
        )
        data = r.json()
        speakers.update({d["id"]: Speaker.from_json(d) for d in data["data"]})
        if (token := data["paging"].get("nextToken")) is not None:
            token_param = {"token": token}
        else:
            break
    return speakers


def load_session_speaker_map(repodir: Path) -> dict[str, list[str]]:
    with (repodir / "session-speaker-map.json").open() as f:
        data = json.load(f)
    return {key.lower(): [v.lower() for v in values] for key, values in data.items()}


def index_page(title: str, links: list[str]) -> str:
    return "\n".join(
        [
            "<ul>",
            *(f"<li>{link}</li>" for link in links),
            "</ul>",
        ]
    )


def workshops_page(sessions: dict[str, Session]) -> str:
    workshops = []
    for session in sessions.values():
        if session.category != SessionCategory.WORKSHOP:
            continue
        workshop_lines = [
            "{{ workshop(",
            f'  name="{session.name}",',
        ]
        for presenter, n in zip(session.speakers, ["", "2", "3"]):
            workshop_lines.append(f'  presenter{n}="{presenter.name}",')
        # Need to remove the comma from the last line before the closing paren
        workshop_lines[-1] = workshop_lines[-1].removesuffix(",")
        workshop_lines.append(") }}")
        workshops.append("\n".join(workshop_lines))
    return "\n\n".join(workshops)


RENAME_ALIASES = {
    "berkeley-symphony": "berkeley-symphony-orchestra",
    "new-organ-chorales-in-the-schübler-tradition": "new-organ-chorales-in-the-schubler-tradition",
    "shin-young-lee-olivier-latry-with-the-berkeley-symphony": "shin-young-lee-olivier-latry-with-the-berkeley-symphony-orchestra",
    "the-castro-tales-of-the-village-walking-tour-sold-out": "the-castro-tales-of-the-village-walking-tour",
    "visit-to-sfmoma-sold-out": "visit-to-sfmoma",
    "walking-tour-of-chinatown-sold-out": "walking-tour-of-chinatown",
}

USED_RENAMES = set()


def _find_old_url(url: str) -> str | None:
    for new_part, old_part in RENAME_ALIASES.items():
        if new_part in url.split("/"):
            USED_RENAMES.add(new_part)
            return url.replace(new_part, old_part)
    return None


def render_page(url: str, title: str, content: str, aliases: list[str] = []):
    date = datetime.now()
    aliases_with_future = [f"/future{url}", *(f"/future{alias}" for alias in aliases)]

    aliases_with_renames: list[str] = []
    if old_url := _find_old_url(url):
        aliases_with_renames.append(old_url)
    # For regular aliases, we need both the new- and old-named versions.
    for alias in aliases:
        aliases_with_renames.append(alias)
        if old_alias := _find_old_url(alias):
            aliases_with_renames.append(old_alias)
    # For future aliases, since the future site used only the old name, we only need
    # the old-named version.
    for future_alias in aliases_with_future:
        if old_future_alias := _find_old_url(future_alias):
            aliases_with_renames.append(old_future_alias)
        else:
            aliases_with_renames.append(future_alias)

    # Additionally, if any aliases contain non-ascii-characters, add an alias with the
    # closest possible ASCII-only representation.
    for alias in aliases_with_renames[:]:
        if (unidecoded := unidecode(alias)) != alias:
            aliases_with_renames.append(unidecoded)

    alias_list = ",".join(f'"{alias}"' for alias in aliases_with_renames)
    return dedent(
        """\
        +++
        title = '''{title}'''
        path = '''{url}'''
        template = "future.html"
        aliases = [{alias_list}]
        +++

        <h1>{title}</h1>

        {content}
        """
    ).format(**locals())


@asynccontextmanager
async def manage_repo(repo_dir: Path, commit: bool, push: bool) -> AsyncIterator[None]:
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
                logger.warning("Repo is dirty, cleaning it...")
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
                "--author=sf24-website <colin+sf24website@lumeh.org>",
                "--message=sf24-website update",
                cwd=repo_dir,
            )
            if (returncode := await proc.wait()) != 0:
                raise RuntimeError(f"'git commit' exited {returncode}")
            if push:
                proc = await create_subprocess_exec("git", "push", cwd=repo_dir)
                if (returncode := await proc.wait()) != 0:
                    raise RuntimeError(f"'git push' exited {returncode}")


def prepare_dir(path: Path) -> None:
    path.mkdir()
    (path / "_index.md").write_text("+++\nrender = false\n+++\n")


async def generate_pages(
    base_url: str,
    repo_dir: Path,
    commit: bool,
    push: bool,
    sessions: dict[str, Session],
    speakers: dict[str, Speaker],
) -> None:
    async with manage_repo(repo_dir, commit, push):
        outdir = repo_dir / "content/_generated"
        if outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir()

        prepare_dir(outdir / "sessions")
        prepare_dir(outdir / "recitals")
        prepare_dir(outdir / "workshops")
        prepare_dir(outdir / "worship")
        for session in sessions.values():
            url_relpath = session.url_relpath
            path = outdir / (url_relpath.removesuffix("/") + ".md")
            if not url_relpath.startswith("sessions"):
                aliases = [
                    base_url + "sessions/" + url_relpath.split("/", maxsplit=1)[1]
                ]
            else:
                aliases = []
            if path.exists():
                logger.warning("Overwriting duplicate %s", url_relpath)
            path.write_text(
                render_page(
                    base_url + url_relpath,
                    session.name,
                    session.page_content(base_url),
                    aliases,
                )
            )
        session_links = [
            s.link(base_url) for s in sorted(sessions.values(), key=attrgetter("name"))
        ]
        (outdir / "sessions/index.md").write_text(
            render_page(
                base_url + "sessions/",
                "Sessions",
                index_page("Sessions", session_links),
            )
        )

        # (outdir / "workshops/index.md").write_text(
        #     render_page(
        #         base_url + "workshops/",
        #         "Workshops",
        #         workshops_page(sessions),
        #     )
        # )

        prepare_dir(outdir / "people")
        prepare_dir(outdir / "composers")
        prepare_dir(outdir / "performers")
        prepare_dir(outdir / "presenters")
        for speaker in speakers.values():
            url_relpath = speaker.url_relpath
            path = outdir / (url_relpath.removesuffix("/") + ".md")
            if not url_relpath.startswith("people"):
                aliases = [base_url + "people/" + url_relpath.split("/", maxsplit=1)[1]]
            else:
                aliases = []
            if path.exists():
                logger.warning("Overwriting duplicate %s", url_relpath)
            path.write_text(
                render_page(
                    base_url + url_relpath,
                    speaker.name,
                    speaker.page_content(),
                    aliases,
                )
            )
        people_links = [
            s.link(base_url) for s in sorted(speakers.values(), key=attrgetter("name"))
        ]
        (outdir / "people/index.md").write_text(
            render_page(
                base_url + "people/",
                "People",
                index_page("People", people_links),
            )
        )

        if unused_renames := (set(RENAME_ALIASES.keys()) - USED_RENAMES):
            raise RuntimeError(f"Some renames were unused: {sorted(unused_renames)}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["gen", "read-more"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.push and not args.commit:
        raise ValueError(f"--push requires --commit")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repodir = Path(__file__).parent
    config = Config.load(repodir)
    auth = Auth.load_or_fetch_token(repodir, config)
    cvent = Cvent(auth, config.event_id)

    if args.mode == "gen":
        map = load_session_speaker_map(repodir)
        speakers = load_all_speakers(cvent)
        sessions = load_all_sessions(cvent, speakers, map)
        asyncio.run(
            generate_pages(
                args.base_url, args.repo_dir, args.commit, args.push, sessions, speakers
            )
        )


if __name__ == "__main__":
    main()
