import os
import json
import requests
import logging

from typing import Optional, Union, Generator, Iterator
from pydantic import BaseModel, Field
from fastapi import Request
from urllib.parse import urlencode

from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion

# Deprecated
# upload audio file to a ASR service and get the STT (speech-to-test) result

class Pipe:
    class Valves(BaseModel):
        MODEL_ID_TO_DIGEST_CONTEXT: str = Field(default="o4-mini-2025-04-16")
        URL_OPENAI_API_TO_DIGEST: str = Field(
            default="https://api.openai.com/v1/chat/completions",
            description="URL for accessing OpenAI API endpoints.",
        )
        KEY_OPENAI_API: str = Field(
            default="",
            description="API key for authenticating requests to the OpenAI API.",
        )
        URL_STT_API: str = Field(
            default="http://192.168.50.41:9000/asr",
            description="URL for speech to text API",
        )
        OUTPUT_WHISPER: str = Field(default="txt")
        PATH_FILE_UPLOAD: str = Field(default="/app/backend/data/uploads")

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger("s2t2d.pipe")  # speech to text to digest

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
    ) -> Union[str, Generator, Iterator]:
        user = Users.get_user_by_id(__user__["id"])
        body["model"] = self.valves.MODEL_ID_TO_DIGEST_CONTEXT

        req_json = await __request__.json()
        file_infos = self.extract_file_infos(req_json)
        self.log.info(f"file_infos: {file_infos}")

        if not file_infos:
            return "none of uploaded audio file"

        f = file_infos[-1]
        # STT via Local Whisper Model
        audio_content = f.get("content", "")
        audio_fname = f.get("filename", "")
        audio_id = f.get("id", "")

        messages = body.get("messages", [])
        self.log.info(f"messages: {messages}")
        self.log.info(f"audio_content: {audio_content}")

        if messages:
            if not audio_content:
                # STT via Whisper Local Web Service
                audio_content = self.speech2text(audio_id, audio_fname)

            messages[-1]["content"] = audio_content
            self.log.info(f"audio_content: {audio_content}")

        payload = {**body}
        self.log.info(f"payload: {payload}")

        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.KEY_OPENAI_API}"
        headers["Content-Type"] = "application/json"

        try:
            r = requests.post(
                url=f"{self.valves.URL_OPENAI_API_TO_DIGEST}",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

    def speech2text(self, audio_id, audio_fname) -> str:
        fname = ""
        for f in os.listdir(self.valves.PATH_FILE_UPLOAD):
            if audio_id and audio_id in f and audio_fname and audio_fname in f:
                fname = f
                break

        if not fname:
            self.log.warning(f"audio file not found: {audio_fname}, {audio_id}")
            return ""

        file_path = "/".join([self.valves.PATH_FILE_UPLOAD, fname])
        if not os.path.exists(file_path):
            self.log.warning(f"File '{file_path}' is not found.")
            return ""

        if not os.access(file_path, os.R_OK):
            self.log.warning(f"You do not have permission for file '{file_path}'.")
            return ""

        url_base = self.valves.URL_STT_API[: self.valves.URL_STT_API.rfind("/")]
        if not self.is_url_available(url_base):
            self.log.error(f"{self.valves.URL_STT_API} is unreachable")
            return ""

        url = self.valves.URL_STT_API
        headers = {"accept": "application/json"}
        params = {
            "encode": "true",
            "task": "transcribe",
            "language": "zh",
            "initial_prompt": "繁體中文",
            "word_timestamps": "false",
            "output": self.valves.OUTPUT_WHISPER,
        }

        try:
            # Open the file in binary read mode
            with open(file_path, "rb") as f:
                files = {"audio_file": (os.path.basename(file_path), f)}

                print(f"Attempting to upload '{os.path.basename(file_path)}' to {url}")
                response = requests.post(
                    url,
                    headers=headers,
                    params=params,
                    files=files,
                )

                # Check the response
                response.raise_for_status()

                # Raises an HTTPError for bad responses (4xx or 5xx)
                self.log.info(
                    f"File uploaded successfully! Status Code: {response.status_code}"
                )

                # Print the server's response content
                self.log.info(f"Server Response: <BEING>{response.text}<END>")

                if not response.text:
                    self.log.warning(
                        f"{audio_fname} audio file format is not supported"
                    )

                return response.text

        except requests.exceptions.HTTPError as e:
            return "\n".join(
                [
                    f"HTTP Error during file upload: {e}",
                    f"Server Response: {e.response.text}",
                ]
            )
        except requests.exceptions.ConnectionError as e:
            return f"Connection Error: Could not connect to the server at {url}. {e}"
        except requests.exceptions.Timeout as e:
            return f"Timeout Error: The request to {url} timed out. {e}"
        except requests.exceptions.RequestException as e:
            return f"An unexpected error occurred during the request: {e}"
        except IOError as e:
            return f"Error opening or reading file: {file_path}. {e}"
        except Exception as e:
            return f"Error: {e}"

    def extract_file_infos(self, req_json: dict) -> list:
        """
        Extracts "filename", "id", and "content" from a given data structure
        without raising a KeyError if the keys are missing.

        Args:
            req_dic (dict): The input request structure.

        Returns:
            list: A list of dictionaries, where each dictionary contains
                  "filename", "id", and "content" if found in the 'files' list.
                  Missing keys will have a value of empty string.
        """
        file_infos = []
        files = req_json.get("files", [])
        for file_entity in files:
            file = file_entity.get("file", {})
            file_infos.append(
                {
                    "filename": file.get("filename", ""),
                    "id": file.get("id", ""),
                    "content": file.get("data", {}).get("content", ""),
                }
            )

        return file_infos

    def is_url_available(self, url: str, timeout: int = 2) -> bool:
        """
        Checks if a given URL is available (reachable and returns a success status code).

        Args:
            url (str): The URL to check.
            timeout (int): The maximum number of seconds to wait for a response.

        Returns:
            bool: True if the URL is available and returns a 2xx status code, False otherwise.
                  Prints specific error messages for debugging.
        """
        if not url.startswith(("http://", "https://")):
            print(
                f"Error: URL must start with 'http://' or 'https://'. Provided: {url}"
            )
            return False

        try:
            # Use a HEAD request for efficiency, as we only need the status code.
            # Allow redirects to follow moved URLs.
            response = requests.get(
                url, timeout=timeout, allow_redirects=True, stream=True
            )

            # Raise an HTTPError for bad responses (4xx or 5xx)
            response.raise_for_status()

            # If no HTTPError was raised, it means the status code is 2xx.
            print(
                f"Success: URL '{url}' is available. Status Code: {response.status_code}"
            )
            return True

        except requests.exceptions.HTTPError as e:
            print(
                f"Error: URL '{url}' returned an HTTP error status: {e.response.status_code} - {e.response.reason}"
            )
            return False
        except requests.exceptions.ConnectionError as e:
            print(
                f"Error: Could not connect to URL '{url}'. Connection refused or DNS error. Details: {e}"
            )
            return False
        except requests.exceptions.Timeout as e:
            print(
                f"Error: Request to URL '{url}' timed out after {timeout} seconds. Details: {e}"
            )
            return False
        except requests.exceptions.RequestException as e:
            print(
                f"Error: An unexpected requests error occurred for URL '{url}'. Details: {e}"
            )
            return False
        except Exception as e:
            print(
                f"Error: An unknown error occurred while checking URL '{url}'. Details: {e}"
            )
            return False
