import openai
from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff
import pandas as pd
import os
import numpy as np
from scipy import spatial
import sys
import os
import tiktoken
from transformers import AutoTokenizer
import openai
from openai import OpenAI
from .web_util import output_to_port, listen_to_server, username_record
from .llm_providers import (
    infer_provider,
    build_openai_chat_kwargs,
    extract_text,
    anthropic_messages_create,
    openai_output_token_estimate,
    anthropic_output_token_count,
    get_anthropic_api_key,
    extract_human_port_response,
)

cwd = os.getcwd()
gpt4_key_file = os.path.join(cwd, "openai_key.txt")
# deepseek_key_file = os.path.join(cwd, "deepseek_key.txt")

with open(gpt4_key_file, "r") as f:
    context = f.read()
openai_key = context.split("\n")[0]

# global statistics
statistics_dict = {
    "total_timestamp": [],
    "total_order_finished": [],
    "total_score": 0,
    "total_action_list": [[], []],
    "content": [],
    "timestep_conversations": [],
}


def reset_statistics_for_episode():
    """Clear per-episode accumulators so each saved JSON is self-contained."""
    statistics_dict["total_timestamp"] = []
    statistics_dict["total_order_finished"] = []
    statistics_dict["total_score"] = 0
    statistics_dict["total_action_list"] = [[], []]
    statistics_dict["content"] = []
    statistics_dict["timestep_conversations"] = []
    statistics_dict.pop("agents", None)

# turn statistics
turn_statistics_dict = {
    "timestamp": 0,
    "order_list": [],
    "actions": [],
    "map": "",
    "statistical_data": {
        "score": 0,
        "communication": [
            {"call": 0, "turn": [], "token": []},
            {"call": 0, "turn": [], "token": []},
        ],
        "error": [
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
        ],
        "error_correction": [
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
        ],
    },
    "content": {
        "observation": [[], []],
        "reflection": [[], []],
        "content": [[], []],
        "action_list": [[], []],
        "original_log": "",
    },
}

# LLM models
tokenizer, model = None, None
# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0125": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "llama3:70b-instruct-fp16": 4096,
}
DEFAULT_TOKEN_LIMIT = 128000
sys.path.append(os.getcwd())
EMBEDDING_MODEL = "text-embedding-3-small"


def _output_token_count_tiktoken(text: str, model_hint: str = "gpt-4") -> int:
    if model_hint == "p50k_base":
        enc = tiktoken.get_encoding("p50k_base")
        return len(enc.encode(text))
    try:
        enc = tiktoken.encoding_for_model(model_hint)
    except Exception:
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class Module(object):
    """
    This module is responsible for communicating with LLMs.
    """

    def __init__(
        self,
        role_messages,
        model="gpt-3.5-turbo-0301",
        model_dirname="~/",
        local_server_api="http://localhost:8000/v1",
        retrival_method="recent_k",
        K=3,
        openai_base_url=None,
        anthropic_max_tokens=4096,
    ):

        self.model = model
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api
        self.retrival_method = retrival_method
        self.K = K
        self.openai_base_url = openai_base_url
        self.anthropic_max_tokens = int(anthropic_max_tokens)

        prov = infer_provider(self.model)
        self.chat_model = prov in ("openai", "anthropic", "openai_compatible")
        self.instruction_head_list = role_messages
        # a dynamic changed dialog_history used for generating  different input for each failure
        self.dialog_history_list = []
        # save the dialog_history of meetting first failture
        self.dialog_history_list_storage = []
        self.current_user_message = None
        self.cache_list = None
        self.experience = []
        self.embedding = None
        self.current_timestep = None

    def load_embedding(self):
        df = pd.read_csv(os.getcwd() + "/data/embedding_" + self.name.lower() + ".csv")
        df["embedding"] = df.embedding.apply(eval).apply(np.array)
        self.embedding = df

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)

    def get_cache(self) -> list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K :]
            else:
                return []
        else:
            return None

    def query_messages(self, rethink) -> list:
        sytem_message = [
            {
                "role": "system",
                "content": "You are an intelligent agent planner, you need to generate output and plan in the specified format according to the game rules and environmental status.",
            }
        ]
        query = sytem_message + [
            {
                "role": "user",
                "content": self.instruction_head_list[0]["content"]
                + "<input>\n"
                + self.current_user_message["content"],
            }
        ]
        return query

    @retry_with_exponential_backoff
    def query(
        self,
        key,
        proxy,
        stop=None,
        temperature=0.7,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
        rec = self.K
        messages = self.query_messages(rethink)
        self.cache_list = self.get_cache()

        if trace == False and not rethink:
            messages[len(messages) - 1][
                "content"
            ] += " Based on the failure explanation and scene description, analyze and plan again."

        self.K = rec
        provider = infer_provider(self.model)

        get_response = False
        retry_count = 0
        rs = ""
        token_count = 0
        encoder_name = "gpt-4"

        while not get_response:
            if retry_count > 1:
                rprint(
                    f"[red][ERROR][/red]: LLM query failed for over 3 times "
                    f"(model={self.model}, provider={infer_provider(self.model)})!"
                )
                self.current_user_message["content"] = self.current_user_message[
                    "content"
                ][:-40]
                return "", 0
            try:
                if provider == "human":
                    if messages[-1]["content"].find("Suppose you are a Chef") != -1:
                        receiver = "agent0"
                    elif (
                        messages[-1]["content"].find("Suppose you are a Assistant")
                        != -1
                    ):
                        receiver = "agent1"
                    else:
                        raise ValueError("Invalide role")
                    input_part = messages[-1]["content"].find("<input>\n") + len(
                        "<input>\n"
                    )
                    human_message = messages[-1]["content"][input_part:]
                    recipe = None
                    if receiver == "agent0":
                        recipe_start = messages[-1]["content"].find(
                            "<Recipe need to know>:\n"
                        ) + len("<Recipe need to know>:\n")
                        recipe_end = messages[-1]["content"].find("**Skill**")
                        recipe = messages[-1]["content"][recipe_start:recipe_end]
                    error = None
                    error_start = human_message.find(
                        "DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n"
                    )
                    if error_start != -1:
                        error = human_message[
                            error_start
                            + len("DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n") :
                        ]
                        human_message = human_message[
                            : human_message.find(
                                "Below are the failed and analysis history"
                            )
                        ]
                    response = output_to_port(
                        receiver, human_message, map=map, recipe=recipe, error=error
                    )
                    rs = extract_human_port_response(response)
                    encoder_name = "gpt-3.5-turbo"
                    token_count = _output_token_count_tiktoken(rs, encoder_name)
                    get_response = True

                elif self.model in ["text-davinci-003"]:
                    openai.api_key = openai_key
                    prompt = convert_messages_to_prompt(messages)
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature,
                        max_tokens=256,
                    )
                    time.sleep(1)
                    rs = extract_text(response, "openai")
                    token_count = _output_token_count_tiktoken(rs, "p50k_base")
                    encoder_name = "p50k_base"
                    get_response = True

                elif provider == "anthropic":
                    from anthropic import Anthropic

                    client = Anthropic(api_key=get_anthropic_api_key(cwd))
                    response = anthropic_messages_create(
                        client,
                        self.model,
                        messages,
                        temperature,
                        self.anthropic_max_tokens,
                    )
                    time.sleep(0.2)
                    rs = extract_text(response, "anthropic")
                    n = anthropic_output_token_count(response)
                    token_count = n if n > 0 else _output_token_count_tiktoken(rs)
                    get_response = True

                elif provider == "openai":
                    openai.api_key = openai_key
                    base = self.openai_base_url or os.environ.get("OPENAI_BASE_URL")
                    if "deepseek" in self.model.lower():
                        base = base or os.environ.get(
                            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
                        )
                    client_kw = {"api_key": openai_key}
                    if base:
                        client_kw["base_url"] = base
                    client = OpenAI(**client_kw)
                    chat_kwargs = build_openai_chat_kwargs(
                        self.model, messages, temperature
                    )
                    response = client.chat.completions.create(**chat_kwargs)
                    time.sleep(0.2)
                    rs = extract_text(response, "openai")
                    n = openai_output_token_estimate(response)
                    token_count = (
                        n if n is not None else _output_token_count_tiktoken(rs, self.model)
                    )
                    encoder_name = "gpt-4"
                    get_response = True

                else:
                    # OpenAI-compatible (vLLM, local servers)
                    client = OpenAI(
                        api_key="token-abc123", base_url=self.local_server_api
                    )
                    response = client.chat.completions.create(
                        model=self.model_dirname + self.model,
                        messages=messages,
                        temperature=temperature,
                    )
                    time.sleep(0.2)
                    rs = extract_text(response, "openai")
                    encoder_name = "llama3"
                    tokenizer = AutoTokenizer.from_pretrained(
                        "../lib/llama_tokenizer", local_files_only=True
                    )
                    token_count = len(tokenizer.encode(rs))
                    get_response = True

            except Exception as e:
                retry_count += 1
                rprint("[red][LLM ERROR][/red]:", e)
                time.sleep(1)

        return rs, token_count

    def parse_response(self, response):
        """Legacy helper for dict-shaped API responses; prefer extract_text in query."""
        if self.model in ["text-davinci-003"]:
            return response["choices"][0]["text"]
        prov = infer_provider(self.model)
        if prov == "human" and isinstance(response, dict):
            return extract_human_port_response(response)
        return extract_text(response, prov if prov != "human" else "openai")

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE.get(self.model, DEFAULT_TOKEN_LIMIT)
        print(f"Current token: {self.prompt_token_length}")
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f"Update token: {self.prompt_token_length}")

    def reset(self):
        self.dialog_history_list = []

    def get_top_k_similar_example(self, key, k=4):
        if k == 0:
            return ""
        prompt_begin_chef = "Here are few examples to teach you the usage of your skills, but these are just some examples, you need to flexibly apply your skills according to the specific environment.\
You should make plan for yourself in 'Chef plan', and make plan for assistant by saying to him.\n"
        prompt_begin_assistant = "Here are few examples to teach you the usage of your skills, but these are just some examples, you need to flexibly apply your skills according to the specific environment.\
If you do not know what to do, just ask chef to make a plan for you.\n"
        recipe = """<example_recipe>
Recipe: 
NAME:
onion_soup

INGREDIENTS:
chopped_onion (1)

COOKING STEPs:
1. Put 1 onion into chopping board directly to get the chopped_onion, you should wait for 3 STEPs.
2. Put 1 chopped_onion into pot directly, you should wait for 10 STEPs.
</example_recipe>

"""  # get embedding for current input
        key = ""
        with open(gpt4_key_file, "r") as f:
            context = f.read()
        key = context.split("\n")[0]
        openai.api_key = key

        get_response = False
        openai.api_key = key

        input = self.current_user_message["content"]
        while not get_response:
            try:
                client = OpenAI(api_key=key)
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL, input=[input]
                )
                get_response = True
            except Exception as e:
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(1)

        input_embedding = response.data[0].embedding
        if self.embedding is None:
            self.load_embedding()

        self.embedding["similarities"] = self.embedding.embedding.apply(
            lambda x: 1 - spatial.distance.cosine(x, input_embedding)
        )
        top_k_strings = self.embedding.sort_values(
            "similarities", ascending=False
        ).head(k)["text"]
        result = ""
        for t in top_k_strings:
            if t[0] == "\n":
                t = t[1:]
            result += f"<example>\n{t}\n</example>\n\n"
        if self.name == "Chef":
            result = prompt_begin_chef + result
        elif self.name == "Assistant":
            result = prompt_begin_assistant + result

        return result


def if_two_sentence_similar_meaning(key, proxy, sentence1, sentence2):
    with open(gpt4_key_file, "r") as f:
        context = f.read()
    key = context.split("\n")[0]
    openai.api_key = key
    if sentence1 == "":
        sentence1 = " "
    if sentence2 == "":
        sentence2 = " "
    get_response = False
    while not get_response:
        try:
            client = OpenAI(api_key=key)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=[sentence1, sentence2]
            )
            get_response = True
        except Exception as e:
            rprint("[red][OPENAI ERROR][/red]:", e)
            time.sleep(1)
    embedding_1 = response.data[0].embedding
    embedding_2 = response.data[1].embedding
    score = 1 - spatial.distance.cosine(embedding_1, embedding_2)
    if score > 0.9:
        return True
    else:
        return False
