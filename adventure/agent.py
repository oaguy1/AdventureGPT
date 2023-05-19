"""
OpenAI based agents for interactiving in the game world

Copyright 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""

import os
import openai
import re
import tiktoken
import time

from typing import Dict, List, Union
from collections import deque


LLM_MODEL = "gpt-3.5-turbo"
MAX_LLM_TOKEN = 4096
OPENAI_TEMPERATURE = 0.0


api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    api_key = input("OpenAI Key:")
    os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = api_key


def prompt_to_history(prompt: str) -> List[Dict[str, str]]:
    """
    Convert a string into a message history format used by ChatGPT
    """
    return [{"role": "system", "content": prompt}]


def chunk_tokens_from_string(string: str, model: str = LLM_MODEL, chunk_size: int = 500) -> List[str]:
    """
    Chunks the string into blocks based on a number of tokens (estimated).
    """
    tokens = string.split()
    total = len(tokens)
    multiplier = 0
    chunks = []

    while multiplier * chunk_size < total:
        start_idx = multiplier * chunk_size
        if start_idx + chunk_size < total:
           chunks.append(" ".join(tokens[start_idx:start_idx + chunk_size]))
        else:
           chunks.append(" ".join(tokens[start_idx:]))

    return chunks


def openai_call(
    messages: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    """
    Call OpenAI using its chat completion API. Covers some nice expected
    scenarios when hitting the API, such as rate limiting
    """
    while True:
        try:
            # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
            # TODO: different limits for different models.

            # Use chat completion API
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 9 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break



class SingleTaskListStorage:
    """
    A task list for storing game tasks
    """

    def __init__(self, initial_list: list = []):
        self.tasks = deque(initial_list)
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def concat(self, tasks):
        self.tasks += deque(tasks.tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]

    def __repr__(self):
        self.tasks

    def __str__(self):
        return "\n".join([f'{i}. {t["task_name"]}' for i, t in enumerate(self.tasks)])


def openai_task_response_to_list(response: str):
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)

    return [{"task_name": task_name} for task_name in new_tasks_list]

          
def gametask_creation_agent(history: List[Dict[str, str]], max_history: int = 30) -> SingleTaskListStorage:
    """
    Creates a list of game tasks to complete based game history
    
    Args:
        history (list): What game inputs/outputs have been received. In the
            form of OpenAI Conversation Prompt
        max_history (int): The max number of historical prompt/responses to give to the API

    Returns:
        SingleTaskListStorage: A list of tasks to be completed to beat the game

    """
    prompt = f"""
You are an agent tasked with creating a list of tasks in order win the text based adventure game Colossal Cave Adventure.

Here is a guide on how to win:

1. Explore every space, You may have to move in a direction rather than entering directly
2. Examine or read every object. There may be more details that will help later on
3. Pick up or take every object you can. Inventory command will remind you of what you have. If you hit a limit of what you can carry, you may need to drop some items
4. Try any and every very you can think of when in new spaces. Experimenting is required to beat every textbased adventure

Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period.
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output.

"""

    prompt += 'Take into account the game history attached here:'
    first_history_index = 0 if len(history) <= max_history else -1 * max_history
    messages = prompt_to_history(prompt) + history[first_history_index:]
    response = openai_call(messages, max_tokens=2000)
    task_list = openai_task_response_to_list(response)
    return SingleTaskListStorage(task_list)


def walkthrough_gametask_creation_agent(walkthrough: str):
    """
    Creates a list of game tasks to complete based on a given walthrough.

    Args:
        walkthrough (str): The text of the game walkthrough

    Returns:
        SingleTaskListStorage: A list of tasks to be completed to beat the game

    """
    prompt = f"""
You are an agent tasked with creating a list of tasks in order win the text based adventure game Colossal Cave Adventure.

Please utilize the following walkthrough to win the game:

{walkthrough}

Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period.
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output.
"""

    response = openai_call(prompt_to_history(prompt), max_tokens=2000)
    task_list = openai_task_response_to_list(response)
    return SingleTaskListStorage(task_list)


def prioritization_agent(task_storage: SingleTaskListStorage, history: List[Dict[str, str]], max_history: int = 30) -> SingleTaskListStorage:
    """
    Given a SingleTaskListStorage and game history, prioritize the task list to be more effective

    Args:
        task_storage (SingleTaskListStorage): The current task list
        history (list): What game inputs/outputs have been received. In the
            form of OpenAI Conversation Prompt
        max_history (int): The max number of historical prompt/responses to give to the API

    Returns:
        SingleTaskListStorage: A list of tasks to be completed to beat the game

    """
    task_names = task_storage.get_task_names()
    bullet_string = '\n'

    prompt = f"""
You are tasked with prioritizing the following tasks: {bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of winning the game.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""

    prompt += 'Take into account these previously completed tasks in the chat history'
    first_history_index = 0 if len(history) <= max_history else -1 * max_history
    messages = prompt_to_history(prompt) + history[first_history_index:]
    response = openai_call(messages, max_tokens=2000)
    if not response:
        # Received empty response from priotritization agent. Keeping task list unchanged.
        return task_storage

    new_tasks = openai_task_response_to_list(response)
    return SingleTaskListStorage(new_tasks)


def player_agent(objective: str, history: List[Dict[str, str]], completed_tasks, max_history: int = 30) -> str:
    """
    Executes a task based on the given objective and previous game history

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        history (list): What game inputs/outputs have been received. In the
            form of OpenAI Conversation Prompt

    Returns:
        str: The response generated by the AI for the given task.

    """

    prompt = """
You are playing the 1977 classic Colossal Cave. 

If you ask the same question in a loop, use the "help" command to get out of the loop. Don't get frustrated and only take one item at a time.

The games text parser is limited, keep your commands to one action and 1-3 words. Enter a single command for each prompt. Use the following guide to beat the game. Look around and e what is visible. if your objective is invisible, keep moving. You can only move north, south, east, and west.

"""
    prompt += f"Choose the next game input based on the following objective: {objective}\n"
    prompt += f"\nThe following objectives have been completed\n{completed_tasks}\n\n"
    prompt += 'Take into account these previously completed tasks in the chat history'
    first_history_index = 0 if len(history) <= max_history else -1 * max_history
    messages = prompt_to_history(prompt) + history[first_history_index:]
    return openai_call(messages, max_tokens=2000)


# Decide if task has been completed
def task_completion_agent(objective: str, history: List[Dict[str, str]], max_history: int = 10) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """

    prompt = "You are playing the 1977 classic Colossal Cave.\n"

    prompt += f"Decide if the currnent objective has been completed. \n\n Objective: {objective}\n"
    prompt += 'Take into account these previously completed tasks in the chat history'
    prompt += f'Reply with a simple "COMPLETE" or "INCOMPLETE". Conversation history is below:\n'
    first_history_index = 0 if len(history) <= max_history else -1 * max_history
    messages = prompt_to_history(prompt) + history[first_history_index:]
    return openai_call(messages, max_tokens=2000).lower() == "complete"

