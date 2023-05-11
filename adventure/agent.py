import openai
import os
import re
import tiktoken
import time

from typing import Dict, List, Union
from collections import deque


LLM_MODEL = "gpt-3.5-turbo"
MAX_LLM_TOKEN = 4096
OPENAI_TEMPERATURE = 0.0
OBJECTIVE = "Win the text based adventure game Colossal Cave Adventure"

WALKTHROUGH = """
Enter' building, and 'take' the keys and lamp. Now go 'out' of the building.
Go down, down to the slit in the rock. Go South to the grate. Unlock grate with key, open grate. Go down.

Go west, take the cage. Go west again, turn on the lamp. Grab the rod and say XYZZY. POOF you're back in the well house! Say XYZZY again to return to the Debris Room.

Go west to the sloping room. Drop the rod. Go west to the Orange River Chamber. Grab the bird (it's afraid of the rod). Now go east and get the rod, and head back west again. Go down into the pit. You're in the caves!

Level 1 is pretty easy. You start at the Hall of Mists. Head south and grab the gold (+7 score, +9 to drop off). Head north. Hall of the Mountain King, with a snake. Open the cage so the bird drives away the snake. OK, passages in all directions.

To the west is a West Side Chamber with coins (+7 score, +5 drop off). Grab and go east again. To the south is the South Side Chamber with jewelry (+7 score, +5 drop off). Grab it and go north again. To the north is a chamber with a hole in it, with silver. Grab the silver (+7 score, +5 drop off).

Go north one more room, to the Y2 room. Say "plugh" to zap back to the Well House. Drop off your treasures.

Drop off the keys for now, then say "plugh" to return to Y2.

If you hit a dwarf in here anywhere, and he throws an axe, take it! Then when you see him again, "throw axe at dwarf" to get him. If you miss, just "take axe" and throw it again.

OK, now head to the Fissure, west of the Hall of Mists. Wave the Rod to make a crystal bridge appear. Cross it to grab the diamonds. Now head west and south to get into the Pirate Maze. You want to go EAST, SOUTH, then NORTH to get to the chest. When you take it, head SOUTHEAST, WEST and SOUTH to get down to the Orange Room. Just go east twice and then say XYZZY to pop back to the well house
"""


api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    api_key = input("OpenAI Key:")
    os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = api_key


def prompt_to_history(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": prompt}]


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def openai_call(
    messages: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
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
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break



# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self, initial_list: list = []):
        self.tasks = deque(initial_list)
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

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


def gametask_creation_agent(
        objective: str = OBJECTIVE, walkthrough: str = WALKTHROUGH
):
    prompt = f"""
You are to use the result from an execution agent to create new tasks with the following objective: {objective}.

Please utilize the following walkthrough to win the game:

{walkthrough}

Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output.

"""

    response = openai_call(prompt_to_history(prompt), max_tokens=2000)
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)

    output = [{"task_name": task_name} for task_name in new_tasks_list]

    tasks_storage = SingleTaskListStorage(output)

    return tasks_storage


# Execute a task based on the objective and five previous tasks
def player_agent(objective: str, history: List[Dict[str, str]]) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """

    prompt = """
You are playing the 1977 classic Colossal Cave. 

If you ask the same question in a loop, use the "help" command to get out of the loop. Don't get frustrated and only take one item at a time.

The games text parser is limited, keep your commands to one action and 1-3 words. Enter a single command for each prompt. Use the following guide to beat the game. Look around and e what is visible. if your objective is invisible, keep moving. You can only move north, south, east, and west.

"""
    prompt += f"Choose the next game input based on the following objective: {objective}\n"
    prompt += 'Take into account these previously completed tasks in the chat history'
    messages = prompt_to_history(prompt) + history
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

