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
1. Enter building. Get keys and lamp.  The lamp appears here should you have to
be reincarnated.
2. Find gate and unlock gate (with key).  Get cage. Find cave, enter cave.
3. Try the magic word, XYZZY.  Try it again.  Get rod.
4. Drop rod. Get bird. Get rod.
5. Wave rod. Drop rod. Go get diamonds and gold.
6. Release bird. Drop cage. Grab coins and silver. Note: Dwarves are lousy
shots, but sometimes, unfortunately, they can get lucky.  Carry the ax, as you
may encounter up to five of these critters.
7. Try the magic word PLUGH also. Drop silver, gold nugget, diamonds, jewelry,
and coins. If you haven't already, you will soon encounter a thieving pirate.
Not to worry, he's got to rob you at least once if you're to win all the
marbles.
the software den (Microsoft Version only). Don't mess with anything--Software
types are a weird lot.  Get magic word LWPI.  Works only from here.
8. Water plant twice. Get water for the second pass at the waterfall.
9. Attack dragon. Yes. Get rug.
10. Get oil (in now empty bottle). Climb the plant.
11. Oil door. Open door. Drop bottle. While here and while carrying golden eggs,

enter FEE, FIE, FOE, FOO, one word at a time. Check your inventory.  Go back to
where you first found the eggs.
12. Throw eggs (to troll).  Cross bridge before the FEE, FIE, whatever.
13. Feed bear.  Unlock chain (with key).  Get golden chain. Drop key.  Get bear.

Don't forget the rare spices.  At volcano view, read and remember the words of
fire.
14. Release bear.  Don't try crossing the bridge with him on the chain.
15. Open oyster (with trident).  Pearl will roll down into the cul-de-sac.
16. Insert coin to get a replacement battery for your lamp, if necessary.  Coins

are a treasure, however, and you won't ge them back, so try beating the game
with the original batteries only.
17. Never drop vase unless you have already dropped the pillow.
18. Drop everything in order to enter.  Get emerald.
19. Say PLOVER.  Get pyramid.  PLOVER, PLUGH and pi-tooie!
20. You must traverse the maze to get Pirate's Treasure Chest, which doesn't
appear until he's robbed you.  Return via the Pit and XYZZY.  Drop all treasures

in the House.
21. Drop magazine in Witt's End for a point.  Get out by entering all different
directions except north.  It may take a while.  Slog around in various and
distant locations until a voice announces that the cave is now closed.  At this
point you are teleported to the two-room Master's Game.
23. Get black rod (with the rusty marks, not the star).  Drop rod in the
northeast room.  Retreat to the southwest room. Type blast. Alternate endings
are possible, but will not yield sufficient points to earn you the rank of
adventure grandmaster.

Treasure List
What                                                            Where
Golden Eggs                                             Giant Room
Trident                                                 Magnificent Canyon
Pearl                                                   Clam Room
Pirate's Chest                                          Pirate's Maze
Platinum Pyramid                                        Dark Room
Emerald                                                 Plover Room
Ming Vase (and Pillow)                                  Oriental Room
Rare Spices                                             Chamber of Boulders
Persian Rug                                             Dragon's Den
Golden Chain                                            Bear's Chamber
Diamonds                                                West Side of Fissure
Jewelry                                                 South Side Chamber
Gold Nugget                                             Gold Room
Silver Bars                                             North-South Passage
Coins                                                   West Side Chamber
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

    print(f'\n*****TASK CREATION AGENT PROMPT****\n{prompt}\n')
    response = openai_call(prompt_to_history(prompt), max_tokens=2000)
    print(f'\n****TASK CREATION AGENT RESPONSE****\n{response}\n')
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)
            # print('New task created: ' + task_name)

    output = [{"task_name": task_name} for task_name in new_tasks_list]

    tasks_storage = SingleTaskListStorage(output)

    return tasks_storage


def subtask_creation_agent(
        objective: str, result: str = "", task_description: str = "", task_list: List[str] = []
):
    prompt = f"""
You are to use the result from an execution agent to create new tasks with the following objective: {objective}.
The last completed task has the result: \n{result}
"""
    if task_description:
        prompt += "This result was based on this task description: {task_description}.\n"""

    if task_list:
        prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
    prompt += "Based on the result, return a list of tasks to be completed in order to meet the objective. "
    if task_list:
        prompt += "These new tasks must not overlap with incomplete tasks. "

    prompt += """
Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output."""

    print(f'\n*****TASK CREATION AGENT PROMPT****\n{prompt}\n')
    response = openai_call(prompt_to_history(prompt), max_tokens=2000)
    print(f'\n****TASK CREATION AGENT RESPONSE****\n{response}\n')
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)
            # print('New task created: ' + task_name)

    output = [{"task_name": task_name} for task_name in new_tasks_list]
    tasks_storage = SingleTaskListStorage(output)

    return tasks_storage


def prioritization_agent(tasks_storage, current_game_task):
    task_names = tasks_storage.get_task_names()
    bullet_string = '\n'

    prompt = f"""
You are tasked with prioritizing the following tasks: {bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of your team: {current_game_task}.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""

    print(f'\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}\n')
    response = openai_call(prompt_to_history(prompt), max_tokens=2000)
    print(f'\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}\n')
    if not response:
        print('Received empty response from priotritization agent. Keeping task list unchanged.')
        return
    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip():
                new_tasks_list.append({"task_id": task_id, "task_name": task_name})

    return SingleTaskListStorage(new_tasks_list)


# Execute a task based on the objective and five previous tasks
def player_agent(objective: str, task: str, history: List[Dict[str, str]]) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """

    # print("\n****RELEVANT CONTEXT****\n")
    # print(context)
    # print('')
    prompt = """
You are playing the 1977 classic Colossal Cave. 

If you ask the same question in a loop, use the "help" command to get out of the loop. Don't get frustrated and only take one item at a time.

The games text parser is limited, keep your commands to one action and 1-3 words. Enter a single command for each prompt. Use the following guide to beat the game. Look around and e what is visible. if your objective is invisible, keep moving. You can only move north, south, east, and west.

"""
    prompt += f"Choose the next game input based on the following objective: {objective}\n"
    prompt += 'Take into account these previously completed tasks in the chat history'
    prompt += f'\nYour task: {task}\n If nothing else remains to be done, return an empty string'
    messages = prompt_to_history(prompt) + history
    return openai_call(messages, max_tokens=2000)


def next_game_input(
    history: List[Dict[str, str]],
    tasks_storage: SingleTaskListStorage,
    current_game_task: str
):
    if not tasks_storage.is_empty():
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in tasks_storage.get_task_names():
            print(" • " + str(t))

        # Step 1: Pull the first incomplete task
        task = tasks_storage.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_name"]))

        # Send to execution function to complete the task based on the context
        result = player_agent(current_game_task, str(task["task_name"]), history)
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Step 2: Add result to history
        history.append({"role": "assistant", "content": result})
    else:
        result = ""

    return result, history


def update_subtask_list(game_output: str, current_game_task: str, tasks_storage: SingleTaskListStorage) -> SingleTaskListStorage:
    new_tasks = subtask_creation_agent(
        current_game_task,
        game_output,
        tasks_storage.get_task_names(),
    )

    prioritized_list = prioritization_agent(tasks_storage, current_game_task)
    if prioritized_list:
        return prioritized_list
    else:
        return new_tasks
