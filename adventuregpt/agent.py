"""
OpenAI based agents for interactiving in the game world

Copyright (c) 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.

MIT License

Copyright (c) 2023 Yohei Nakajima

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re
import time

from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from typing import Dict, List, Union
from collections import deque


LLM_MODEL = "gpt-3.5-turbo"
MAX_LLM_TOKEN = 4096
OPENAI_TEMPERATURE = 0.0


api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    api_key = input("OpenAI Key:")
    os.environ["OPENAI_API_KEY"] = api_key


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


class ConversationAgent:
    """
    Basline agent class
    """

    def __init__(self):
        self.llm = ChatOpenAI(temperture=OPENAI_TEMPERATURE)
        self.memory = ConversationBufferMemory(return_messages=True)


class GameTaskCreationAgent(ConversationAgent):
    """
    Agent that creates a list of game tasks to complete based game history
    """

    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
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

Take into account the game history attached here: {history}
"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)


    def run(message: str) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            message (str): next game input

        Returns:
            SingleTaskListStorage: A list of tasks to be completed to beat the game

        """
        response = self.conversation.predict(message=message)
        task_list = openai_task_response_to_list(response)
        return SingleTaskListStorage(task_list)


class WalkthroughGameTaskCreationAgent:
    """
    Agent that creates a list of game tasks to complete based on a given walthrough.
    """

    def __init__(self):
        self.llm = OpenAI(temperture=OPENAI_TEMPERATURE)
        self.prompt = PromptTemplate(
                input_variables=["walkthrough"],
                template="""
You are an agent tasked with creating a list of tasks in order win the text based adventure game Colossal Cave Adventure.

Please utilize the following walkthrough to win the game:

{walkthrough}

Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period.
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output.
""")
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm)


    def run(walkthrough: str) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            walkthrough (str): The text of the walkthrough to summarize

        Returns:
            SingleTaskListStorage: A list of tasks to be completed to beat the game

        """
        response = self.chain.run(walkthrough=walkthrough)
        task_list = openai_task_response_to_list(response)
        return SingleTaskListStorage(task_list)


class PrioritizationAgent:
    """
    Agent that given a SingleTaskListStorage prioritizes the task list to be more effective
    """

    def __init__(self):
        self.llm = OpenAI(temperture=OPENAI_TEMPERATURE)
        self.prompt = PromptTemplate(
                input_variables=["tasks"],
                template="""
You are tasked with prioritizing a task list

Consider the ultimate objective of winning the game.

Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output.

These are the tasks : {tasks}
""")
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm)


    def run(task_storage: SingleTaskListStorage) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            task_storage (SingleTaskListStorage): The current task list

        Returns:
            SingleTaskListStorage: A list of tasks to be completed to beat the game

        """
        task_names = task_storage.get_task_names()
        bullet_string = '\n'
        response = self.chain.run(tasks=bullet_string + bullet_string.join(task_names))
        if not response:
            # Received empty response from priotritization agent. Keeping task list unchanged.
            return task_storage

        new_tasks = openai_task_response_to_list(response)
        return SingleTaskListStorage(new_tasks)


class PlayerAgent(ConversationAgent):
    """
    Agent that executes a task based on the given objective and previous game history
    """

    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate(
                input_variables=["completed_tasks", "history", "input", "objective"],
                template="""
You are playing the 1977 classic Colossal Cave. 

If you ask the same question in a loop, use the "help" command to get out of the loop. Don't get frustrated and only take one item at a time.

The games text parser is limited, keep your commands to one action and 1-3 words. Enter a single command for each prompt. Use the following guide to beat the game. Look around and e what is visible. if your objective is invisible, keep moving. You can only move north, south, east, and west.

Choose the next game input based on the following objective: {objective}

The following objectives have been completed:

{completed_tasks}

Current conversation:
{history}
Human: {input}
AI:""")
        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)


    def run(objective: str, message: str, completed_tasks: SingleTaskListStorage) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            objective (str): next game taks
            message (str): next game output
            completed_tasks (SingleTaskListStorage): list of completed tasks

        Returns:
            str: the next game input

        """
        task_names = completed_tasks.get_task_names()
        bullet_string = '\n'
        return self.conversation.predict(input=message, objective=objective, completed_tasks=task_names)


class TaskCompletionAgent(ConversationAgent):
    """
    Agent that decides if the current objective has been completed
    """

    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are playing the 1977 classic Colossal Cave. 

Decide if the current objective has been completed.

Objective: {objective}

Take into account the game history attached here: {history}

Reply with a simple "COMPLETE" or "INCOMPLETE".
"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)


    def run(objective: str, message: str) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            objective (str): current game objective
            message (str): next game input

        Returns:
            bool: whether the task is complete or not

        """
        return self.conversation.predict(objective=objective, message=message).lower() == "complete"
