"""
LangChain LLMChains to process game i/o

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

from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseMessage
from pydantic import root_validator
from typing import Dict, List

from adventuregpt.collections import SingleTaskListStorage

OPENAI_TEMPERATURE = 0.0


api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    api_key = input("OpenAI Key:")
    os.environ["OPENAI_API_KEY"] = api_key

def openai_task_response_to_list(response: str):
    """
    Convert a list of tasks from the format:

    1. task1
    2. task2

    into a list of tasks for later processing
    """
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


def langchain_history_to_prompt(history: List[BaseMessage]) -> str:
    """
    Given a set of historical messages from a LangChain memory class, return a nicely
    formatted string for inclusion in a prompt, attempts to match what LangChain does for
    ConversationChains w/ memory
    """
    prompt = ""
    for msg in history:
        # remove trailing whitespace
        cleaned_content = msg.content
        while cleaned_content[-1] == '\n':
            cleaned_content = cleaned_content[:-1]

        if msg.type == "ai":
            prompt += f"{msg.type.upper()}: {cleaned_content}\n"
        else:
            prompt += f"{msg.type.capitalize()}: {cleaned_content}\n"

        if msg.type == "human":
            prompt += "\n\n"

    return prompt

class WalkthroughGameTaskCreationAgent:
    """
    Agent that creates a list of game tasks to complete based on a given walthrough.
    """

    def __init__(self, verbose: bool = False):
        self.llm = OpenAI(temperature=OPENAI_TEMPERATURE)
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
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=verbose)


    def run(self, walkthrough: str) -> SingleTaskListStorage:
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

    def __init__(self, verbose: bool = False):
        self.llm = OpenAI(temperature=OPENAI_TEMPERATURE)
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
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=verbose)


    def run(self, task_storage: SingleTaskListStorage) -> SingleTaskListStorage:
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


class CustomConversationChain(ConversationChain):
    """
    Custom ConversationChain with more variables, removes validation
    """
    
    @root_validator
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """
        don't perform the validation, just pass the values
        """
        return values


class PlayerAgent:
    """
    Agent that executes a task based on the given objective and previous game history
    """

    def __init__(self, verbose: bool = False):
        self.llm = ChatOpenAI(temperature=OPENAI_TEMPERATURE)
        self.memory = ConversationBufferWindowMemory(return_messages=True, input_key="input", k=15)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are playing the 1977 classic Colossal Cave. 

If you ask the same question in a loop, use the "help" command to get out of the loop. Don't get frustrated and only take one item at a time.

The games text parser is limited, keep your commands to one action and 1-3 words. Enter a single command for each prompt. Use the following guide to beat the game. Look around and e what is visible. if your objective is invisible, keep moving. You can only move north, south, east, and west.

Choose the next game input based on the following objective: {objective}

The following objectives have been completed:

{completed_tasks}

"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.conversation = CustomConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=verbose)


    def run(self, objective: str, message: str, completed_tasks: SingleTaskListStorage) -> SingleTaskListStorage:
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


class TaskCompletionAgent:
    """
    Agent that decides if the current objective has been completed
    """

    def __init__(self, verbose: bool = False):
        self.llm = OpenAI(temperature=OPENAI_TEMPERATURE)
        self.prompt = PromptTemplate(
                input_variables=["objective", "history", "input"],
                template="""
You are playing the 1977 classic Colossal Cave. 

Decide if the current objective has been completed.

Objective: {objective}

Reply with a simple "COMPLETE" or "INCOMPLETE".

Below is the history of the game interactions:

{history}
Human: {input}""")
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=verbose)


    def run(self, objective: str, history: ConversationBufferWindowMemory, message: str,) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            objective (str): current game objective
            message (str): next game input

        Returns:
            bool: whether the task is complete or not

        """
        formatted_history = langchain_history_to_prompt(history.load_memory_variables({})['history'])
        return self.chain.run(objective=objective, history=formatted_history, input=message.strip()).lower() == "complete"


class GameTaskCreationAgent:
    """
    Agent that creates a list of game tasks to complete based game history
    """

    def __init__(self, verbose: bool = False):
        self.llm = ChatOpenAI(temperature=OPENAI_TEMPERATURE)
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""
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

Take into account the game history attached here:

{history}
Human: {input}""")
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=verbose)


    def run(self, history: ConversationBufferWindowMemory,  message: str) -> SingleTaskListStorage:
        """
        Creates a list of game tasks to complete based game history
        
        Args:
            message (str): next game input

        Returns:
            SingleTaskListStorage: A list of tasks to be completed to beat the game

        """
        formatted_history = langchain_history_to_prompt(history.load_memory_variables({})['history'])
        response = self.chain.run(history=formatted_history, input=message)
        task_list = openai_task_response_to_list(response)
        return SingleTaskListStorage(task_list)
