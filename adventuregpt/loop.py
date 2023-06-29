"""
Create a shell through which the LLM agents can play Adventure.

Copyright 2010-2015 Brandon Rhodes.
Copyright 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""
import functools
import operator
import re
import sys
import pprint
from time import sleep

from adventure import load_advent_dat
from adventure.game import Game
from langchain.text_splitter import CharacterTextSplitter

from adventuregpt.chain import (
        GameTaskCreationAgent,
        WalkthroughGameTaskCreationAgent,
        PrioritizationAgent,
        PlayerAgent,
        TaskCompletionAgent
    )
from adventuregpt.collections import SingleTaskListStorage


BAUD = 1200


class Loop():
    """
    The loop that does it all! Inits and plays the game in a loop until the
    game is won or an exception is thrown
    """

    def __init__(self, walkthrough_path: str = "", output_file_path: str = "game_output.txt", verbose: bool = False):
        self.history = []
        self.game_tasks = SingleTaskListStorage()
        self.completed_tasks = SingleTaskListStorage()
        self.walkthrough_path = walkthrough_path
        self.output_file_path = output_file_path
        self.current_task = None
        self.verbose = verbose
        
        self.game = Game()
        load_advent_dat(self.game)
        self.game.start()
        self.curr_game_output = self.game.output

    def run(self, user_input:str) -> str:
        """
        For use as a tool for LangChain agent 
        """
        if not self.game.is_finished:
            # Ask Player Agent what to do next
            self.history.append({"role": "assistant", "content": user_input})
           
            # split lines by newlines and periods and flatten list
            newline_split = user_input.lower().split('\n')
            period_split = [ line.split('.') for line in newline_split ]
            split_lines = functools.reduce(operator.iconcat, period_split, []) 
            
            # We got input! Act on it.
            for line in split_lines:
                words = re.findall(r'\w+', line)
                if words:
                    self.curr_game_output = self.game.do_command(words)
                    self.history.append({
                        "role": "system", "content": self.curr_game_output
                    })
                    self.baudout(f"> {line}\n\n")
                    self.baudout(self.curr_game_output)
        else:
            return "COMPLETED"


    def next_game_task(self):
        """
        Get the next game task
        """
        print("***************** TASK LIST *******************")
        print(self.game_tasks)
        print()
        if self.current_task:
            self.completed_tasks.append({"task_name": self.current_task})

        next_task = self.game_tasks.popleft()
        if next_task:
            self.current_task = next_task["task_name"]

    def baudout(self, s: str):
        """"
        Output text like an old-school terminal
        """
        out = sys.stdout
        for c in s:
            sleep(9. / BAUD)  # 8 bits + 1 stop bit @ the given baud rate
            out.write(c)
            out.flush()

    def dump_history(self):
        """
        Pretty print game output to a file
        """
        with open(self.output_file_path, 'w') as f:
            pprint.pprint(self.history, stream=f)

    def loop(self):
        """
        Main Game Loop
        """
        # initialize agents 
        self.game_task_creation_agent = GameTaskCreationAgent(self.verbose)
        self.walkthrough_game_task_creation_agent = WalkthroughGameTaskCreationAgent(self.verbose)
        self.prioritization_agent = PrioritizationAgent(self.verbose)
        self.player_agent = PlayerAgent(self.verbose)
        self.task_completion_agent = TaskCompletionAgent(self.verbose)
        
        print("***************** INITIALIZING GAME *******************") 
        # if usng walkthrough, read into memory in chunks of 500ish tokens
        # and pass to walkthrough gametask agent, else use gametask_creation_agent
        # with the limited history
        if self.walkthrough_path:
            chunk_size = 300
            chunk_overlap = 10
            
            with open(self.walkthrough_path, 'r') as f:
                walkthrough = f.read()
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                text_chunks = text_splitter.split_text(walkthrough)

            for chunk in text_chunks:
                tasks = self.walkthrough_game_task_creation_agent.run(chunk)
                self.game_tasks.concat(tasks)
        else:
            self.game_tasks = self.game_task_creation_agent.run(self.player_agent.memory, self.curr_game_output)

        self.next_game_task()
        self.baudout(self.curr_game_output)
        self.history.append({
            "role": "system", "content": self.curr_game_output
        })

        while not self.game.is_finished:
            # Ask Player Agent what to do next
            result = self.player_agent.run(self.current_task, self.curr_game_output, self.completed_tasks)
            self.history.append({"role": "assistant", "content": result})
           
            # split lines by newlines and periods and flatten list
            newline_split = result.lower().split('\n')
            period_split = [ line.split('.') for line in newline_split ]
            split_lines = functools.reduce(operator.iconcat, period_split, []) 
            
            # We got input! Act on it.
            for line in split_lines:
                words = re.findall(r'\w+', line)
                if words:
                    self.curr_game_output = self.game.do_command(words)
                    self.history.append({
                        "role": "system", "content": self.curr_game_output
                    })
                    self.baudout(f"> {line}\n\n")
                    self.baudout(self.curr_game_output)

                    # if not using a walthrough, come up with more tasks and prioritize
                    if not self.walkthrough_path:
                        new_tasks = self.game_task_creation_agent.run(self.player_agent.memory, self.curr_game_output)
                        curr_tasks = self.game_tasks
                        self.game_tasks = SingleTaskListStorage.concat(curr_tasks, new_tasks)
                        self.game_tasks = self.prioritization_agent.run(self.game_tasks)
                    
                    completed = self.task_completion_agent.run(self.current_task, self.player_agent.memory, self.curr_game_output)
                    if completed:
                        self.next_game_task()