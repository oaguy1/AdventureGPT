"""
Create a shell through which the LLM agents can play Adventure.

Copyright 2010-2015 Brandon Rhodes.
Copyright 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""
import argparse
import functools
import os
import operator
import re
import readline
import sys
import pprint
from time import sleep

from adventure import load_advent_dat
from adventure.agent import gametask_creation_agent, player_agent, task_completion_agent, chunk_tokens_from_string, SingleTaskListStorage
from adventure.game import Game


BAUD = 1200


class Loop():
    """
    The loop that does it all! Inits and plays the game in a loop until the
    game is won or an exception is thrown
    """

    def __init__(self, walkthrough_path: str = "", output_file_path: str = "game_output.txt"):
        self.history = []
        self.game_tasks = SingleTaskListStorage()
        self.completed_tasks = SingleTaskListStorage()
        self.walkthrough_path = walkthrough_path
        self.output_file_path = output_file_path

    def next_game_task(self):
        print("***************** TASK LIST *******************")
        print(self.game_tasks)
        print()
        next_task = self.game_tasks.popleft()
        if next_task:
            self.current_task = next_task["task_name"]

    def baudout(self, s: str):
        out = sys.stdout
        for c in s:
            sleep(9. / BAUD)  # 8 bits + 1 stop bit @ the given baud rate
            out.write(c)
            out.flush()

    def dump_history(self):
        with open(self.output_file_path, 'w') as f:
            pprint.pprint(self.history, stream=f)

    def loop(self):
        """ Main Game Loop """
        print("***************** INITIALIZING GAME *******************")
        text_chunks = []
        chunk_size = 500
        
        if self.walkthrough_path:
            with open(self.walkthrough_path, 'r') as f:
                curr_chunk = []
                for line in f:
                    tokens = line.split()
                    curr_chunk += tokens

                    if len(curr_chunk) >= chunk_size:
                        text_chunks.append(" ".join(curr_chunk))
                        curr_chunk = []


        for chunk in text_chunks:
            tasks = gametask_creation_agent(chunk)
            self.game_tasks.concat(tasks)

        self.next_game_task()
        
        self.game = Game()
        load_advent_dat(self.game)
        self.game.start()
        next_input = self.game.output
        self.baudout(next_input)
        self.history.append({
            "role": "system", "content": next_input
        })

        while not self.game.is_finished:
            # Ask Player Agent what to do next
            result = player_agent(self.current_task, self.history)
            self.history.append({"role": "assistant", "content": result})
           
            # split lines by newlines and periods and flatten list
            newline_split = result.lower().split('\n')
            period_split = [ line.split('.') for line in newline_split ]
            split_lines = functools.reduce(operator.iconcat, period_split, []) 
            
            # We got input! Act on it.
            for line in split_lines:
                words = re.findall(r'\w+', line)
                if words:
                    command_output = self.game.do_command(words)
                    self.history.append({
                        "role": "system", "content": command_output
                    })
                    self.baudout(f"> {line}\n\n")
                    self.baudout(command_output)
                    completed = task_completion_agent(self.current_task, self.history)

                    if completed:
                        self.next_game_task()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="AdventureGPT",
        description="A modified port of the game ADVENTURE played by ChatGPT"
    )
    parser.add_argument("walkthrough_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()

    try:
        loop = Loop(args.walkthrough_path, args.output_path)
        loop.loop()
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        loop.dump_history()
