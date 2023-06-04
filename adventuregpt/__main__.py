"""
Create a shell through which the LLM agents can play Adventure.

Copyright 2010-2015 Brandon Rhodes.
Copyright 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""
import argparse
import functools
import operator
import re
import sys
import pprint
from time import sleep

from adventure import load_advent_dat
from adventure.game import Game

from adventuregpt.agent import (
        GametaskCreationAgent,
        WalkthroughGametaskCreationAgent,
        PrioritizationAgent,
        PlayerAgent,
        TaskCompletionAgent,
        SingleTaskListStorage
    )


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
        self.current_task = None
        self.game_task_creation_agent = GametaskCreationAgent()
        self.walkthrough_game_task_creation_agent = WalkthroughGametaskCreationAgent()
        self.prioritization_agent = PrioritizationAgent()
        self.player_agent = PlayerAgent()
        self.task_creation_agent = TaskCompletionAgent()

    def next_game_task(self):
        print("***************** TASK LIST *******************")
        print(self.game_tasks)
        print()
        if self.current_task:
            self.completed_tasks.append({"task_name": self.current_task})

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
        # if usng walkthrough, read into memory in chunks of 500ish tokens
        # and pass to walkthrough gametask agent, else use gametask_creation_agent
        # with the limited history
        if self.walkthrough_path:
            text_chunks = []
            chunk_size = 500
            
            with open(self.walkthrough_path, 'r') as f:
                curr_chunk = []
                for line in f:
                    tokens = line.split()
                    curr_chunk += tokens

                    if len(curr_chunk) >= chunk_size:
                        text_chunks.append(" ".join(curr_chunk))
                        curr_chunk = []


            for chunk in text_chunks:
                tasks = self.walkthrough_gametask_creation_agent.run(chunk)
                self.game_tasks.concat(tasks)
        else:
            self.game_tasks = self.gametask_creation_agent.run(self.history)

        self.next_game_task()
        
        self.game = Game()
        load_advent_dat(self.game)
        self.game.start()
        curr_game_output = self.game.output
        self.baudout(curr_game_output)
        self.history.append({
            "role": "system", "content": curr_game_output
        })

        while not self.game.is_finished:
            # Ask Player Agent what to do next
            result = self.player_agent.run(self.current_task, curr_game_output, self.completed_tasks)
            self.history.append({"role": "assistant", "content": result})
           
            # split lines by newlines and periods and flatten list
            newline_split = result.lower().split('\n')
            period_split = [ line.split('.') for line in newline_split ]
            split_lines = functools.reduce(operator.iconcat, period_split, []) 
            
            # We got input! Act on it.
            for line in split_lines:
                words = re.findall(r'\w+', line)
                if words:
                    curr_game_output = self.game.do_command(words)
                    self.history.append({
                        "role": "system", "content": command_output
                    })
                    self.baudout(f"> {line}\n\n")
                    self.baudout(curr_game_output)

                    # if not using a walthrough, come up with more tasks and prioritize
                    if not self.walkthrough_path:
                        new_tasks = self.gametask_creation_agent.run(curr_game_output)
                        self.game_tasks.concat(new_tasks)
                        self.game_tasks = self.prioritization_agent.run(self.game_tasks, curr_game_output)
                    
                    completed = self.task_completion_agent.run(self.current_task, curr_game_output)
                    if completed:
                        self.next_game_task()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="AdventureGPT",
        description="The game ADVENTURE played by ChatGPT"
    )
    parser.add_argument("-w", "--walkthrough_path")
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
