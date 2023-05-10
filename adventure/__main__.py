"""
Offer Adventure at a custom command prompt.

Copyright 2010-2015 Brandon Rhodes.  Licensed as free software under the
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

from adventure import load_advent_dat, agent
from adventure.game import Game


BAUD = 1200


class Loop():

    def __init__(self):
        print("***************** INITIALIZING GAME *******************")
        self.history = []
        self.game_tasks = agent.gametask_creation_agent()
        self.next_game_task()

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
        pprint.pprint(self.history)


    def loop(self):
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
            result = agent.player_agent(self.current_task, self.history)
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
                    completed = agent.task_completion_agent(self.current_task, self.history)

                    if completed:
                        self.next_game_task()


if __name__ == '__main__':
    try:
        loop = Loop()
        loop.loop()
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        loop.dump_history()
