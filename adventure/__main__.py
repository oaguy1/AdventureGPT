"""
Offer Adventure at a custom command prompt.

Copyright 2010-2015 Brandon Rhodes.  Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""
import argparse
import os
import re
import readline
import sys
import tiktoken
import openai
import pprint
from time import sleep

from adventure import load_advent_dat
from adventure.game import Game

BAUD = 1200
LLM = "gpt-3.5-turbo"
MAX_LLM_TOKEN = 4096
INTRO = """

You are playing the 1977 classic Colossal Cave. 

If you ask the same question in a loop, use the "help" command to get out of the loop. Don't get frustrated and only take one item at a time.


The games text parser is limited, keep your commands to one action and 1-3 words. Enter a single command for each prompt.. Use the following guide to beat the game. Look around and e what is visible. if your objective is invisible, keep moving. You can only move north, south, east, and west.

----------------------

Getting In to the Caves


'Enter' building, and 'take' the keys and lamp. Now go 'out' of the building.
Go down, down to the slit in the rock. Go South to the grate. Unlock grate with key, open grate. Go down.

Go west, take the cage. Go west again, turn on the lamp. Grab the rod and say XYZZY. POOF you're back in the well house! Say XYZZY again to return to the Debris Room.

Go west to the sloping room. Drop the rod. Go west to the Orange River Chamber. Grab the bird (it's afraid of the rod). Now go east and get the rod, and head back west again. Go down into the pit. You're in the caves!


Level 1 - Snakes and Plughs


Level 1 is pretty easy. You start at the Hall of Mists. Head south and grab the gold (+7 score, +9 to drop off). Head north. Hall of the Mountain King, with a snake. Open the cage so the bird drives away the snake. OK, passages in all directions. 

To the west is a West Side Chamber with coins (+7 score, +5 drop off). Grab and go east again. To the south is the South Side Chamber with jewelry (+7 score, +5 drop off). Grab it and go north again. To the north is a chamber with a hole in it, with silver. Grab the silver (+7 score, +5 drop off). 

Go north one more room, to the Y2 room. Say "plugh" to zap back to the Well House. Drop off your treasures. 

Drop off the keys for now, then say "plugh" to return to Y2. 

If you hit a dwarf in here anywhere, and he throws an axe, take it! Then when you see him again, "throw axe at dwarf" to get him. If you miss, just "take axe" and throw it again. 

OK, now head to the Fissure, west of the Hall of Mists. Wave the Rod to make a crystal bridge appear. Cross it to grab the diamonds. Now head west and south to get into the Pirate Maze. You want to go EAST, SOUTH, then NORTH to get to the chest. When you take it, head SOUTHEAST, WEST and SOUTH to get down to the Orange Room. Just go east twice and then say XYZZY to pop back to the well house. 

"""

class Loop():

    def __init__(self, args):
        self.history = [
            {
                "role": "system", "content": INTRO
            }
        ]
        self.loop(args)

    def baudout(self, s: str):
        out = sys.stdout
        for c in s:
            sleep(9. / BAUD)  # 8 bits + 1 stop bit @ the given baud rate
            out.write(c)
            out.flush()

    def dump_history(self):
        pprint.pprint(self.history)

    def loop(self, args):
        parser = argparse.ArgumentParser(
            description='Adventure into the Colossal Caves.',
            prog='{} -m adventure'.format(os.path.basename(sys.executable)))
        parser.add_argument(
            'savefile', nargs='?', help='The filename of game you have saved.')
        args = parser.parse_args(args)

        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            api_key = input("OpenAI Key:")
            os.environ["OPENAI_API_KEY"] = api_key

        openai.api_key = api_key

        if args.savefile is None:
            self.game = Game()
            load_advent_dat(self.game)
            self.game.start()
            next_input = self.game.output
            self.baudout(next_input)
            self.history.append({
                "role": "system", "content": next_input
            })
        else:
            self.game = Game.resume(args.savefile)
            self.baudout('GAME RESTORED\n')

        while not self.game.is_finished:
            lines = "" 
            while not lines:
                try:
                    response = openai.ChatCompletion.create(model=LLM, messages=self.history, temperature=0)
                    lines = response['choices'][0]['message']['content'].lower() 
                except:
                    self.baudout("OpenAI is being cranky...\n")
                    sleep(10)
            
            self.history.append(response['choices'][0]['message'])
            for line in lines.split('.'):
                words = re.findall(r'\w+', line)
                if words:
                    command_output = self.game.do_command(words)
                    self.history.append({
                        "role": "system", "content": command_output
                    })
                    self.baudout(f"> {line}\n\n")
                    self.baudout(command_output)

if __name__ == '__main__':
    try:
        loop = Loop(sys.argv[1:])
    except EOFError:
        pass
    except:
        loop.dump_history()
