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
8. Carrying food, ax, bottle of water, key (all for later) and lantern, visit
the software den (Microsoft Version only). Don't mess with anything--Software
types are a weird lot.  Get magic word LWPI.  Works only from here.
9. Water plant twice. Get water for the second pass at the waterfall.
10. Attack dragon. Yes. Get rug.
11. Get oil (in now empty bottle). Climb the plant.
12. Oil door. Open door. Drop bottle. While here and while carrying golden eggs,

enter FEE, FIE, FOE, FOO, one word at a time. Check your inventory.  Go back to
where you first found the eggs.
13. Throw eggs (to troll).  Cross bridge before the FEE, FIE, whatever.
14. Feed bear.  Unlock chain (with key).  Get golden chain. Drop key.  Get bear.

Don't forget the rare spices.  At volcano view, read and remember the words of
fire.
15. Release bear.  Don't try crossing the bridge with him on the chain.
16. Open oyster (with trident).  Pearl will roll down into the cul-de-sac.
17. Insert coin to get a replacement battery for your lamp, if necessary.  Coins

are a treasure, however, and you won't ge them back, so try beating the game
with the original batteries only.
18. Never drop vase unless you have already dropped the pillow.
19. Drop everything in order to enter.  Get emerald.
20. Say PLOVER.  Get pyramid.  PLOVER, PLUGH and pi-tooie!
21. You must traverse the maze to get Pirate's Treasure Chest, which doesn't
appear until he's robbed you.  Return via the Pit and XYZZY.  Drop all treasures

in the House.
22. Drop magazine in Witt's End for a point.  Get out by entering all different
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
