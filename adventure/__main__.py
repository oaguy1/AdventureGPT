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
from time import sleep

from adventure import load_advent_dat
from adventure.game import Game

BAUD = 1200
LLM = "gpt-3.5-turbo"
MAX_LLM_TOKEN = 4096
INTRO = """

You are playing the 1977 classic Colossal Cave. 

IF YOU WANT TO END YOUR ADVENTURE EARLY, SAY "QUIT". TO SUSPEND YOUR
ADVENTURE SUCH THAT YOU CAN CONTINUE LATER, SAY "SUSPEND" (OR "PAUSE"
OR "SAVE").  TO SEE WHAT HOURS THE CAVE IS NORMALLY OPEN, SAY "HOURS".
TO SEE HOW WELL YOU'RE DOING, SAY "SCORE".  TO GET FULL CREDIT FOR A
TREASURE, YOU MUST HAVE LEFT IT SAFELY IN THE BUILDING, THOUGH YOU GET
PARTIAL CREDIT JUST FOR LOCATING IT.  YOU LOSE POINTS FOR GETTING 
KILLED, OR FOR QUITTING, THOUGH THE FORMER COSTS YOU MORE.  THERE ARE
ALSO POINTS BASED ON HOW MUCH (IF ANY) OF THE CAVE YOU'VE MANAGED TO 
EXPLORE; IN PARTICULAR, THERE IS A LARGE BONUS JUST FOR GETTING IN (TO                                                     
DISTINGUISH THE BEGINNERS FROM THE REST OF THE PACK), AND THERE ARE
OTHER WAYS TO DETERMINE WHETHER YOU'VE BEEN THROUGH SOME OF THE MORE
HARROWING SECTIONS.  IF YOU THINK YOU'VE FOUND ALL THE TREASURES, JUST
KEEP EXPLORING FOR A WHILE.  IF NOTHING INTERESTING HAPPENS, YOU      
HAVEN'T FOUND THEM ALL YET.  IF SOMETHING INTERESTING *DOES* HAPPEN,
IT MEANS YOU'RE GETTING A BONUS AND HAVE AN OPPORTUNITY TO GARNER MANY
MORE POINTS IN THE MASTER'S SECTION.  I MAY OCCASIONALLY OFFER HINTS
IF YOU SEEM TO BE HAVING TROUBLE.  IF I DO, I'LL WARN YOU IN ADVANCE
HOW MUCH IT WILL AFFECT YOUR SCORE TO ACCEPT THE HINTS.  FINALLY, TO
SAVE PAPER, YOU MAY SPECIFY "BRIEF", WHICH TELLS ME NEVER TO REPEAT   
THE FULL DESCRIPTION OF A PLACE UNLESS YOU EXPLICITLY ASK ME TO.
I KNOW OF PLACES, ACTIONS, AND THINGS.  MOST OF MY VOCABULARY
DESCRIBES PLACES AND IS USED TO MOVE YOU THERE.  TO MOVE, TRY WORDS
LIKE FOREST, BUILDING, DOWNSTREAM, ENTER, EAST, WEST, NORTH, SOUTH,
UP, OR DOWN.  I KNOW ABOUT A FEW SPECIAL OBJECTS, LIKE A BLACK ROD
HIDDEN IN THE CAVE.  THESE OBJECTS CAN BE MANIPULATED USING SOME OF
THE ACTION WORDS THAT I KNOW. USUALLY YOU WILL NEED TO GIVE BOTH THE
OBJECT AND ACTION WORDS (IN EITHER ORDER), BUT SOMETIMES I CAN INFER                                                       
THE OBJECT FROM THE VERB ALONE.  SOME OBJECTS ALSO IMPLY VERBS; IN
PARTICULAR, "INVENTORY" IMPLIES "TAKE INVENTORY", WHICH CAUSES ME TO
GIVE YOU A LIST OF WHAT YOU'RE CARRYING.  THE OBJECTS HAVE SIDE
EFFECTS; FOR INSTANCE, THE ROD SCARES THE BIRD.  USUALLY PEOPLE HAVING
TROUBLE MOVING JUST NEED TO TRY A FEW MORE WORDS.  USUALLY PEOPLE
TRYING UNSUCCESSFULLY TO MANIPULATE AN OBJECT ARE ATTEMPTING SOMETHING
BEYOND THEIR (OR MY!) CAPABILITIES AND SHOULD TRY A COMPLETELY
DIFFERENT TACK.  TO SPEED THE GAME YOU CAN SOMETIMES MOVE LONG
DISTANCES WITH A SINGLE WORD. FOR EXAMPLE, "BUILDING" USUALLY GETS
YOU TO THE BUILDING FROM ANYWHERE ABOVE GROUND EXCEPT WHEN LOST IN THE
FOREST.  ALSO, NOTE THAT CAVE PASSAGES TURN A LOT, AND THAT LEAVING A
ROOM TO THE NORTH DOES NOT GUARANTEE ENTERING THE NEXT FROM THE SOUTH.
GOOD LUCK!

Answer the first questionwith "Y"

"""

class Loop():

    def __init__(self, args):
        self.history = [
            {
                "role": "system", "content": INTRO
            }
        ]
        self.loop(args)


    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


    def baudout(self, s: str):
        out = sys.stdout
        for c in s:
            sleep(9. / BAUD)  # 8 bits + 1 stop bit @ the given baud rate
            out.write(c)
            out.flush()

    def loop(self, args):
        parser = argparse.ArgumentParser(
            description='Adventure into the Colossal Caves.',
            prog='{} -m adventure'.format(os.path.basename(sys.executable)))
        parser.add_argument(
            'savefile', nargs='?', help='The filename of game you have saved.')
        args = parser.parse_args(args)

        openai.api_key = input("OpenAI Key:")

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
            line = "" 
            while not line:
                try:
                    response = openai.ChatCompletion.create(model=LLM, messages=self.history, temperature=0)
                    line = response['choices'][0]['message']['content'].lower() 
                except:
                    self.baudout("OpenAI is being cranky...\n")
                    sleep(10)
                finally:
                    self.history.append(response['choices'][0]['message'])

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
        Loop(sys.argv[1:])
    except EOFError:
        pass
