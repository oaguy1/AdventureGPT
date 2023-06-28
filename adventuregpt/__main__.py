"""
Create a shell through which the LLM agents can play Adventure.

Copyright 2010-2015 Brandon Rhodes.
Copyright 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.
"""
import argparse
from adventuregpt.loop import Loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="AdventureGPT",
        description="The game ADVENTURE played by ChatGPT"
    )
    parser.add_argument("-w", "--walkthrough_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()

    try:
        game_loop = Loop(args.walkthrough_path, args.output_path)
        game_loop.loop()
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        game_loop.dump_history()
