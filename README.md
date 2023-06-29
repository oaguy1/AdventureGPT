# AdventureGPT

An set of autonomous agents designed to play the 1977 game
ADVENTURE or [Colossal Cave Adventure](https://en.m.wikipedia.org/wiki/Colossal_Cave_Adventure). The code base here is based off code from the following repos:

* [python-adventure](https://github.com/brandon-rhodes/python-adventure)
* [BabyAGI](https://github.com/yoheinakajima/babyagi)

That said, code from other repos has been heavily modified and all modifications are licensed under the Apache 2.0 License.

Currently, the only requirement is an OpenAI API key and to have it set as the `OPENAI_API_KEY` environment variable.

## Running

Run the code by cloning the repository, navigate to the cloned repository, and run the following commands:

```bash
python -m pip install -r requirements.txt
python -m adventuregpt
```
Add a `--help` flag to see the command line arguments.

## TODO

Here is a list of eventual goals for the project:

* Add map/location agent
* Win the game
* Add a curses style UI for displaying tasks and prompts while showing gameplay in its own pane
* Better utilization of context/memory in prompts, maybe storing results in a vector DB
* Add a "replay" feature to take a history dump from a run and replay the main game text (for media creation)

## Contributing

This project is a playground for me to learn more about prompt engineering and play with OpenAI's models. That said, I am interested in pushing this to the absolute limit of what is possible. If you want to contribute, make a fork and create pull requests. I will do my best to be a good steward of the project and comment on pull requests within a timely manner.
