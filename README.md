# AdventureGPT

An set of autonomous agents designed to play the 1977 game
ADVENTURE or [Colossal Cave Adventure](https://en.m.wikipedia.org/wiki/Colossal_Cave_Adventure). Currently utilizing OpenAI
python SDK,the eventual goal is to switch to LangChain.
The code base here is based off code from the following repos:

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
