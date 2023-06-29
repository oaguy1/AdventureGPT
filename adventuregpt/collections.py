"""
LangChain LLMChains to process game i/o

Copyright (c) 2023 Lily Hughes-Robinson.

Licensed as free software under the
Apache License, Version 2.0 as detailed in the accompanying README.txt.

MIT License

Copyright (c) 2023 Yohei Nakajima

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import deque
from typing import Dict, List


class SingleTaskListStorage:
    """
    A task list for storing game tasks
    """

    def __init__(self, initial_list: list = []):
        self.tasks = deque(initial_list or [])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]

    def __repr__(self):
        return self.tasks

    def __str__(self):
        return "\n".join([f'{i}. {t["task_name"]}' for i, t in enumerate(self.tasks)])
    
    @classmethod
    def concat(cls, a, b):
        """
        Concatenate the tasks from two lists together
        """
        return cls(a.tasks + b.tasks)