from abc import ABC, abstractmethod


class Solution(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def parse_input(self, input_str: str):
        pass

    @abstractmethod
    def solve(self) -> str:
        pass