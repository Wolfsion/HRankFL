import argparse
from abc import ABC, abstractmethod
from typing import List

from env.pathHandler import *
from federal.Message import *
from data.dataProvider import DataLoader

from env.runtimeEnv import *


class FLSimNet:
    ERROR_MESS1 = "Current buffer is null."
    ERROR_MESS2 = "Message in buffer is mismatching."

    def __init__(self) -> None:
        self.buffer = []
        self.tmp = []
        self.curt_index = -1
        self.workers = workers
        self.read = 0
        self.written = 0

    def curt_status(self) -> int:
        return self.curt_index

    def m_consume(self) -> List[FLMessage]:
        assert self.curt_index != -1, self.ERROR_MESS1
        assert isinstance(self.buffer[self.curt_index], List), self.ERROR_MESS2
        curt = deepcopy(self.buffer[self.curt_index])
        self.buffer.pop()
        self.curt_index -= 1
        return curt

    def m_produce(self, mess: FLMessage):
        self.buffer.append(deepcopy(mess))
        self.curt_index += 1

    def w_consume(self) -> FLMessage:
        assert self.curt_index != -1, self.ERROR_MESS1
        assert isinstance(self.buffer[self.curt_index], FLMessage), self.ERROR_MESS2
        self.read += 1
        curt = deepcopy(self.buffer[self.curt_index])
        if self.read == self.workers:
            self.curt_index -= 1
            self.buffer.pop()
            self.read = 0
        return curt

    def w_produce(self, mess: FLMessage):
        self.tmp.append(deepcopy(mess))
        self.written += 1
        if self.written == self.workers:
            self.curt_index += 1
            self.buffer.append(self.tmp)
            self.written = 0


default_bridge = FLSimNet()


class FLMaster(ABC):
    def __init__(self, args: argparse.Namespace,
                 bridge: FLSimNet = default_bridge, save_interval=50):
        self.args = args
        self.save_interval = save_interval

        self.init_algorithm()
        self.save_exp_config()
        self.bridge = bridge

    @abstractmethod
    def init_algorithm(self):
        pass

    @abstractmethod
    def save_exp_config(self):
        pass

    def send_mess(self, mess: FLMessage):
        self.bridge.m_produce(mess)

    def recv_mess(self) -> List[FLMessage]:
        return self.bridge.m_consume()


class FLWorker(ABC):
    def __init__(self, bridge: FLSimNet = default_bridge):
        self.loader: DataLoader = None
        self.init_loader()
        self.init_algorithm()
        self.bridge = bridge

    @abstractmethod
    def init_loader(self):
        pass

    @abstractmethod
    def init_algorithm(self):
        pass

    def send_mess(self, mess: FLMessage):
        self.bridge.w_produce(mess)

    def recv_mess(self) -> FLMessage:
        return self.bridge.w_consume()
