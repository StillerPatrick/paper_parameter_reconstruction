from enum import Enum


class DataSets(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class runEnum(Enum):
    RUN = 1
    STOP = 0