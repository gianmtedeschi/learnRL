"""Utils functions"""


class RhoElem:
    MEAN = 0
    STD = 1


class LearnRates:
    RHO = 0
    LAMBDA = 1
    ETA = 2


class TrajectoryResults:
    PERF = 0
    RewList = 1
    ScoreList = 2
    StateList = 3


class ParamSamplerResults:
    THETA = 0
    PERF = 1


class SplitResults:
    Gradient = 0
    RewardTrajectories = 1
    SplitThetas = 2
    ValidTrajectories = 3
