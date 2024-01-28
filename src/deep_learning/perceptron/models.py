from pydantic import BaseModel
from typing import Callable, List, Optional


class Layer(BaseModel):
    forward: Callable[[float], float]
    backward: Callable[[float], float]
    name: Optional[str] = "layer"


class Loss(Layer):
    forward: Callable[[List[float], List[float]], float]
    backward: Callable[[List[float], List[float]], float]
    name: Optional[str] = "loss_layer"
