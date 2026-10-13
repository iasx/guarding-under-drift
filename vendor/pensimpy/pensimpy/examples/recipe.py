import json

from typing import Dict, List


class Setpoint:
    """
    Encapsulate a value at a point of time
    """

    def __init__(self, time: float, value: float):
        self.time = time
        self.value = value

    def get_sp_dict(self):
        return {"time": self.time, "value": self.value}

    def __str__(self):
        return json.dumps(self.get_sp_dict())


class Recipe:
    """
    A store for a series of setpoints [{"time": , "value": v}...]
    """

    def __init__(self, sp_list: List[Dict[str, float]], name: str):
        assert len(sp_list) > 0, (
            "Can't initiate a recipe instance with an empty setpoint list"
        )
        assert isinstance(name, str), "Can't initiate a recipe instance without a name"
        self.name = name
        self._sp_list = []
        self.index_lookup = {}
        self.add_setpoints(sp_list)
        dict()

    @property
    def sp_list(self):
        return self._sp_list

    @sp_list.setter
    def sp_list(self, val: List[Setpoint]):
        self._sp_list = sorted(val, key=lambda x: x.time)
        self.index_lookup = self.create_index_lookup(self._sp_list)

    @classmethod
    def create_index_lookup(cls, sp_list: List[Setpoint]) -> Dict:
        return dict([(sp.time, i) for i, sp in enumerate(sp_list)])

    def add_setpoint(self, sp: Dict[str, float]):
        sp = Setpoint(**sp)
        assert sp.time not in self.index_lookup.keys(), (
            f"time = {sp.time} already exists"
        )
        self.sp_list = self.sp_list + [sp]

    def add_setpoints(self, sp_list: List[Dict[str, float]]):
        for sp in sp_list:
            self.add_setpoint(sp)

    def get_value_at(self, time: float) -> float:
        """
        Value of the recipe at ``time`` (hours).

        Each setpoint holds *until* its listed time, as in IndPenSim's
        sequential batch control: ``{"time": 24, "value": 150}`` following
        ``{"time": 20, "value": 75}`` means 150 on (20, 24]. Past the last
        setpoint the last value holds.
        """
        assert len(self.sp_list) > 0, "no setpoints available"
        for sp in self.sp_list:
            if time <= sp.time:
                return sp.value
        return self.sp_list[-1].value


class RecipeCombo:
    """
    A collection of recipes
    """

    def __init__(self, recipe_dict: Dict[str, Recipe]):
        assert len(recipe_dict) > 0, (
            "Can't initiate a recipe combo instance with an empty recipe dict"
        )
        self.recipe_dict = recipe_dict

    def get_values_dict_at(self, time: float) -> Dict:
        """
        Get value of each recipe at given time (hours).
        """
        values_dict = {}
        for name, recipe in self.recipe_dict.items():
            values_dict[name] = recipe.get_value_at(time=time)

        return values_dict
