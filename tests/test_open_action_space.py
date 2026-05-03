import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_OVERCOOKED = ROOT / "lib" / "overcooked_ai"
if str(LOCAL_OVERCOOKED) not in sys.path:
    sys.path.insert(0, str(LOCAL_OVERCOOKED))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plotly_module = types.ModuleType("plotly")
plotly_graph_objects_module = types.ModuleType("plotly.graph_objects")
sys.modules.setdefault("plotly", plotly_module)
sys.modules.setdefault("plotly.graph_objects", plotly_graph_objects_module)
dtw_module = types.ModuleType("dtw")
dtw_module.dtw = lambda *args, **kwargs: None
dtw_module.dtwPlot = lambda *args, **kwargs: None
dtw_module.stepPattern = None
dtw_module.warp = lambda *args, **kwargs: None
dtw_module.warpArea = lambda *args, **kwargs: None
dtw_module.window = None
sys.modules.setdefault("dtw", dtw_module)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from src.eval_utils import Evaluation


def action_spaces_for_layout(layout_name):
    evaluation = Evaluation.__new__(Evaluation)
    evaluation.layout_name = layout_name
    evaluation.mdp = OvercookedGridworld.from_layout_name(layout_name)
    evaluation.action_space = []
    evaluation.action_space_agent_0 = []
    evaluation.action_space_agent_1 = []
    evaluation._get_action_space()
    return set(evaluation.action_space_agent_0), set(evaluation.action_space_agent_1)


class OpenActionSpaceTests(unittest.TestCase):
    def test_open_layout_gives_both_roles_full_task_action_access(self):
        p0_actions, p1_actions = action_spaces_for_layout("new_env_open")

        shared_required_actions = {
            "pickup(egg,ingredient_dispenser)",
            "pickup(dish,dish_dispenser)",
            "pickup(dish,counter)",
            "put_obj_in_utensil(pot0)",
            "put_obj_in_utensil(oven0)",
            "put_obj_in_utensil(chopping_board0)",
            "put_obj_in_utensil(blender0)",
            "cook(pot0)",
            "bake(oven0)",
            "cut(chopping_board0)",
            "stir(blender0)",
            "fill_dish_with_food(pot0)",
            "fill_dish_with_food(oven0)",
            "deliver_soup()",
        }

        self.assertTrue(shared_required_actions <= p0_actions)
        self.assertTrue(shared_required_actions <= p1_actions)

    def test_required_layout_keeps_original_role_asymmetry(self):
        p0_actions, p1_actions = action_spaces_for_layout("new_env")

        self.assertNotIn("pickup(egg,ingredient_dispenser)", p0_actions)
        self.assertIn("pickup(egg,ingredient_dispenser)", p1_actions)
        self.assertIn("cook(pot0)", p0_actions)
        self.assertNotIn("cook(pot0)", p1_actions)


if __name__ == "__main__":
    unittest.main()
