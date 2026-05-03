import unittest

from scripts.analysis.trace_token_tables import (
    build_summary_tables,
    episode_metric_row,
    tgco_rows_for_episode,
)


def _entry(timestep, actions, p0_tokens=0, p1_tokens=0, p0_val=0, p1_val=0, say0="[NOTHING]", say1="[NOTHING]"):
    return {
        "timestamp": timestep,
        "actions": actions,
        "order_list": ["boiled_egg"],
        "statistical_data": {
            "communication": [
                {"call": 1 if p0_tokens else 0, "turn": [say0] if say0 else [], "token": [p0_tokens] if p0_tokens else []},
                {"call": 1 if p1_tokens else 0, "turn": [say1] if say1 else [], "token": [p1_tokens] if p1_tokens else []},
            ],
            "error": [
                {"validator_error": {"error_num": p0_val}, "format_error": {"error_num": 0}},
                {"validator_error": {"error_num": p1_val}, "format_error": {"error_num": 0}},
            ],
        },
        "content": {
            "content": [
                [
                    {"agent": 0, "say": say0, "plan": ""},
                    {"agent": 1, "say": say1, "plan": ""},
                ]
            ]
        },
    }


def _episode():
    return {
        "agents": [
            {"player": "P0", "model": "gpt-4o", "agent_type": "a-tom"},
            {"player": "P1", "model": "claude-sonnet-4-20250514", "agent_type": "baseline"},
        ],
        "total_order_finished": ["boiled_egg"],
        "total_score": 20,
        "total_timestamp": [0, 1, 2, 3],
        "total_action_list": [
            [{"timestamp": 3, "action": "pickup(boiled_egg, pot0)"}, {"timestamp": 4, "action": "deliver_soup()"}],
            [{"timestamp": 2, "action": "pickup(egg, ingredient_dispenser)"}],
        ],
        "content": [
            _entry(
                0,
                ["wait(1)", "pickup(egg, ingredient_dispenser)"],
                p0_tokens=100,
                p1_tokens=25,
                say0="Please pick up an egg from the ingredient dispenser.",
                say1="[NOTHING]",
            ),
            _entry(1, ["wait(1)", "pickup(egg, ingredient_dispenser)"], p0_tokens=40, p1_tokens=10, p1_val=1),
        ],
    }


class EpisodeMetricRowTests(unittest.TestCase):
    def test_episode_row_contains_trace_and_token_metrics(self):
        row = episode_metric_row(_episode(), "fake/boiled_egg/experiment.json")

        self.assertEqual(row["order"], "boiled_egg")
        self.assertEqual(row["order_level"], "level_1")
        self.assertEqual(row["success"], True)
        self.assertEqual(row["num_timesteps"], 4)
        self.assertEqual(row["total_team_actions"], 3)
        self.assertEqual(row["p0_actions"], 2)
        self.assertEqual(row["p1_actions"], 1)
        self.assertEqual(row["sum_validator_errors"], 1)
        self.assertEqual(row["p0_comm_tokens"], 140)
        self.assertEqual(row["p1_comm_tokens"], 35)
        self.assertEqual(row["total_comm_tokens"], 175)
        self.assertEqual(row["p0_model"], "gpt-4o")
        self.assertEqual(row["p1_model"], "claude-sonnet-4-20250514")


class TgcoEpisodeRowsTests(unittest.TestCase):
    def test_tgco_rows_are_enriched_with_episode_metadata(self):
        episode = _episode()
        rows = tgco_rows_for_episode(episode, "fake/boiled_egg/experiment.json", window=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["order_level"], "level_1")
        self.assertEqual(rows[0]["speaker_model"], "gpt-4o")
        self.assertEqual(rows[0]["target_model"], "claude-sonnet-4-20250514")
        self.assertEqual(rows[0]["speaker_comm_tokens"], 140)
        self.assertEqual(rows[0]["target_comm_tokens"], 35)
        self.assertEqual(rows[0]["tgco_label"], "Assisted")
        self.assertEqual(rows[0]["matched_total_action_index"], 0)


class SummaryTableTests(unittest.TestCase):
    def test_builds_level_model_team_token_and_tgco_tables(self):
        episode = _episode()
        run_rows = [episode_metric_row(episode, "fake/boiled_egg/experiment.json")]
        tgco_rows = tgco_rows_for_episode(episode, "fake/boiled_egg/experiment.json", window=5)

        tables = build_summary_tables(run_rows, tgco_rows)

        level = tables["level_summary"]
        self.assertEqual(level.loc[0, "episodes"], 1)
        self.assertEqual(level.loc[0, "success_rate"], 1.0)
        self.assertEqual(level.loc[0, "mean_team_actions"], 3.0)

        model_tokens = tables["token_model_by_level"]
        gpt = model_tokens[model_tokens["model"] == "gpt-4o"].iloc[0]
        self.assertEqual(gpt["slots"], 1)
        self.assertEqual(gpt["total_tokens"], 140)

        team_tokens = tables["token_team_by_level"]
        self.assertEqual(team_tokens.loc[0, "tokens_per_episode"], 175.0)

        tgco_level = tables["tgco_by_level"]
        self.assertEqual(tgco_level.loc[0, "request_units"], 1)
        self.assertEqual(tgco_level.loc[0, "assisted_rate"], 1.0)
        self.assertEqual(tgco_level.loc[0, "tokens_per_effective"], None)

        tgco_speaker = tables["tgco_by_speaker_model_level"]
        self.assertEqual(tgco_speaker.loc[0, "total_tokens"], 140)


if __name__ == "__main__":
    unittest.main()
