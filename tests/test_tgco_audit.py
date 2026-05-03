import json
import unittest
from pathlib import Path

from scripts.analysis.tgco_audit import (
    TGCO_ASSISTED,
    TGCO_EFFECTIVE,
    TGCO_INEFFECTIVE,
    TGCO_REDUNDANT,
    audit_run,
    classify_request,
    parse_request_units,
)


class TgcoParserTests(unittest.TestCase):
    def test_splits_explicit_partner_request_into_action_units(self):
        units = parse_request_units(
            "Please pick up an egg from the ingredient dispenser and place it on the counter.",
            speaker=0,
            timestep=0,
        )

        self.assertEqual([u.action for u in units], ["pickup", "place_obj_on_counter"])
        self.assertEqual([u.obj for u in units], ["egg", "egg"])
        self.assertEqual([u.target_agent for u in units], [1, 1])

    def test_ignores_self_plans_chatter_and_waits(self):
        cases = [
            "I'll pick up the egg.",
            "I'm placing the egg on the counter now.",
            "Good job.",
            "Please wait while the egg cooks.",
            "Please wait for the egg to cook. I'll let you know once it's ready.",
            "[NOTHING]",
        ]

        for message in cases:
            with self.subTest(message=message):
                self.assertEqual(parse_request_units(message, speaker=0, timestep=0), [])

    def test_deliver_request_with_boiled_egg_is_not_parsed_as_cook(self):
        units = parse_request_units(
            "Please make sure you're delivering the boiled_egg you're currently holding.",
            speaker=1,
            timestep=14,
        )

        self.assertEqual([u.action for u in units], ["deliver_soup"])
        self.assertEqual([u.obj for u in units], ["boiled_egg"])


class TgcoMatchingTests(unittest.TestCase):
    def test_effective_uses_target_total_action_list_only(self):
        request = parse_request_units(
            "Please pick up the egg from the ingredient dispenser.",
            speaker=0,
            timestep=0,
        )[0]
        total_action_list = [
            [{"timestamp": 1, "action": "pickup(onion, ingredient_dispenser)"}],
            [{"timestamp": 3, "action": "pickup(egg, ingredient_dispenser)"}],
        ]

        row = classify_request(
            request,
            total_action_list=total_action_list,
            content=[],
            window=5,
        )

        self.assertEqual(row["tgco_label"], TGCO_EFFECTIVE)
        self.assertEqual(row["matched_total_action_agent"], 1)
        self.assertEqual(row["matched_total_action_index"], 0)
        self.assertEqual(row["matched_action_timestep"], 3)
        self.assertEqual(row["validator_errors_between"], 0)

    def test_late_or_wrong_action_is_ineffective(self):
        request = parse_request_units(
            "Please pick up the egg from the ingredient dispenser.",
            speaker=0,
            timestep=0,
        )[0]
        total_action_list = [
            [],
            [
                {"timestamp": 2, "action": "pickup(onion, ingredient_dispenser)"},
                {"timestamp": 8, "action": "pickup(egg, ingredient_dispenser)"},
            ],
        ]

        row = classify_request(
            request,
            total_action_list=total_action_list,
            content=[],
            window=5,
        )

        self.assertEqual(row["tgco_label"], TGCO_INEFFECTIVE)
        self.assertEqual(row["matched_action"], "")

    def test_assisted_when_validator_error_occurs_before_match(self):
        request = parse_request_units(
            "Please place the egg on the counter.",
            speaker=0,
            timestep=4,
        )[0]
        total_action_list = [
            [],
            [{"timestamp": 6, "action": "place_obj_on_counter()"}],
        ]
        content = [
            {
                "timestamp": 5,
                "statistical_data": {
                    "error": [
                        {"validator_error": {"error_num": 0}},
                        {"validator_error": {"error_num": 1}},
                    ]
                },
            }
        ]

        row = classify_request(
            request,
            total_action_list=total_action_list,
            content=content,
            window=5,
        )

        self.assertEqual(row["tgco_label"], TGCO_ASSISTED)
        self.assertEqual(row["validator_errors_between"], 1)

    def test_pot_request_matches_numbered_pot_action(self):
        request = parse_request_units(
            "Please pick it up and put it in the pot to start boiling.",
            speaker=1,
            timestep=4,
        )[0]
        total_action_list = [
            [{"timestamp": 8, "action": "put_obj_in_utensil(pot0)"}],
            [],
        ]

        row = classify_request(
            request,
            total_action_list=total_action_list,
            content=[],
            window=5,
        )

        self.assertEqual(row["tgco_label"], TGCO_EFFECTIVE)
        self.assertEqual(row["matched_action"], "put_obj_in_utensil(pot0)")

    def test_repeated_request_to_same_action_is_redundant(self):
        first, second = [
            parse_request_units(
                "Please place the egg on the counter.",
                speaker=0,
                timestep=t,
            )[0]
            for t in (0, 1)
        ]
        total_action_list = [
            [],
            [{"timestamp": 3, "action": "place_obj_on_counter()"}],
        ]

        first_row = classify_request(
            first,
            total_action_list=total_action_list,
            content=[],
            window=5,
        )
        second_row = classify_request(
            second,
            total_action_list=total_action_list,
            content=[],
            window=5,
            consumed_matches={(1, 0)},
        )

        self.assertEqual(first_row["tgco_label"], TGCO_EFFECTIVE)
        self.assertEqual(second_row["tgco_label"], TGCO_REDUNDANT)


class TgcoRealLogTests(unittest.TestCase):
    def test_real_log_emits_evidence_rows_from_total_action_list(self):
        path = Path(
            "src/data/reflexion_gpt-4o__reflexion_claude-sonnet-4-20250514/"
            "boiled_egg/experiment_2026-04-23_09-15-04_boiled_egg.json"
        )
        data = json.loads(path.read_text())

        rows = audit_run(data, source_file=str(path), window=5)

        first_pickup = next(
            row
            for row in rows
            if row["speaker"] == 0
            and row["target_agent"] == 1
            and row["request_action"] == "pickup"
            and row["request_object"] == "egg"
        )
        self.assertEqual(first_pickup["tgco_label"], TGCO_ASSISTED)
        self.assertEqual(first_pickup["matched_total_action_agent"], 1)
        self.assertEqual(first_pickup["matched_total_action_index"], 0)
        self.assertEqual(first_pickup["matched_action_timestep"], 3)
        self.assertEqual(first_pickup["validator_errors_between"], 1)
        self.assertIn("total_action_list", first_pickup["evidence_source"])


if __name__ == "__main__":
    unittest.main()
