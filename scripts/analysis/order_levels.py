"""Recipe order → level (duplicated from compile_experiment_run to avoid PYTHONPATH coupling)."""

ORDER_LEVELS = {
    "level_1": [
        "baked_bell_pepper",
        "baked_sweet_potato",
        "boiled_egg",
        "boiled_mushroom",
        "boiled_sweet_potato",
    ],
    "level_2": [
        "baked_potato_slices",
        "baked_pumpkin_slices",
        "boiled_corn_slices",
        "boiled_green_bean_slices",
        "boiled_potato_slices",
    ],
    "level_3": [
        "baked_bell_pepper_soup",
        "baked_carrot_soup",
        "baked_mushroom_soup",
        "baked_potato_soup",
        "baked_pumpkin_soup",
    ],
    "level_4": [
        "sliced_bell_pepper_and_corn_stew",
        "sliced_bell_pepper_and_lentil_stew",
        "sliced_eggplant_and_chickpea_stew",
        "sliced_pumpkin_and_chickpea_stew",
        "sliced_zucchini_and_chickpea_stew",
    ],
    "level_5": [
        "mashed_broccoli_and_bean_patty",
        "mashed_carrot_and_chickpea_patty",
        "mashed_cauliflower_and_lentil_patty",
        "mashed_potato_and_pea_patty",
        "mashed_sweet_potato_and_bean_patty",
    ],
    "level_6": [
        "potato_carrot_and_onion_patty",
        "romaine_lettuce_pea_and_tomato_patty",
        "sweet_potato_spinach_and_mushroom_patty",
        "taro_bean_and_bell_pepper_patty",
        "zucchini_green_pea_and_onion_patty",
    ],
}


def order_level(name: str) -> str:
    if not name:
        return "unknown"
    for lv, xs in ORDER_LEVELS.items():
        if name in xs:
            return lv
    return "unknown"
