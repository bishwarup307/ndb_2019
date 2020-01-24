'''
 Created on Fri Dec 27 2019
 __author__: bishwarup
'''
import os
import numpy as np
from collections import defaultdict

ROOT_DIR = "/home/bishwarup/kaggle/ndb2019"
CACHE_DIR = os.path.join(ROOT_DIR, '.cache')
MODEL_PREFIX = 'LGB'
N_BAG = 10

event_code_type_dict = {
    2000 : "starter",
    2010 : "quit",
    2020 : "round",
    2025 : "misc_gameplay",
    2030 : "round",
    2035 : "misc_gameplay",
    2040 : "level",
    2050 : "level",
    2060 : "tutorial",
    2070 : "tutorial",
    2075 : "skip_tutorial",
    2080 : "movie",
    2083 : "movie",
    2081 : "skip_movie",
    3010 : "system_instruction",
    3110 : "system_instruction",
    3020 : "incorrect_feedback",
    3120 : "incorrect_feedback",
    3021 : "correct_feedback",
    3121 : "correct_feedback",
    
    4010: "start_game",
    4020: "misc_gameplay",
    4021: "misc_gameplay",
    4022: "misc_gameplay",
    4025: "misc_gameplay",
    4030: "misc_gameplay",
    4031: "misc_gameplay",
    4230: "misc_gameplay",
    4235: "misc_gameplay",
    5000: "misc_gameplay",
    5010: "misc_gameplay",
    
    4040: "misc_gameplay_strategy",
    4045: "misc_gameplay_strategy",
    4050: "misc_gameplay_strategy",
    
    4035 : "incorrect_drag",
    4070 : "incorrect_drag",
    
    4080 : "hover_mouse",
    4090 : "seek_help",
    
    4095 : "play_again",
    4220 : "post_victory",
    4100 : "finish_assessment",
    4110 : "finish_assessment"
}
########
alphabet_map_treetopcity = {
    "Ordering Spheres": "A",
    "All Star Sorting": "B",
    "Costume Box": "C", 
    "Fireworks (Activity)": "D",
    "12 Monkeys": "E",
    "Flower Waterer (Activity)": "F",
    "Pirate's Tale": "G",
    "Mushroom Sorter (Assessment)": "H",
    "Air Show": "I",
    "Treasure Map": "J",
    "Crystals Rule": "K",
    "Rulers": "L",
    "Bug Measurer (Activity)": "M",
    "Bird Measurer (Assessment)": "N"
}

alphabet_map_magmapeak = {
    "Sandcastle Builder (Activity)": "A",
    "Slop Problem": "B",
    "Scrub-A-Dub": "C",
    "Watering Hole (Activity)": "D",
    "Dino Drink": "E",
    "Bubble Bath": "F",
    "Bottle Filler (Activity)": "G",
    "Dino Dive": "H",
    "Cauldron Filler (Assessment)": "I"
}

alphabet_map_crystalcaves = {
    "Chow Time": "A",
    "Balancing Act": "B",
    "Chicken Balancer (Activity)": "C",
    "Lifting Heavy Things": "D",
    "Honey Cake": "E",
    "Happy Camel": "F",
    "Cart Balancer (Assessment)": "G",
    "Leaf Leader": "H",
    "Heavy, Heavier, Heaviest": "I",
    "Pan Balance": "J",
    "Egg Dropper (Activity)": "K",
    "Chest Sorter (Assessment)": "L"
}

alphabet_map_treetopcity = defaultdict(lambda: "Z", alphabet_map_treetopcity)
alphabet_map_magmapeak = defaultdict(lambda: "Z", alphabet_map_magmapeak)
alphabet_map_crystalcaves = defaultdict(lambda : "Z", alphabet_map_crystalcaves)

assess_titles = [
    'Mushroom Sorter (Assessment)', 
    'Bird Measurer (Assessment)',
    'Cauldron Filler (Assessment)', 
    'Cart Balancer (Assessment)',
    'Chest Sorter (Assessment)'
]

title_world_map = {
        "Mushroom Sorter (Assessment)" : "TREETOPCITY",
        "Bird Measurer (Assessment)" : "TREETOPCITY",
        "Cauldron Filler (Assessment)" : "MAGMAPEAK",
        "Cart Balancer (Assessment)" : "CRYSTALCAVES",
        "Chest Sorter (Assessment)" : "CRYSTALCAVES"
    }

title_coding = dict(zip(assess_titles, np.arange(len(assess_titles))))

ignore_path_nodes =  [
    "Welcome to Lost Lagoon!",
    "Magma Peak - Level 1", "Magma Peak - Level 2", "Magma Peak - Level 3",
    "Tree Top City - Level 1", "Tree Top City - Level 2", "Tree Top City - Level 3",
    "Crystal Caves - Level 1", "Crystal Caves - Level 2", "Crystal Caves - Level 3"
]

lgb_params = {
    "objective": "regression",
    "metric" : "None",
    "learning_rate": 0.01,
    "num_leaves": 32,
    "max_depth" : 7,
    "feature_fraction": 0.3,
    'feature_fraction_seed': 10,
    "verbosity": 0,
    "subsample": 1.,
    "num_threads" : 18,
    'min_sum_hessian_in_leaf' : 120,
}