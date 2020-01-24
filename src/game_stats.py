'''
 Created on Fri Dec 27 2019
 __author__: bishwarup
'''

from __future__ import division, print_function
import os
import re
import gc
import json
import joblib
import warnings
from ast import literal_eval
import itertools
import time
import shutil
from collections import Counter
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None 

def get_count_animals(x):
    try:
        return len(x)
    except:
        return 0

def get_avg_run_length(s):
    groups = [list(g) for _, g in itertools.groupby(s)]
    lens = [len(g) for g in groups]
    return np.mean(lens)

def events_to_correct(x):
    P = [list(g) for _, g in itertools.groupby(x)]
    return len(P[0])

def get_max_run(x):
    P = [list(g) for _, g in itertools.groupby(x)]
    lens = [len(p) for p in P]
    return np.max(lens)

def Chow_Time(df):
	all_df = df.query('title == "Chow Time"')[["game_session"]].drop_duplicates()
	chow_time = df.query('title == "Chow Time" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(chow_time.event_data.apply(json.loads))[["round", "misses"]]
	chow_time = pd.concat([chow_time, ld], axis = 1)[["game_session", "round", "misses"]]
	chow_time["acc"] = np.exp(-chow_time["misses"]/ 3.)
	chow_time["round"][chow_time["round"] % 3 != 0] = chow_time["round"] % 3
	chow_time["round"][chow_time["round"] % 3 == 0] = 3
	chow_time["max_r"] = chow_time.groupby("game_session")["round"].transform("max")
	chow_time = chow_time.groupby(["game_session", "round"]).agg({
	    "acc" : "mean",
	    "max_r" : "last"
	}).reset_index()
	chow_time_scores = chow_time.groupby("game_session").agg({
    "acc" : "mean",
    "max_r": "last"
	}).reset_index()
	chow_time_scores["acc"] = chow_time_scores["acc"] - (1 - np.exp(-(3 - chow_time_scores["max_r"])**2/10))
	chow_time_scores.drop("max_r", axis = 1, inplace=True)
	all_df = all_df.merge(chow_time_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def All_Star_Sorting(df):
	all_df = df.query('title == "All Star Sorting"')[["game_session"]].drop_duplicates()
	all_star_sorting = df.query('title == "All Star Sorting" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(all_star_sorting.event_data.apply(json.loads))[["round", "misses"]]
	all_star_sorting = pd.concat([all_star_sorting, ld], axis = 1)[["game_session", "round", "misses"]]
	all_star_sorting["weights"] = .23
	all_star_sorting["weights"][(all_star_sorting["round"] % 3) == 0] = .45
	all_star_sorting["weights"][(all_star_sorting["round"] % 3) == 2] = .32

	all_star_sorting["round"][all_star_sorting["round"] % 3 != 0] = all_star_sorting["round"] % 3
	all_star_sorting["round"][all_star_sorting["round"] % 3 == 0] = 3

	all_star_sorting["penalty"] = 1/2
	all_star_sorting["penalty"][all_star_sorting["round"] == 2] = 1/5
	all_star_sorting["penalty"][all_star_sorting["round"] == 3] = 1/10

	all_star_sorting["acc"] = all_star_sorting["weights"] * np.exp(-all_star_sorting["misses"] * all_star_sorting["penalty"])
	all_star_sorting = all_star_sorting.groupby(["game_session", "round"])["acc"].mean().reset_index()
	all_star_sorting_scores = all_star_sorting.groupby("game_session")["acc"].sum().reset_index()
	all_df = all_df.merge(all_star_sorting_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Dino_Dive(df):
	all_df = df.query('title == "Dino Dive"')[["game_session"]].drop_duplicates()
	dino_dive = df.query('title == "Dino Dive" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(dino_dive.event_data.apply(json.loads))[["round", "misses"]]
	dino_dive = pd.concat([dino_dive, ld], axis = 1)[["game_session", "round", "misses"]]

	dino_dive["weights"] = .18
	dino_dive["weights"][(dino_dive["round"] % 4) == 0] = .32
	dino_dive["weights"][(dino_dive["round"] % 4) == 1] = .18
	dino_dive["weights"][(dino_dive["round"] % 4) == 3] = .32

	dino_dive["round"][dino_dive["round"] % 4 != 0] = dino_dive["round"] % 4
	dino_dive["round"][dino_dive["round"] % 4 == 0] = 4

	dino_dive["penalty"] = 1/4
	dino_dive["penalty"][dino_dive["round"] == 3] = 1/10
	dino_dive["penalty"][dino_dive["round"] == 4] = 1/10


	dino_dive["acc"] = dino_dive["weights"] * np.exp(-dino_dive["misses"] * dino_dive["penalty"])
	dino_dive = dino_dive.groupby(["game_session", "round"])["acc"].mean().reset_index()
	dino_dive_scores = dino_dive.groupby("game_session")["acc"].sum().reset_index()

	all_df = all_df.merge(dino_dive_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Crystals_Rule(df):
	all_df = df.query('title == "Crystals Rule"')[["game_session"]].drop_duplicates()
	crystals_rule = df.query('title == "Crystals Rule" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(crystals_rule.event_data.apply(json.loads))[["round", "misses"]]
	crystals_rule = pd.concat([crystals_rule, ld], axis = 1)[["game_session", "round", "misses"]]

	crystals_rule["acc"] = .5**crystals_rule["misses"]
	crystals_rule_scores = crystals_rule.groupby("game_session").agg({
	    "acc": "mean",
	    "round": "max"
	}).reset_index()

	crystals_rule_scores["acc"] = crystals_rule_scores["acc"] - (1 - np.exp(-(9 - crystals_rule_scores["round"])**2/100))
	crystals_rule_scores.drop("round", axis = 1, inplace=True)

	all_df = all_df.merge(crystals_rule_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Air_Show(df):
	all_df = df.query('title == "Air Show"')[["game_session"]].drop_duplicates()
	air_show = df.query('title == "Air Show" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(air_show.event_data.apply(json.loads))[["round", "misses"]]
	air_show = pd.concat([air_show, ld], axis = 1)[["game_session", "round", "misses"]]
	
	air_show["acc"] = np.exp(-air_show["misses"]/ 5.)
	air_show_scores = air_show.groupby("game_session").agg({
	    "acc": "mean",
	    "round": "max"
	}).reset_index()
	air_show_scores["acc"] = air_show_scores["acc"] - (1 - np.exp(-(3 - air_show_scores["round"])**2/100))
	air_show_scores.drop("round", axis = 1, inplace=True)

	all_df = all_df.merge(air_show_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Happy_Camel(df):
	all_df = df.query('title == "Happy Camel"')[["game_session"]].drop_duplicates()
	happy_camel = df.query('title == "Happy Camel" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(happy_camel.event_data.apply(json.loads))[["round", "misses"]]
	happy_camel = pd.concat([happy_camel, ld], axis = 1)[["game_session", "round", "misses"]]

	happy_camel["weights"] = .2
	happy_camel["weights"][(happy_camel["round"] % 3) == 0] = .5
	happy_camel["weights"][(happy_camel["round"] % 3) == 2] = .3

	happy_camel["round"][happy_camel["round"] % 3 != 0] = happy_camel["round"] % 3
	happy_camel["round"][happy_camel["round"] % 3 == 0] = 3

	happy_camel["penalty"] = 1/2
	happy_camel["penalty"][happy_camel["round"] == 2] = 1/5
	happy_camel["penalty"][happy_camel["round"] == 3] = 1/10


	happy_camel["acc"] = happy_camel["weights"] * np.exp(-happy_camel["misses"] * happy_camel["penalty"])
	happy_camel = happy_camel.groupby(["game_session", "round"])["acc"].mean().reset_index()
	happy_camel_scores = happy_camel.groupby("game_session")["acc"].sum().reset_index()

	all_df = all_df.merge(happy_camel_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Leaf_Leader(df):
	all_df = df.query('title == "Leaf Leader"')[["game_session"]].drop_duplicates()
	leaf_leader = df.query('title == "Leaf Leader" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(leaf_leader.event_data.apply(json.loads))[["round", "misses"]]
	leaf_leader = pd.concat([leaf_leader, ld], axis = 1)[["game_session", "round", "misses"]]

	leaf_leader["acc"] = np.exp(-leaf_leader["misses"]/ 5.)
	leaf_leader["round"][leaf_leader["round"] % 3 != 0] = leaf_leader["round"] % 3
	leaf_leader["round"][leaf_leader["round"] % 3 == 0] = 3
	leaf_leader["max_r"] = leaf_leader.groupby("game_session")["round"].transform("max")
	leaf_leader = leaf_leader.groupby(["game_session", "round"]).agg({
	    "acc" : "mean",
	    "max_r" : "last"
	}).reset_index()

	leaf_leader_scores = leaf_leader.groupby("game_session").agg({
	    "acc" : "mean",
	    "max_r": "last"
	}).reset_index()
	leaf_leader_scores["acc"] = leaf_leader_scores["acc"] - (1 - np.exp(-(3 - leaf_leader_scores["max_r"])**2/10))
	leaf_leader_scores.drop("max_r", axis = 1, inplace=True)

	all_df = all_df.merge(leaf_leader_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Pan_Balance(df):
	all_df = df.query('title == "Pan Balance"')[["game_session"]].drop_duplicates()
	pan_balance = df.query('title == "Pan Balance" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(pan_balance.event_data.apply(json.loads))[["round", "misses"]]
	pan_balance = pd.concat([pan_balance, ld], axis = 1)[["game_session", "round", "misses"]]

	pan_balance_scores = pan_balance.groupby("game_session").agg({
	    "round": "max",
	    "misses" : "sum"
	}).reset_index()
	pan_balance_scores["acc"] = np.exp(-pan_balance_scores["misses"] / 5.) * (1 - np.exp(-pan_balance_scores["round"]/2.))
	pan_balance_scores.drop(["misses", "round"], axis = 1, inplace=True)

	all_df = all_df.merge(pan_balance_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Scrub_A_Dub(df):
	all_df = df.query('title == "Scrub-A-Dub"')[["game_session"]].drop_duplicates()
	srcd = df.query('title == "Scrub-A-Dub" & (event_code == 2020  | event_code == 2030)').reset_index()
	ld = pd.io.json.json_normalize(srcd.event_data.apply(json.loads))[["misses", "animals", "round"]]
	srcd = pd.concat([srcd, ld], axis = 1)[["game_session", "misses", "animals", "event_code", "round"]]

	srcd["animals"] = srcd["animals"].map(get_count_animals)
	srcd = srcd.fillna(0).drop("event_code", axis = 1).groupby(["game_session", "round"], as_index = False).agg('sum')
	srcd = srcd.groupby(["game_session", "animals"], as_index = False)["misses"].agg("sum")
	srcd["weights"] = 0.0
	srcd["weights"][srcd["animals"] == 2] = 0.1
	srcd["weights"][srcd["animals"] == 3] = 0.2
	srcd["weights"][srcd["animals"] == 4] = 0.3
	srcd["weights"][srcd["animals"] == 5] = 0.4

	srcd["penalty"] = 1/2
	srcd["penalty"][srcd["animals"] == 3] = 1/5
	srcd["penalty"][srcd["animals"] == 4] = 1/8
	srcd["penalty"][srcd["animals"] == 5] = 1/10

	srcd["acc"] = srcd["weights"] * np.exp(-srcd["misses"] * srcd["penalty"])
	srcd_scores = srcd.groupby("game_session", as_index = False)["acc"].agg("sum")

	all_df = all_df.merge(srcd_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Bubble_Bath(df):
	all_df = df.query('title == "Bubble Bath"')[["game_session"]].drop_duplicates()
	bubble_bath = df.query('title == "Bubble Bath" & event_code == 4020').reset_index()
	ld = pd.io.json.json_normalize(bubble_bath.event_data.apply(json.loads))[["correct", "containers", "target_containers","round"]]
	bubble_bath = pd.concat([bubble_bath, ld], axis = 1)[["game_session", "correct", "containers", "target_containers","round", "event_code"]]

	bubble_bath["correct"] = bubble_bath["correct"].astype(np.int8).replace(0, -1)
	bubble_bath["acc"] = bubble_bath["correct"] * (1. / (1. + np.exp(-bubble_bath["target_containers"]*bubble_bath["correct"] / 4)))
	bubble_bath_scores = bubble_bath.groupby("game_session", as_index = False)["acc"].mean()

	all_df = all_df.merge(bubble_bath_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Dino_Drink(df):
	all_df = df.query('title == "Dino Drink"')[["game_session"]].drop_duplicates()
	dino_drink = df.query('title == "Dino Drink" & (event_code == 2020  | event_code == 2030)').reset_index()
	ld = pd.io.json.json_normalize(dino_drink.event_data.apply(json.loads))[["misses", "holes", "round"]]
	dino_drink = pd.concat([dino_drink, ld], axis = 1)[["game_session", "misses", "holes", "event_code", "round"]]

	dino_drink["holes"] = dino_drink["holes"].map(get_count_animals)
	dino_drink = dino_drink.fillna(0).drop("event_code", axis = 1)\
                    .groupby(["game_session", "round"], as_index = False).agg('sum')
	dino_drink["weights"] = 0.2
	dino_drink["weights"][dino_drink["holes"] == 3] = 0.3
	dino_drink["weights"][dino_drink["holes"] == 5] = 0.5

	dino_drink["penalty"] = 1/3
	dino_drink["penalty"][dino_drink["holes"] == 3] = 1/5
	dino_drink["penalty"][dino_drink["holes"] == 5] = 1/10

	dino_drink["acc"] = dino_drink["weights"] * np.exp(-dino_drink["misses"] * dino_drink["penalty"])
	dino_drink = dino_drink.groupby(['game_session', 'holes'], as_index = False)['acc'].agg("mean")
	dino_drink_scores = dino_drink.groupby("game_session", as_index= False)['acc'].agg("sum")

	all_df = all_df.merge(dino_drink_scores, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Chicken_Balancer(df):
	all_df = df.query('title == "Chicken Balancer (Activity)"')[["game_session"]].drop_duplicates()
	chicken_balancer = df.query('title == "Chicken Balancer (Activity)" & event_code == 4020').reset_index()
	ld = pd.io.json.json_normalize(chicken_balancer.event_data.apply(json.loads))\
	    [["layout.left.chickens", "layout.right.chickens", "layout.left.pig", "layout.right.pig"]]
	ld.columns = [x.replace(".", "_") for x in ld.columns]
	chicken_balancer = pd.concat([chicken_balancer, ld], axis = 1)\
	    [["game_session", "event_count", "layout_left_chickens", "layout_right_chickens", "layout_left_pig", "layout_right_pig"]]
	chicken_balancer["layout_left_pig"] = chicken_balancer["layout_left_pig"].astype(np.uint8).replace(1, 6)
	chicken_balancer["layout_right_pig"] = chicken_balancer["layout_right_pig"].astype(np.uint8).replace(1, 6)

	chicken_balancer["balanced"] = chicken_balancer.apply(lambda row:
	    (row["layout_left_chickens"] + row["layout_left_pig"]) == (row["layout_right_chickens"] + row["layout_right_pig"]),
	                                                    axis = 1 )

	chicken_balancer = chicken_balancer.groupby("game_session", as_index = False).agg({
	    "event_count": "max",
	    "balanced" : "sum"
	})
	chicken_balancer["cb_balance_ratio"] = chicken_balancer["balanced"] / chicken_balancer["event_count"]
	chicken_balancer.drop(["event_count", "balanced"], axis = 1, inplace=True)

	all_df = all_df.merge(chicken_balancer, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Egg_Dropper(df):
	all_df = df.query('title == "Egg Dropper (Activity)"')[["game_session"]].drop_duplicates()
	egg_dropper = df.query('title == "Egg Dropper (Activity)" & event_code == 4025').reset_index()
	ld = pd.io.json.json_normalize(egg_dropper.event_data.apply(json.loads))[["nest"]]
	egg_dropper = pd.concat([egg_dropper, ld], axis = 1)[["game_session", "event_count", "nest"]]

	egg_dropper = egg_dropper.groupby("game_session", as_index = False).agg({
	    "event_count": "size",
	    "nest": get_avg_run_length
	})

	egg_dropper["ed_activity_level"] = (1 - egg_dropper["nest"] / egg_dropper["event_count"]) * \
	                                    (1 - np.exp(-egg_dropper["event_count"]/2.))
	egg_dropper.drop(["event_count", "nest"], axis = 1, inplace=True)

	all_df = all_df.merge(egg_dropper, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Bug_Measurer(df):
	all_df = df.query('title == "Bug Measurer (Activity)"')[["game_session"]].drop_duplicates()
	bug_measurer = df.query('title == "Bug Measurer (Activity)" & event_code == 4025').reset_index()
	ld = pd.io.json.json_normalize(bug_measurer.event_data.apply(json.loads))[["buglength", "bug"]]
	bug_measurer = pd.concat([bug_measurer, ld], axis = 1)[["game_session", "event_count", "buglength", "bug"]]

	bug_measurer = bug_measurer.groupby(["game_session", "bug"])["event_count"].agg("size").reset_index()

	bug_measurer["bm_activity_level"] = (1 - np.exp(-2. * bug_measurer["event_count"])) * (1 / 7.)
	bug_measurer = bug_measurer.groupby("game_session", as_index = False)["bm_activity_level"].agg("sum")

	all_df = all_df.merge(bug_measurer, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Fireworks(df):
	all_df = df.query('title == "Fireworks (Activity)"')[["game_session"]].drop_duplicates()
	fireworks = df.query('title == "Fireworks (Activity)" & event_code == 4020').reset_index()
	ld = pd.io.json.json_normalize(fireworks.event_data.apply(json.loads))[["rocket", "height"]]
	fireworks = pd.concat([fireworks, ld], axis = 1)[["game_session", "event_count", "rocket", "height"]]
	fireworks["height"] = fireworks["height"]/ 500.

	fireworks = fireworks.groupby("game_session", as_index = False).agg({
	    "rocket": pd.Series.nunique,
	    "height": np.std
	})
	fireworks["rocket"] = fireworks["rocket"] / 6.

	fireworks["fw_activity_level"] = fireworks["rocket"] * fireworks["height"]
	fireworks.drop(["rocket", "height"], axis = 1, inplace=True)

	all_df = all_df.merge(fireworks, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Flower_Waterer(df):
	all_df = df.query('title == "Flower Waterer (Activity)"')[["game_session"]].drop_duplicates()
	flower_waterer = df.query('title == "Flower Waterer (Activity)" & (event_code == 4020 | event_code == 4030)')\
                                                                .reset_index()
	ld = pd.io.json.json_normalize(flower_waterer.event_data.apply(json.loads))[["growth", "flower"]]
	flower_waterer = pd.concat([flower_waterer, ld], axis = 1)[["game_session", "event_count", 
	                                                            "growth", "flower", "event_code"]]

	flower_waterer = flower_waterer.groupby("game_session", as_index = False).agg({
	    "growth": lambda x: pd.notnull(x).sum(),
	    "flower": lambda x: pd.notnull(x).sum(),
	    "event_count": "max"
	})

	flower_waterer["flw_activity_level"] = (flower_waterer["growth"] + 
	                                        flower_waterer["flower"]) / flower_waterer["event_count"]
	flower_waterer.drop(["growth", "flower", "event_count"], axis = 1, inplace= True)

	all_df = all_df.merge(flower_waterer, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Sandcastle_Builder(df):
	all_df = df.query('title == "Sandcastle Builder (Activity)"')[["game_session"]].drop_duplicates()
	sandcastle = df.query('title == "Sandcastle Builder (Activity)" & \
                                        (event_code == 4020 | event_code == 4021)')\
                                                                .reset_index()
	ld = pd.io.json.json_normalize(sandcastle.event_data.apply(json.loads))[["filled", "size"]]
	sandcastle = pd.concat([sandcastle, ld], axis = 1)[["game_session", "event_count", 
	                                                            "filled", "size", "event_code"]]
	sandcastle = sandcastle.groupby("game_session", as_index = False).agg({
	    "filled": lambda x: (x == True).sum(),
	    "event_code" : lambda x: (x == 4020).sum(),
	    "event_count": "max",
	})
	sandcastle["sc_activity_level"] = (sandcastle["filled"] + sandcastle["event_code"]) / sandcastle["event_count"]
	sandcastle.drop(["filled", "event_code", "event_count"], axis = 1, inplace=True)

	all_df = all_df.merge(sandcastle, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Bottle_Filler(df):
	all_df = df.query('title == "Bottle Filler (Activity)"')[["game_session"]].drop_duplicates()
	bottle_filler = df.query('title == "Bottle Filler (Activity)" & event_code == 2030').reset_index()
	ld = pd.io.json.json_normalize(bottle_filler.event_data.apply(json.loads))[["jar", "round"]]
	bottle_filler = pd.concat([bottle_filler, ld], axis = 1)[["game_session", "event_count", "jar", "round"]]

	bottle_filler["jar"] = bottle_filler["jar"].map(len)

	bottle_filler = bottle_filler.groupby("game_session", as_index = False).agg({
	    "jar": "sum",
	    "round" : "max",
	})
	bottle_filler.rename(columns={"jar": "bf_n_jars", "round": "bf_max_round"}, inplace=True)

	all_df = all_df.merge(bottle_filler, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Watering_Hole(df):
	all_df = df.query('title == "Watering Hole (Activity)"')[["game_session"]].drop_duplicates()
	water_hole = df.query('title == "Watering Hole (Activity)" & event_code == 4021').reset_index()
	ld = pd.io.json.json_normalize(water_hole.event_data.apply(json.loads))[["cloud"]]
	water_hole = pd.concat([water_hole, ld], axis = 1)[["game_session", "event_count", "cloud"]]

	def get_runs(s):
	    groups = [k for k, _ in itertools.groupby(s)]
	    return len(groups)
	water_hole = water_hole.groupby("game_session", as_index =False).agg({
	    "cloud": lambda x: get_runs(x),
	    "event_count" : "max"
	})
	water_hole["wh_activity_level"] = water_hole["cloud"] / water_hole["event_count"]
	water_hole.drop(["cloud", "event_count"], axis = 1, inplace=True)

	all_df = all_df.merge(water_hole, on = "game_session", how = "left")
	all_df.fillna(0., inplace = True)
	return all_df

def Mushroom_Sorter(df, fillna = False):
    all_df = df.query('title == "Mushroom Sorter (Assessment)"')[["game_session"]].drop_duplicates()
    mushroom = df.query('title == "Mushroom Sorter (Assessment)" & event_code == 4020').reset_index()
    ld = pd.io.json.json_normalize(mushroom.event_data.apply(json.loads))[["stumps", "correct"]]
    mushroom = pd.concat([mushroom, ld], axis = 1)[["game_session", "event_count", "game_time", "stumps", "correct"]]

    mushroom["all_correct"] = mushroom.apply(lambda row: np.logical_and(np.min(row['stumps']) > 0, row['correct']), axis = 1)
    mushroom['tt_correct'] = mushroom.groupby(['game_session', 'all_correct'])['game_time'].transform('max')

    mushroom["tr_to_correct"] = mushroom.groupby('game_session')['all_correct'].transform(lambda x: events_to_correct(x))
    mushroom["got_coorect"] = mushroom.groupby('game_session')['all_correct'].transform(lambda x: events_to_correct(x))
    mushroom["idx"] = mushroom.groupby("game_session").cumcount()

    mushroom["tt_1_correct"] = -1
    mushroom["tt_1_correct"][mushroom['tr_to_correct'] == mushroom['idx']] = mushroom['game_time']
    mushroom["tt_1_correct"] = mushroom.groupby('game_session')['tt_1_correct'].transform('max')
    mushroom["solved"] = mushroom.groupby("game_session")["all_correct"].transform('last')

    mushroom = mushroom.groupby("game_session", as_index = False).agg({
        "idx": "max",
        "tt_correct": "max",
        "tt_1_correct": "first",
        "solved": "first"
    })

    mushroom["idx"][~mushroom["solved"]] = -1 * mushroom["idx"]
    mushroom["tt_1_correct"][~mushroom["solved"]] = mushroom["tt_correct"]
    mushroom["conf"] = np.exp(-(mushroom["idx"] - 2) / 5.) * np.exp(-(mushroom["tt_correct"] - 10) / 100.) * \
                np.exp(-(mushroom["tt_correct"] - mushroom["tt_1_correct"]) / 30.) * mushroom["solved"].astype(np.int8).replace(0, -1)
    mushroom.drop(["idx", "tt_correct", "tt_1_correct", "solved"], axis = 1, inplace = True)

    all_df = all_df.merge(mushroom, on = "game_session", how = "left")
    if fillna:
        all_df.fillna(-99., inplace = True)
    return all_df

def Cauldron_Filler(df, fillna = False):
    all_df = df.query('title == "Cauldron Filler (Assessment)"')[["game_session"]].drop_duplicates()
    cauldron = df.query('title == "Cauldron Filler (Assessment)" & (event_code == 4020 | event_code == 4100)').reset_index()
    ld = pd.io.json.json_normalize(cauldron.event_data.apply(json.loads))[["buckets_placed", "correct"]]
    cauldron = pd.concat([cauldron, ld], axis = 1)[["game_session", "event_code", "event_count", "game_time", "buckets_placed", "correct"]]

    cauldron["idx"] = cauldron.query('event_code != 4100').groupby("game_session")["event_code"].cumcount()
    cauldron["idx"] = cauldron.groupby("game_session")["idx"].transform("max")
    cauldron["solved"] = np.logical_and(cauldron["event_code"] == 4100, cauldron["correct"])
    cauldron["solved"] = cauldron["solved"].astype(np.int8)
    cauldron["solved"] = cauldron.groupby("game_session")["solved"].transform('max')
    cauldron['rl'] = cauldron.query('event_code == 4100').groupby('game_session')['buckets_placed'].transform(lambda x: get_max_run(x))
    cauldron['rl'] = cauldron.groupby('game_session')['rl'].transform('max')
    cauldron.rl.fillna(1, inplace=True)

    cauldron = cauldron.groupby('game_session', as_index = False).agg({
        "game_time": "max",
        "idx" : "max",
        "rl": "max",
        "solved": "max"
    })

    cauldron["conf"] = np.exp(-(cauldron["idx"] - 2) / 5.) * np.exp(-(cauldron["game_time"] - 4) / 100.) * \
                np.exp(-(cauldron['rl'] - 1) / 3.) *  cauldron["solved"].astype(np.int8).replace(0, -1)
    cauldron.drop(["game_time", "idx", "rl", "solved"], axis = 1, inplace = True)

    all_df = all_df.merge(cauldron, on = "game_session", how = "left")
    if fillna:
        all_df.fillna(-99., inplace = True)
    return all_df

def Cart_Balancer(df, fillna = False):
    all_df = df.query('title == "Cart Balancer (Assessment)"')[["game_session"]].drop_duplicates()
    cart = df.query('title == "Cart Balancer (Assessment)" & (event_code == 4020 | event_code == 4100)').reset_index()
    ld = pd.io.json.json_normalize(cart.event_data.apply(json.loads))[["left", "right", "correct"]]
    cart = pd.concat([cart, ld], axis = 1)[["game_session", "event_code", "event_count", "game_time", "left", "right", "correct"]]
    cart["correct"].fillna(False, inplace= True)
    cart["correct"] = cart["correct"].astype(np.uint8)
    cart["n_tr"] = cart.groupby("game_session")["event_code"].transform(lambda x: (x == 4100).sum())

    cart["left_n_item"] = cart["left"].map(len)
    cart["right_n_item"] = cart["right"].map(len)

    cart = cart.groupby("game_session", as_index = False).agg({
        "left_n_item" : "max",
        "right_n_item" : "max",
        "correct" : "max",
        "game_time": "max",
        "n_tr": "max",
    })

    cart["total_items"] = cart["right_n_item"] + cart["left_n_item"]
    cart["max_item"] = cart.apply(lambda row: max(row["right_n_item"], row["left_n_item"]), axis = 1)
    cart["conf"] = ((1. - np.exp(-cart["max_item"]**2)) * \
                        np.exp(-(cart["game_time"] / cart["total_items"] - 1)/50.) + \
                        np.exp(np.abs(cart["left_n_item"] - cart["right_n_item"]))/10. - 0.1) / cart["n_tr"]
    cart["conf"][cart["correct"] == 0] = -1
    cart.drop(["left_n_item", "right_n_item", "correct", "game_time", "n_tr", "total_items", "max_item"], axis = 1, inplace=True)

    all_df = all_df.merge(cart, on = "game_session", how = "left")
    if fillna:
        all_df.fillna(-99., inplace = True)
    return all_df

def Chest_Sorter(df, fillna = False):
    all_df = df.query('title == "Chest Sorter (Assessment)"')[["game_session"]].drop_duplicates()
    chest = df.query('title == "Chest Sorter (Assessment)" & (event_code == 4020 | event_code == 4100)').reset_index()
    ld = pd.io.json.json_normalize(chest.event_data.apply(json.loads))[["weight", "destination", "pillars", "correct"]]
    chest = pd.concat([chest, ld], axis = 1)[["game_session", "event_code", "event_count", "game_time", "weight", "destination", "pillars", "correct"]]

    chest['pillars_str'] = chest['pillars'].astype(str)
    chest["n_same_mistake"] = chest.query('event_code == 4100').groupby(['game_session', 'pillars_str'])['event_count'].transform('size')
    chest["n_same_mistake"] = chest["n_same_mistake"] - 1
    chest["tt_to_correct"] = chest.query('event_code == 4100').groupby('game_session')['game_time'].transform('max')
    chest["tt_to_correct"].fillna(-1, inplace= True)
    chest["solved"] = 0
    chest["solved"][np.logical_and(chest['correct'], chest['event_code'] == 4100)] = 1
    chest["idx"] = chest.groupby("game_session")["event_count"].transform("size") - 1

    tmp_df = chest.groupby(["game_session", "pillars_str"]).agg({
        "idx": "first",
        "tt_to_correct": "max",
        "solved": "max",
        "n_same_mistake": "first"
    }).reset_index()

    chest = tmp_df.groupby("game_session", as_index = False).agg({
        "idx": "first",
        "solved": "max",
        "tt_to_correct": "max",
        "n_same_mistake": "sum"
    })

    chest["conf"] = np.exp(-(chest['idx'] - 3) / 20.) \
                * np.exp(-(chest['tt_to_correct'] - 8) / 100.) \
                * np.exp(-chest['n_same_mistake'] / 2.) + chest["solved"]

    chest.drop(["idx", "solved", "tt_to_correct", "n_same_mistake"], axis = 1, inplace = True)

    all_df = all_df.merge(chest, on = "game_session", how = "left")
    if fillna:
        all_df.fillna(-99., inplace = True)
    return all_df

def Bird_Measurer(df, fillna = False):
    all_df = df.query('title == "Bird Measurer (Assessment)"')[["game_session"]].drop_duplicates()
    bird = df.query('title == "Bird Measurer (Assessment)" & (event_code == 4025 | event_code == 4110)').reset_index()
    ld = pd.io.json.json_normalize(bird.event_data.apply(json.loads))[["height", "bird_height", "correct"]]
    bird = pd.concat([bird, ld], axis = 1)[["game_session", "event_code", "event_count", "game_time", "height", "bird_height", "correct"]]
    bird["error"] = np.abs(bird["height"] - bird["bird_height"])
    bird["total_submit"] = bird.groupby("game_session")["event_code"].transform(lambda x: (x == 4110).sum())
    bird['solved'] = 0
    bird['solved'][np.logical_and(bird['correct'], bird['event_code'] == 4110)] = 1
    bird = bird.groupby('game_session', as_index = False).agg({
        'error': 'mean',
        'total_submit': 'first',
        'solved': 'max',
        'game_time': 'max'
    })
    bird["conf"] = np.exp(-(bird["game_time"] - 6) / 50.) * \
            np.exp(-(bird["total_submit"] - 1) / 50.) * \
            np.exp(-bird['error'] / 3.) + bird['solved']
    bird["conf"][pd.isnull(bird["conf"])] = -1 * np.exp(bird["total_submit"] / 3.)
    bird.drop(["error", "total_submit", "solved", "game_time"], axis = 1, inplace=True)

    all_df = all_df.merge(bird, on = "game_session", how = "left")
    if fillna:
        all_df.fillna(-999., inplace = True)
    return all_df

