import pickle

dataPath = "data/"

projectPath = "project/"

ipinyouPath = dataPath + "ipinyou-data/"

ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
#ipinyou_camps = ["1458", "2259"]

ipinyou_max_market_price = 300

info_keys = ["imp_test", "cost_test", "clk_test", "imp_train", "cost_train", "clk_train", "field", "dim",
             "price_counter_train"]

delimiter = " "

def get_camp_info(camp, src="ipinyou"):
    if src == "ipinyou":
        info = pickle.load(open(ipinyouPath + camp + "/info.txt", "rb"))
    return info


