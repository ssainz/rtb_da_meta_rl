import time

# obj_type: clk, profit, imp
class Opt_Obj:
    def __init__(self, obj_type="clk", clk_v=500):
        self.obj_type = obj_type
        self.clk_v = clk_v
        if obj_type == "clk":
            self.v1 = 1
            self.v0 = 0
            self.w = 0
        elif obj_type == "profit":
            self.v1 = clk_v
            self.v0 = 1
            self.w = 0
        else:
            self.v1 = 0
            self.v0 = 0
            self.w = 1

    def get_obj(self, imp, clk, cost):
        return self.v1 * clk - self.v0 * cost + self.w * imp


def getTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))