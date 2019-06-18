

class bidding_agent:


    def init(self, environment, camp_info):

        self.env = environment
        self.cpm = camp_info["cost_train"] / camp_info["imp_train"]
        self.theta_avg = camp_info["clk_train"] / camp_info["imp_train"]
        self.b0 = 0
        self.step = 6
        self.valid_rate = 1
        self.min_valid = 300000


    def run(self):

        run = 1