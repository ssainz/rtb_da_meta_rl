# rtb_da_meta_rl
Real Time Bidding in Display Advertising with Meta Reinforcement Learning



### Start experiments

To start experiment to measure performance future actions (T-shoot learning): 

```bash
$python bid_performance_on_future_auctions.py
```



To start experiment to measure performance on future campaigns:

```
$python bid_performance_new_campaigns.py
```



#### Adjust which methods to experiment on

Edit either `bid_performance_on_future_auctions.py` or `bid_performance_new_campaigns.py` and search for array `agents_to_execute`, there will be a set of potential agents. 

The potential agent values are:

-  meta_imitation: Meta RL with imitation learning as initialization step.
- lin: Linear agent
- rlb_rl_dp_tabular: Reinforcement Learning for bid based on dynamic programming.
- rlb_rl_fa: Reinforcement Learning for bid, based on function approximation with imitation learning as initialization step.





