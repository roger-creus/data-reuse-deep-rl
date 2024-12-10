# vec-dqn

python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
pip install -r requirements.txt


### Train DQN

## 1) **DQN - Single Transtions**

In this settings, we train with TD(1) updates and single transitions, basically vanilla DQN.

We investigate different settings for data collection and training. 
- 1) scaling num_envs which implies:
- 2) scaling batch size too keep a replay ratio of 8 which implies:
- 3) scaling the learning rate by sqrt(k)

# 1 Env - TD(1)
python src/dqn_td1.py \
--env_id=Breakout-v5 \
--num_envs=1 \
--buffer_size=1000000 \
--batch_size=32 \
--learning_rate=0.0001 \
--learning_starts=80000 \
--train_frequency=4 \
--target_network_frequency=250 \
--lstm_hidden_size=-1

# 16 Env - TD(1)
python src/dqn_td1.py \
--env_id=Breakout-v5 \
--num_envs=16 \
--buffer_size=1000000 \
--batch_size=128 \
--learning_rate=0.0002 \
--learning_starts=80000 \
--train_frequency=1 \
--target_network_frequency=250 \
--lstm_hidden_size=-1

# 64 Env - TD(1)
python src/dqn_td1.py \
--env_id=Breakout-v5 \
--num_envs=64 \
--buffer_size=1000000 \
--batch_size=512 \
--learning_rate=0.0004 \
--learning_starts=80000 \
--train_frequency=1 \
--target_network_frequency=250 \
--lstm_hidden_size=-1

# 128 Env - TD(1)
python src/dqn_td1.py \
--env_id=Breakout-v5 \
--num_envs=128 \
--buffer_size=1000000 \
--batch_size=1024 \
--learning_rate=0.0008 \
--learning_starts=80000 \
--train_frequency=1 \
--target_network_frequency=250 \
--lstm_hidden_size=-1

## 2) **DQN - Rollouts**.

In this settings, we investigate training either with n-steps or with q-lambda for estimating the values.


# 1 Env - Q-Lambda
python src/dqn_rollout.py \
--env_id=Breakout-v5 \
--num_envs=1 \
--buffer_size=1000000 \
--batch_size=1 \
--learning_rate=0.0001 \
--learning_starts=80000 \
--train_frequency=4 \
--target_network_frequency=250 \
--lstm_hidden_size=-1 \
--use_q_lambda

# 16 Env - Q-Lambda
python src/dqn_rollout.py \
--env_id=Breakout-v5 \
--num_envs=16 \
--buffer_size=1000000 \
--batch_size=4 \
--learning_rate=0.0002 \
--learning_starts=80000 \
--train_frequency=1 \
--target_network_frequency=250 \
--lstm_hidden_size=-1 \
--use_q_lambda

# 64 Env - Q-Lambda
python src/dqn_rollout.py \
--env_id=Breakout-v5 \
--num_envs=64 \
--buffer_size=1000000 \
--batch_size=16 \
--learning_rate=0.0004 \
--learning_starts=80000 \
--train_frequency=1 \
--target_network_frequency=250 \
--lstm_hidden_size=-1 \
--use_q_lambda

# 128 Env - Q-Lambda
python src/dqn_rollout.py \
--env_id=Breakout-v5 \
--num_envs=128 \
--buffer_size=1000000 \
--batch_size=64 \
--learning_rate=0.0008 \
--learning_starts=80000 \
--train_frequency=1 \
--target_network_frequency=250 \
--lstm_hidden_size=-1 \
--use_q_lambda