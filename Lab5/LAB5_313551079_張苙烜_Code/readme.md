 # test task1
❯ python test1_model.py \
  --model-path ./LAB5_313551079_task1_cartpole.pt \
  --output-dir ./cartpole_videos \
  --episodes 20

# test task2 & task3
❯ python test_model.py --model-path ./LAB5_313551079_task2_pong.pt --episodes 20

# train task1
❯ python dqn.py \                                                                          --wandb-run-name "task1" --save-dir "./task1" 

# train task2 % task3
❯ python dqn2.py --env ALE/Pong-v5 --wandb-run-name "task2" --save-dir "./task2"          


