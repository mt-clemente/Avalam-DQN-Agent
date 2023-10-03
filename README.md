# Avalam-DQN-Agent

<p align="center">
  <img src=https://github.com/mt-clemente/Avalam-DQN-Agent/assets/92925667/26eab121-cd13-4dca-a5bb-d84528d9a489
 />
</p>

The Avalam game is a very simple game in concept, but has a very large branching factor which makes it the perfect opportunity to use alphabeta search. However, the state tree is so large / deep
that the algorithm still has to heavily rely on heuristics to be able to play at a correct level. This project's goal is to be able to learn to outplay any given adversary with enough training.
The learning agent uses:
 * A 3-layer CNN, with batch normalization between layers.
 * An basic adaptive depth alphabeta search to follow the maximum move time without losing too much performance.

The RL algorithm is a DDQN, with Prioritized Experience Replay. You have the possibility to use generation training, meaning that you can train each generation for $n$ episodes then
train a new generation against the previous one. Note that only one the agents is trained at the time, not both.



<!-- GETTING STARTED -->
## The game interface

 * The game agents are launched and bound to specific ports using the following commands:

  ```
  python path/to/player.py --bind localhost --port 8000
  ```

 * You can launch a game and play against an agent. For this, you can specify how much time the agent is going to have to make each move (if useful). To start playing, just run:
  
  ```
  python game.py http://localhost:8080 human --time 900
  ```

 * If you want to have two agents battle against each other, use:
  
  ```
  python game.py http://localhost:PORT1 http://localhost:PORT2 (--time SECONDS) (--verbose) (--no-gui)
  ```



## Using the agent:

Whether it is to test your agent, or do a warm start for your training, you can use the agent with a pretrained model. Simply Copy your model_name.pt model to ./models/currDQN.pt
If the file does not exist, the agent will simply start from scratch with a new random model.
To parametrize the model just change the training/testing/network shape settings in `config.py`.

 * Model training:

  ```
  python3 master_train.py
  ```

 * Model testing:
  ```
  python3 test_launcher.py
  ```


If you want to train the model and have a GPU, make sure that torch.cuda.is_available() returns true
or the agent will use your CPU exclusively.

There are some pre_trained models in the ./saved_models/ folder.

When using generation training, your models are saved in the ./session_models folder. **If you start another training session, these models will be overwritten one by one**.

You can monitor the agent's progress using tensorboard:
	$ tensorbord --logdir=runs


## Some results

As the only computing power at hand was a laptop, the model was not trained extensively. However, some intersting results still arose:
* The agent beat a random player quite easily.
* The agent was also being to beat a score greedy player, with longer training. 
* The generation training seemed to be to unstable to learn anything.
* The agent won a few games in a Challonge competition before being eliminated.

All in all, the performance of the agent is not outstanding but a quite small neural net is able to learn to outplay basic heuristic agents.
It would be interesting to try and train it against more competitive agents, and to try new configurations to see how far it can go!



