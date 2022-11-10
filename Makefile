clean: clean_games	clean_stdout clean_figs

clean_figs:
	rm -rf ./figs/scores/*

clean_games:
	rm -rf ./logs/games/*

clean_stdout:
	rm -rf ./logs/stdout/*

clean_ssh:
	rm -rf ssh/vm1/* !.gitkeep
	rm -rf ssh/vm2/* !.gitkeep

vm1:
	gcloud compute ssh --project avalam-dqn-367921 --zone us-west1-b deeplearning-1-vm -- -L 8080:localhost:8080                                                                                                                            

vm2:
	gcloud compute ssh --project avalam-dqn-367921 --zone us-west2-a deeplearning-2-vm -- -L 8080:localhost:8080

dl1:
	gcloud compute scp --project avalam-dqn-367921 --zone us-west1-b --recurse deeplearning-1-vm:~/Avalam-DQN-Agent/$d ./ssh/vm1

dl2:
	gcloud compute scp --project avalam-dqn-367921 --zone us-west2-a --recurse deeplearning-1-vm:~/Avalam-DQN-Agent/$d ./ssh/vm2

push_model1:
	gcloud compute scp --project avalam-dqn-367921 --zone us-west1-b --recurse $(m) deeplearning-1-vm:~/Avala-DQN-Agent/session_models/model_0.pt

push_model2:
	gcloud compute scp --project avalam-dqn-367921 --zone us-west2-a --recurse $(m) deeplearning-2-vm:~/Avala-DQN-Agent/session_models/model_0.pt