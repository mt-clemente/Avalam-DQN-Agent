clean: clean_games	clean_stdout clean_figs

clean_figs:
	rm -rf ./figs/scores/*

clean_games:
	rm -rf ./logs/games/*

clean_stdout:
	rm -rf ./logs/stdout/*

vm1:
	gcloud compute ssh --project avalam-dqn-367921 --zone us-west1-b deeplearning-1-vm -- -L 8080:localhost:8080                                                                                                                            

vm2:
	gcloud compute ssh --project avalam-dqn-367921 --zone us-west2-a deeplearning-2-vm -- -L 8080:localhost:8080
	
get_scores:
	gcloud compute scp --project avalam-dqn-367921 --zone us-west1-b --recurse deeplearning-1-vm:~/Avalam-DQN-Agent/figs/scores/ ./test-models/ 
