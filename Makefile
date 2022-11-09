clean: clean_games	clean_losses clean_figs

clean_figs:
	rm -rf ./figs/loss/*
	rm -rf ./figs/scores/*

clean_games:
	rm -rf ./logs/games/*

clean_losses:
	rm -rf ./logs/results/*
