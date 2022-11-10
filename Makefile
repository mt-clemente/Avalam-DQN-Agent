clean: clean_games	clean_stdout clean_figs

clean_figs:
	rm -rf ./figs/scores/*

clean_games:
	rm -rf ./logs/games/*

clean_stdout:
	rm -rf ./logs/stdout/*
