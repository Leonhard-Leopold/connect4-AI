# connect4-AI
Connect 4 AI with MCTS and a Neural Network

A Connect-4 AI implemented in Python.

The AI chooses moves with the help of a Monte Carlo Search Tree which gets predictions from a neural network.
The Code is documented throughout.
Inside the "run 1" folder is pre-trained version that was trained for 1 hour (plays at a competent but not perfect level)

	Options:  
        playing human vs human 
        human vs AI 
        training the AI vs itself

	Usage:  
        connect4.py (untrained AI vs human player)
        connect4.py -x a -o h --run 1 --ver 999 -- mem 999 (human testing the pre-trained version)
        connect4.py -x a -o a --run 1 --ver 999 -- mem 999 (continue training the pre-trained version)
        connect4.py -x a -o a (start training an untrained version)
        connect4.py -x h -o h (human vs human)

	Command Line Arguments:
        -x (choosing the type of player 1; a = AI; h = human)
        -o (choosing the type of player 2; a = AI; h = human) 
        --run (choosing the run folder)
        --ver (choosing the version of a trained neural network model)
        --mem (loading saved action values - not particularly necessary; if memory is not loaded action values are created in training)
        



*Credit to: https://github.com/AppliedDataSciencePartners*
