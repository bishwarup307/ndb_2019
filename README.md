# Kaggle - National Data Science Bowl 2019

| Public Leeaderboard(QWK) | Private Leaderboard(QWK) |
|-------------------------:|-------------------------:| 
| 593(0.551)               | 62(0.550)                |

### Features
This repository contains my solution for the National Data Science Bowl 2019 held in [Kaggle](https://www.kaggle.com/c/data-science-bowl-2019) over the last 3 months. The solution in itself is pretty straightforward but I manually engineered a number of features which I found very interesting.

1. Individual `game` / `activity` performance features:
    - for each of the `game` and `activity` I looked into mainly 3 things (they are different for each of the games and activities but kind of follow the below pattern):
    a. `weight`: Assign weights to each round of the game as per their difficulty (I played the games - so the weights were subjective)
    b. `round`: Number of rounds completed
    c. `misses`: Number of misses each round

Then the accuracy/performance of the game/activity was measured by something like:
```math
acc = weights * exp{-misses * penalty}
```
where a `penalty` was assigned for skipping `rounds`. The equations for penalty are subjective and I played the games over and over to tune the numbers so it's not a good approach but worked out pretty well at the end.

2. Path Efficiency:
I calculated the efficiency of the `path` taken by a child to arrive at the assignment. It was done by a very similar process to what we use for full backward tree search in dynamic programming a discrete time Markov Reward Process (MRP) with a discounting. This was driven by the fact that child who skips or deviates from the sequence of games designed by PBS KIDS, often does a poorly in an assignment.

The above two features helped the model quite a bit.

Apart from that past accuracy, count and a few onehot features were used. In total there were only ~35-40 features in my final model.

### Model
The final model is a plain single LGB bagged 10 times. Each of the bag takes around ~2 minutes including the CV in my 32-core machine to run. I didn't get much time to tune the hyperparameters so used the one from a few I tried out manually. In whole the complete solution runs in ~30 minutes.
