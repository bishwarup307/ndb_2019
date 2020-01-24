# Kaggle - National Data Science Bowl 2019

| Public Leeaderboard(QWK) | Private Leaderboard(QWK) |
|-------------------------:|-------------------------:| 
| 593(0.551)               | 62(0.550)                |

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
I calculated the efficiency of the `path` taken by a child to arrive at the assignment. It was done by a very similar process to what we use for full backward tree search in a discrete time Markov Reward Process (MRP) with a discounting. 

Apart from that