### Overview


### Task1
The challenge gives us leakage hypothesis around final-state register, means when the transition is from S9 to Ciphertext. We can calculate the Hamming Distance between these transition, and in this transition fake key is used. We can use CPA and then choose the candidate with highest absolute correlation peak to be the winner and thus obtaining the fake key byte.

In solve.py code, we use the value of k from 0x00 to 0xff, and calculate H(k), suppose Hj(k) is predicted leakage for the byte k of trace j. And the power at trace j  at time t is Pj(t). So for candidate k and t, we have two vectors H(k) and P(t).

We calculate pearson correlation between these two which gives us a number between -1 to 1. As the readme stated we take the absolute correlation value.
and then we calculate Score = max(over time)absolute(rho(k,t))
for one byte k and time sample from 0 to 200.

TODO: mathematical proof and clean the comments from code

### Task 2
In this we have to find out, the minimum number of traces required to calculate the fake key.
