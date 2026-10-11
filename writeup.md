### Overview

### Task1

The challenge gives us leakage hypothesis around final-state register, means when the transition is from S9 to Ciphertext. We can calculate the Hamming Distance between these transition, and in this transition fake key is used. We can use CPA and then choose the candidate with highest absolute correlation peak to be the winner and thus obtaining the fake key byte.

In solve.py code, we use the value of k from 0x00 to 0xff, and calculate H(k), suppose Hj(k) is predicted leakage for the byte k of trace j. And the power at trace j at time t is Pj(t). So for candidate k and t, we have two vectors H(k) and P(t).

We calculate pearson correlation between these two which gives us a number between -1 to 1. As the readme stated we take the absolute correlation value.
and then we calculate Score = max(over time)absolute(rho(k,t))
for one byte k and time sample from 0 to 200.

TODO: mathematical proof and clean the comments from code

### Task 2

In this we have to find out, the minimum number of traces required to calculate the fake key. In second_task.py, we first calculate the refernce key which is the fake key, and for different number of traces N, we calculate the score(according to absolute correlation) for a byte position and then for the byte position we find what is expected position of the reference key in this trace, and thus finding rank. If the rank is 1 for all the byte position then that means we found the minimum Number of traces.

### Task 3 — actual round-10 key

We have obtained fake key in Task1 and Minimum number of traces required to find the fake key in Task2. In this, we obtained the actual round10 key. According to the Readme doc, The final actual round10 key is changed to some FRK10 key so that the correlation plot will point to the fake key instead of the actual RK10 key. But all other round keys are same in both with and without countermeasures. 

One thing to note in task1 is that state register transitioned from FRK10 key to C( ciphertext) at time sample 49. If the time samples are continuos, and error free then there must be power traces recorded at each state transition. In simple terms, everytime Srminus1 transitions to Sr, there's more power consumed recorded than usual. 

### Task 4 — actual master key


