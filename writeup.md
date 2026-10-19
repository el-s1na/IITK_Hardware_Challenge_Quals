### Overview

### Task1

The challenge gives us leakage hypothesis around final-state register, means when the transition is from S9 to Ciphertext. We can calculate the Hamming Distance between these transition, and in this transition fake key is used. We can use CPA and then choose the candidate with highest absolute correlation peak to be the winner and thus obtaining the fake key byte.

In solve.py code, we use the value of k from 0x00 to 0xff, and calculate H(k), suppose Hj(k) is predicted leakage for the byte k of trace j. And the power at trace j at time t is Pj(t). So for candidate k and t, we have two vectors H(k) and P(t).

We calculate pearson correlation between these two which gives us a number between -1 to 1. As the readme stated we take the absolute correlation value.
and then we calculate Score = max(over time)absolute(rho(k,t))
for one byte k and time sample from 0 to 200.

HD=HW(InvSBox(C[i]^k)^C[INV_SR[i]])

TODO: mathematical transition between A and B.
TODO: mathematical proof and clean the comments from code

### Task 2

In this we have to find out, the minimum number of traces required to calculate the fake key. In second_task.py, we first calculate the refernce key which is the fake key, and for different number of traces N, we calculate the score(according to absolute correlation) for a byte position and then for the byte position we find what is expected position of the reference key in this trace, and thus finding rank. If the rank is 1 for any byte position, means the highest correlation byte is same as reference key for some number of traces less than N. And for all the 16 byte positions, if the worst rank is 0x1, then that trace is minimum for which we can obtain the whole fake key.
TODO: explain RANDOM SEED

### Task 3 — actual round-10 key

We have obtained fake key in Task1 and Minimum number of traces required to find the fake key in Task2. In this, we obtained the actual round10 key. According to the Readme doc, The final actual round10 key is changed to some FRK10 key so that the correlation plot will point to the fake key instead of the actual RK10 key. But all other round keys are same in both with and without countermeasures.

One thing to note in task1 is that state register transitioned from FRK10 key to C( ciphertext) at time sample 49. If the time samples are continuos, and error free then there must be power traces recorded at each state transition. In simple terms, everytime Srminus1 transitions to Sr, there's more power consumed recorded than usual.

But we have another problem, how to identify these activity bursts happening during the round transition. And Variance comes to the rescue.
TODO:mathematical equations regarding th3 variance

In variance_plot.png, we can see that there is around 9 local maxima before time sample 49, and at detected time sample blah blah...


It is mentioned in the readme doc that final ciphertext which is produced without any changes, so final equation is
C\_{b}=SBox(z\_{b})⊕ RK10\_{b} where z_b is the Sbox input that ShiftRows routes to ciphertext position b.

After observing histogram, we found that for every byte position there is a byte value that never appears i.e. M_b and there is a byte value which appears more than normal which is D_b. And there is a relation between D_b and F_b(fake key obtained in task1) which is
D\_{b}⊕ F\_{b}=SBox(0). The most alien observation here is that for every byte position of D_b and F_b, there is only input in Sbox.

C\_{i,b}=f\_{b}(X\_{i,b}),  f\_{b}(x)=SBox(x)⊕ RK10\_{b}
The histogram says M\_{b} never appears among the observed C\_{i,b} values. Since f\_{b} is one-to-one and maps all 256 byte inputs to all 256 byte outputs, there is a unique input z\_{b} that would produce M\_{b}

M\_{b}=f\_{b}(z\_{b}) =SBox(z\_{b})⊕ RK10\_{b}.

After assuming shared-z hypothesis, we come to the equation M\_{b}=SBox(z\_{b})⊕ RK10\_{b}.
Now for every value of z from 0x0 to 0xff, we can calculate a different RK10, and from there we can find all the previous round keys. Using the leakage model hypothesis(same as Task1), we can find for which values of z, we can get the highest correlation at the detected time sample.

At each time sample t, correlation plot is calculated across traces. To find the ranking of z, which suggests the highest correaltion, we have to take averagei.e. Score(z)


TODO: Mathematical equations

So for z=0xff at every round from 1 to 8, and at every byte position, there is a peak in absolute correlation plot, which is not present in the other cases.
So this is highly likely that for z=0xff, the RK10 derived is the actual RK10 key.
After that we can find the master key.



### Task 4 — actual master key

We have obtained the actual round-10 key in Task 3. Since the AES-128 key schedule is reversible, we can use this round key to calculate the initial AES master key. In fourth_task.py, we start with the round-10 key `d014f9a8c9ee2589e13f0cc8b6630ca6` and calculate the expanded key words backward.

The AES-128 expanded key contains 44 words, each of 4 bytes, written as $w[0], w[1], \ldots, w[43]$. The first four words form the master key, while the last four words form the round-10 key. During forward expansion, each word is calculated using

$$
w[i] = w[i-4] \oplus F_i(w[i-1]).
$$

At every fourth word, $F_i$ applies RotWord, SubWord and the round constant. At all other positions, it leaves the input unchanged:

$$
F_i(x) =
\begin{cases}
\operatorname{SubWord}(\operatorname{RotWord}(x)) \oplus Rcon[i/4], & i \equiv 0 \pmod{4}, \\
x, & \text{otherwise}.
\end{cases}
$$

Here, RotWord moves the first byte to the end of the four-byte word, and SubWord replaces each byte with its value from the forward AES S-box. The round constant is placed in the first byte of a four-byte word, with the remaining bytes set to zero.

To calculate an earlier word, we XOR both sides of the forward equation with $F_i(w[i-1])$. Since XORing the same value twice cancels it, we get

$$
w[i-4] = w[i] \oplus F_i(w[i-1]).
$$

We place the recovered round-10 key into $w[40], \ldots, w[43]$, and run this calculation for $i=43, 42, \ldots, 4$. For example, the first step gives $w[39] = w[43] \oplus w[42]$. At each step, the words required on the right-hand side are already known. Continuing backward gives all the words down to $w[0]$.

One thing to note is that reversing the key schedule still uses the forward S-box. We already know the input $w[i-1]$ to $F_i$, so we can calculate its output directly. The earlier word is then recovered using XOR, so we do not need the inverse S-box for this calculation.

The round constants are used in reverse round order. At $i=40, 36, \ldots, 4$, their first bytes are `36, 1b, 80, 40, 20, 10, 08, 04, 02, 01` in hexadecimal. In fourth_task.py, the RCON list stores these constants in forward order with indexing starting at zero. Therefore, the script uses `RCON[i // 4 - 1]`; for example, $i=40$ selects entry 9, which is `0x36`.

After reaching the first four words, we join $w[0], \ldots, w[3]$ in their original order to get the master-key candidate:

`2b7e151628aed2a6abf7158809cf4f3c`

To check the calculation, the script expands this master key forward and compares its round-10 key with the input from Task 3. The resulting round-10 key is `d014f9a8c9ee2589e13f0cc8b6630ca6`, which matches the input. This confirms that the backward and forward key-schedule calculations agree for the selected round key. The evidence that this is the device's actual key comes from the trace analysis in Task 3 and depends on the assumptions used there.
