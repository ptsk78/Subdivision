# [Subdivision algorithm](https://link.springer.com/article/10.1007/s002110050240) using [PyOpencl](https://github.com/inducer/pyopencl)

Unlike subdivision algorithm, this software package subdivides in all dimensions at the same time. So minor differences in comparison with [Approximation of box dimension of attractors using the subdivision algorithm](https://www.tandfonline.com/doi/abs/10.1080/14689360500141772). Now with int64 support, you can theoretically go to ((r=20)+1)*(dim=3)=63<64, which is quite furher than the dimension paper.

Chaotic attractor

    pepe@pepe-ms7c90:~/code/pepe78/subdivision_opencl$ python3 main.py
    0 - chaotic
    1 - rossler
    2 - lorenz
    3 - aizawa
    4 - halvorsen
    system = 0
    r = 3 , Number of active boxes = 512
    Number of testing points = 13824000
    Number of active boxes for next step = 1066
    r = 4 , Number of active boxes = 1066
    Number of testing points = 28782000
    Number of active boxes for next step = 2181
    r = 5 , Number of active boxes = 2181
    Number of testing points = 58887000
    Number of active boxes for next step = 5782
    r = 6 , Number of active boxes = 5782
    Number of testing points = 156114000
    Number of active boxes for next step = 15218
    r = 7 , Number of active boxes = 15218
    Number of testing points = 410886000
    Number of active boxes for next step = 41431
    r = 8 , Number of active boxes = 41431
    Number of testing points = 1118637000 -> Yes, you are testing over one billion points on GPU
    Number of active boxes for next step = 112544

(2^6)^3 = 262,144 boxes

![chaotic](./images/chaotic_0005.png)

(2^9)^3 = 134,217,728 boxes

![chaotic](./images/chaotic_0008.png)

(2^12)^3 = 68,719,476,736 boxes

![chaotic](./images/chaotic_0011.png)

Rossler attractor

(2^6)^3 = 262,144 boxes

![rossler](./images/rossler_0005.png)

(2^9)^3 = 134,217,728 boxes

![rossler](./images/rossler_0008.png)

Lorenz attractor

(2^6)^3 = 262,144 boxes

![lorenz](./images/lorenz_0005.png)

(2^9)^3 = 134,217,728 boxes

![lorenz](./images/lorenz_0008.png)

Halvorsen attractor

(2^6)^3 = 262,144 boxes

![halvorsen](./images/halvorsen_0005.png)

(2^9)^3 = 134,217,728 boxes

![halvorsen](./images/halvorsen_0008.png)

Or in 3D with [Anaglyph 3d glasses needed](https://en.wikipedia.org/wiki/Anaglyph_3D) (red / cyan glasses):

![halvorsen](./images/01.png)

![halvorsen](./images/02.png)

![halvorsen](./images/03.png)

![halvorsen](./images/04.png)

![halvorsen](./images/05.png)
