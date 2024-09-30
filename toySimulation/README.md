# Toy simulations


We consider four toy simulations:
- **againstsupervisedSimulation:** The oracle solves a MCFP. The costs of the MCFP are uniformly distributed $\sim U(0, 100)$.
- **easySimulation:** The oracle solves a MCFP. The costs of the MCFP equal the street length.
- **easySimulationEquilibrium:** The oracle solves a Wardrop Equilibrium (WE). The latency function of the WE for arc $a$ is $d_a + y_a$ with $d_a$ representing the length of arc $a$ and $y_a$ denoting the aggregated flow on arc $a$.
- **randomSimulationEquilibrium:** The oracle solves a Wardrop Equilibrium (WE). The latency function of the WE for arc $a$ is $\theta_{1,a} + \theta_{2,a} ∗ y_a$ with $\theta_{1,a} \sim U(1, 100)$ randomly distributed, $\theta_{2,a} \sim U(1, 20)$ randomly distributed, and $y_a$ denoting the accumulated flow on arc $a$.