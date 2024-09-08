# Toy simulations


We consider four toy simulations:
- **againstsupervisedSimulation:** The oracle solves a MCFP. The costs of the MCFP are uniformly distributed $\sim U(0, 100)$.
- **easySimulation:** The oracle solves a MCFP. The costs of the MCFP equal the street length.
- **easySimulationEquilibrium:** The oracle solves a Wardrop Equilibrium (WE). The latency function of the WE for edge $e$ is $length^e + y^e$ with $length^e$ representing the length of edge $e$ and $y^e$ denoting the aggregated flow on edge $e$.
- **randomSimulationEquilibrium:** The oracle solves a Wardrop Equilibrium (WE). The latency function of the WE for edge $e$ is $\theta_{1,e} + \theta_{2,e} ∗ y^e$ with $\theta_{1,e} \sim U(1, 100)$ randomly distributed, $\theta_{2,a} \sim U(1, 20)$ randomly distributed, and $y^e$ denoting the accumulated flow on edge $e$.