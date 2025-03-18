# CsnSubclass

CsnSubclass is a Julia package that provides tools for simulating statistical models considering spatio-temporal structures and skewness. It also facilitates sampling from the posterior distribution of these models. Please refer to Kuno and Murakami (2024) for details.

## Installation

To install CsnSubclass, use the following command in Julia's REPL:

```julia
using Pkg
Pkg.add("https://github.com/Kuno3/CsnSubclass")
```

## Importing the Package

To start using CsnSubclass, simply import the package:

```julia
using CsnSubclass
```

## Simulation

CsnSubclass allows you to specify parameters for your spatiotemporal model and run simulations. Here’s an example of how to set parameters and execute a simulation:

### Usage

`CsnSubclassSimulator`

**Description**: Initializes the simulation environment with specified parameters for D-FS-CSN.

**Arguments**:
- `T::Int`: Time length
- `W::Array{Float64,2}`: Adjacency matrix
- `feature::Array{Float64,3}`: Features
- `beta::Array{Float64,1}`(optional): Coefficinet
- `sigma2::Float64`(optional): Variance of observation noise
- `tau2::Float64`(optional): Variance of process noise
- `rhoS::Float64`(optional): Parameter for spatial dependence
- `rhoT::Float64`(optional): Parameter for temporal dependence
- `l::Float64`(optional): Parameter for skewness
- `theta_init::Array{Float64,1}`(optional): Initial value of theta
- `seed::Int`(optional): Seed for random sampling

**Return**:
- `CsnSubclassSimulator`: An object that holds the simulation environment.

`DCarSimulator`

**Description**: Initializes the simulation environment with specified parameters for D-CAR.

**Arguments**:
- `T::Int`: Time length
- `W::Array{Float64,2}`: Adjacency matrix
- `feature::Array{Float64,3}`: Features
- `beta::Array{Float64,1}`(optional): Coefficinet
- `sigma2::Float64`(optional): Variance of observation noise
- `tau2::Float64`(optional): Variance of process noise
- `rhoS::Float64`(optional): Parameter for spatial dependence
- `rhoT::Float64`(optional): Parameter for temporal dependence
- `theta_init::Array{Float64,1}`(optional): Initial value of latent variable theta
- `seed::Int`(optional): Seed for random sampling

**Return**:
- `DCarSimulator`: An object that holds the simulation environment.

`simulate`

**Description**: Simulates data based on the initialized model.

**Arguments**:
- `simulator::CsnSubclassSimulator` or `simulator::DCarSimulator`: The initialized simulator object.

**Return**:
- `Nothing`: The simulation result are stored in `simulator.y`

### Example

```julia
# Define parameters
T = 10 # Time length
W = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0] # Adjacency matrix
feature = randn(T, size(W, 1), 2) # Features
simulator = CsnSubclassSimulator(T, W, feature)
simulate(simulator)
```

## Sampling from the Posterior Distribution

The package also includes functions for sampling from posterior distributions, allowing for in-depth statistical analysis and inference.

### Usage

`CsnSubclassPostSampler`

**Description**: Initializes the sampling environment with specified parameters for D-FS-CSN.

**Arguments**:
- `y::Array{Float64,2}`: Observations
- `W::Array{Float64,2}`: Adjacency matrix
- `nu2_beta::Float64`(optional): Variance of the prior for beta (default is 100)
- `a_sigma2::Float64`(optional) and `b_sigma2::Float64`(optional): Parameter of the inverse gamma prior for sigma2 (default is 1 and 0.01)
- `a_tau2::Float64`(optional) and `b_tau2::Float64`(optional): Parameter of the inverse gamma prior for tau2(default is 1 and 0.01)
- `nu2_l::Float64`(optional): Variance of the prior for l(default is 100)
- `seed::Int`(optional): Seed for random sampling

**Return**:
- `CsnSubclassPostSampler`: An object that holds the sampling environment.

`DCarPostSampler`

**Description**: Initializes the sampling environment with specified parameters for D-CAR.

**Arguments**:
- `y::Array{Float64,2}`: Observations
- `W::Array{Float64,2}`: Adjacency matrix
- `nu2_beta::Float64`(optional): Variance of the prior for beta (default is 100)
- `a_sigma2::Float64`(optional) and `b_sigma2::Float64`(optional): Parameter of the inverse gamma prior for sigma2 (default is 1 and 0.01)
- `a_tau2::Float64`(optional) and `b_tau2::Float64`(optional): Parameter of the inverse gamma prior for tau2(default is 1 and 0.01)
- `seed::Int`(optional): Seed for random sampling

**Return**:
- `DCarPostSampler`: An object that holds the sampling environment.

`sampling`

**Description**: Simulates data based on the initialized model.

**Arguments**:
- `sampler::CsnSubclassPostSampler` or `sampler::DCarPostSampler`: The initialized simulator object.
- `num_sample::Int`: Number of samples
- `burn_in::Int`(optional): Number of saples for burn in
- `thinning::Int`(optional): Number of saples for thinning
- `step_size::Float64`(optional and only for CsnSubclassPostSampler): Parameter for sampling
- `num_step::Int`(optional and only for CsnSubclassPostSampler): Parameter for sampling
- `rhoS_prop_scale::Float64`(optional): Parameter for sampling
- `theta_init::Matrix{Float64}`(optional): Initial value for theta
- `alpha_init::Matrix{Float64}`(optional): Initial value for alpha
- `beta_init::Vector{Float64}`(optional): Initial value for beta
- `sigma2_init::Vector{Float64}`(optional): Initial value for sigma2
- `tau2_init::Vector{Float64}`(optional): Initial value for tau2
- `rhoS_init::Vector{Float64}`(optional): Initial value for rhoS
- `rhoT_init::Vector{Float64}`(optional): Initial value for rhoT
- `l_init::Vector{Float64}`(optional and only for CsnSubclassPostSampler): Initial value for l

**Return**:
- `Array`: A dictionary of the samples from the posterior distribution.

### Example

```julia
# Define observations
y = simulator.y
W = simulator.W
feature = simulator.feature

# Sample from the posterior distribution
sampler = CsnSubclassPostSampler(y, W, feature)
results = sampling(sampler, 1000)
```

## License

CsnSubclass is licensed under the MIT License.

## Reference
Kuno, H., Murakami, D., 2024. Modeling heterogeneity in higher-order moments while preserving mean and variance: application to spatio-temporal modeling. Preprint