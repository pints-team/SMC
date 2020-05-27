import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import scipy
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
plt.rcParams['figure.figsize'] = (17, 7)


# NOTE: By a "particle" we mean an array of values which represents the underlying state of the system. 
# E.g. particle = [3] is a representation of a system whose only state value is 3, while particle = [2, 5] represents a two-dimensional state


class SimpleModel:
    """ 
    To be used when defining a linear gaussian model by its 4 matrices
    """
    def __init__(self, x_0, F, H, Q, R):
        self.F = F # the transition matrix
        self.H = H # the observation matrix
        self.Q = Q # the transition error covariance matrix
        self.R = R # the measurement error covariance matrix
        self.states = [x_0] # x_0 is the initial state of the system
        self.measurements = [np.matmul(self.H, self.states[0]) + multivariate_normal(mean=[0] * len(self.R), cov=self.R)] # initialising with the first measurement
        self.state_dim = len(Q) # the dimension of the underlying state
        self.measurement_dim = len(R) # the dimension of the observed state
                
    def step(self, particle):
        """
        Transitions the particle one step in time by multiplying it with the transition matrix F and adding the step noise
        """
        step_noise = lambda : multivariate_normal(mean=np.zeros(len(self.Q)), cov=self.Q) # 0-mean Q-covariance Gaussian noise
        return np.matmul(self.F, particle) + step_noise()
    
    def measure(self, t):
        """
        Returns the t^th observation of the system. Uses memoization to store the values of precomputed steps.
        """
        if t < len(self.measurements):
            return self.measurements[t]
        else:
            # x_{n+1} = F * x_n + noise, noise ~ N(0, Q).
            # Equivalent to: new_state = self.step(self.states[-1]).
            # self.states[-1] denotes the last computed state
            new_state = np.matmul(self.F, self.states[-1]) + multivariate_normal(mean=[0] * len(self.Q), cov=self.Q)

            # y_{n+1} = H * x_{n+1} + noise, noise ~ N(0, R)
            new_measurement = np.matmul(self.H, new_state) + multivariate_normal(mean=[0] * len(self.R), cov=self.R)

            self.states.append(new_state)
            self.measurements.append(new_measurement)
            return self.measure(t)
    
    def eval(self, particle):
        """
        Evaluates a particle by multiplying it with the observation matrix H
        """
        return np.matmul(self.H, particle)


class Event:
    """
    To be used as part of the Tau Leaping calculation, nothing but a nice wrapper around the two functions defined abstractly in init
    """
    def __init__(self, rate_calculator, change_function):
        self.rate_calculator = rate_calculator
        self.change_function = change_function


class StochasticModelTauLeaping:
    """
    Implements the Tau Leaping algorithm on a stochastic model defined by a collection of events, using a constant step_size
    """
    def __init__(self, x_0, events, step_size, R):
        """
        events has to be a list of Events, R is the assumed measurement error covariance matrix.
        x_0 is the initial state. step_size is an arbitrary step size to be used in Tau Leaping.
        We assume that the measurement matrix is the identity, which means that measurement = underlying state + measurement error (0-mean R-cov gaussian)
        """
        self.events = events
        self.x_0 = x_0
        self.step_size = step_size
        self.R = R
        self.states = [x_0]
        self.measurements = [x_0 + multivariate_normal(mean=[0] * len(self.R), cov=self.R)]
        self.state_dim = len(R)

    def step(self, particle):
        """
        Advance the particle one step ahead in time, and by step we mean the step_size given initially
        """
        for e in self.events:
            # Kj = approximately how many times the event e would occur in the given step_size, computed by a Poisson distribution
            Kj = np.random.poisson(self.step_size * e.rate_calculator(particle))

            # Apply the event e's change function Kj times
            for _ in range(Kj):
                particle = e.change_function(particle)
        return particle

    def measure(self, t):
        """
        Returns the t^th observation of the system. Uses memoization to store the values of precomputed steps.
        """
        if t < len(self.measurements): # We have all we need
            return self.measurements[t]
        else: # We need to simulate the model for more steps
            diff = t - len(self.measurements) + 1 # How many steps are we missing?
            for _ in range(diff): 
                new_state = self.step(self.states[-1]) # Obtain a new state by applying the step function to the last state
                self.states.append(new_state)
                new_measurement = new_state + multivariate_normal(mean=[0] * len(self.R), cov=self.R) # As we said, we observe the state just by adding some noise
                self.measurements.append(new_measurement)
            return self.measure(t)

    def eval(self, particle):
        """
        As mentioned before, we've taken the evaluation function to be identity 
        """
        return particle
    
    
class ParticleFilter:
    def __init__(self, model, N, measurements, neff_threshold=1/3, resampling_algo="stratified_resample", prior_cov_matrix=None, ranges=None, print_results=False):
        """
        Initialising the particle filter with all the required parameters:
        model: the model, which should be either a SimpleModel (e.g. linear gaussian problems) or a StochasticModelTauLeaping (e.g. logistic growth problems)
        N: the number of particles to be generated
        measurements: the precomputed set of measurements. Usually obtained by [model.measure(t) for t in range(steps)]
        neff_threshold: the effective N threshold. if neff < neff_threshold * N, we do a resampling step
        resampling_algo: the name of one of the 4 resampling algorithms implemented (simple_resample, stratified_resample, residual_resample, systematic_resample)
        (optional) prior_cov_matrix: the assumed prior covariance matrix. To be provided if one wants the initial distribution to be gaussian, centered around model.x_0 and with this covariance matrix
        (optional) ranges: a list of ranges for uniformly initialising the particles. E.g.: [[0, 1], [-20, 100]] for a 2D model
        (optional) print_results: False by default. useful for debugging/visual purposes
        """
        self.model = model
        self.N = N
        self.measurements = measurements
        self.neff_threshold = neff_threshold

        # This is an example of bad practice and developing algorithms on the fly (because of time constraints). 
        # self.resampling_algo has to be a resampling algorithm that returns the indices of the particles that we want to resample. We've implemented 4 such algorithms.
        if resampling_algo == "simple_resample":
            self.resampling_algo = self.simple_resample
        elif resampling_algo == 'stratified_resample':
            self.resampling_algo = self.stratified_resample
        elif resampling_algo == 'residual_resample':
            self.resampling_algo = self.residual_resample
        elif resampling_algo == 'systematic_resample':
            self.resampling_algo = self.systematic_resample
        else:
            assert False, f"Resampling algorithm {resampling_algo} unknown"
        
        
        if ranges is not None:
            self.particles = self.uniform_particles(N, model, ranges)
        else:
            assert prior_cov_matrix is not None, "Prior covariance matrix missing"
            self.particles = self.normal_particles(N, model, model.states[0], prior_cov_matrix)
        
        # Initialising all the weights equal
        self.weights = np.ones(N) / N
        self.prior_cov_matrix = prior_cov_matrix

        # The PF can only be used once. Re-running it is not currently a feature.
        self.used = False
        self.print_results = print_results
        
    def plot_particles(self):
        """
        Plots the distribution of the particles. Requires seaborn (for seaborn.distplot)
        """
        fig, axes = plt.subplots(nrows=self.model.state_dim)
        if self.model.state_dim == 1:
            axes = [axes]
        for i in range(self.model.state_dim):
            sns.distplot(self.particles[:, i], ax=axes[i])    
        
    def uniform_particles(self, N, model, ranges):
        """
        Initializes the particles uniformly according to the ranges provided, making use of numpy.random.uniform
        """
        particles = np.empty((N, model.state_dim))
        for i, (start, end) in enumerate(ranges):
            particles[:, i] = np.random.uniform(low=start, high=end, size=N)
        return particles
    
    def normal_particles(self, N, model, x_0, prior_cov_matrix):
        """
        Initializes the particles according to a Gaussian distribution, making use of numpy.random.normal
        """
        particles = np.empty((N, model.state_dim))
        for i in range(model.state_dim):
            particles[:, i] = np.random.normal(loc=x_0[i], scale=prior_cov_matrix[i][i], size=N)
        return particles
    
    def predict(self, model, particles):
        """
        Advances the collection of particles by one step in time.
        Maps the model step function across all our particles.
        """
        return np.array(list(map(model.step, particles)))
    
    def update(self, z, model, particles, weights): 
        """
        Given a new measurement (z), updates the weights of the particles
        """
        observations = np.array(list(map(model.eval, particles)))
        
        weights *= scipy.stats.multivariate_normal.pdf(x=observations, mean=z, cov=self.model.R) 
        weights += 1.e-100 # avoid round off to zero
        weights /= np.sum(weights) # normalize

        return weights
    
    def estimate(self, particles, weights):
        """
        Returns the mean and variance of the particles according to their weights
        """
        mean = np.average(particles, weights=weights, axis=0)
        var = np.sqrt(np.average((particles - mean)**2, weights=weights, axis=0))
        return mean, var
    
    def resample(self, particles, weights):
        """
        Returns a new set of particles and weights by performing a resampling
        """
        indexes = self.resampling_algo(particles, weights) # the resampling algorithm returns a list of indices of the particles to be resampled
        particles[:] = particles[indexes] # getting the resampled particles
        weights[:] = weights[indexes] # getting the resampled particles' weights
        weights.fill(1.0/len(particles))
        return particles, weights
    
    def neff(self, weights):
        """
        Computes the effective N
        """
        return 1. / np.sum(np.square(weights))
        
    def run(self):
        """
        Actually running the PF. Returns the posterior means, variances and the overall likelihood
        """

        # Make sure the PF is only runnable once
        assert not self.used, "PF already ran. Please instantiate a new PF"
        self.used = True

        time = len(self.measurements)
        means = []
        variances = []
        likelihood = 0

        # These two lines follow Doucet 2000's likelihood calculation precisely, also explained in the report
        p_ykxk = lambda i: scipy.stats.multivariate_normal.pdf(self.model.states[0], mean=self.model.eval(self.particles[i]), cov=self.model.R) # p(yk|xki)
        Pyk = np.sum([p_ykxk(i) * self.weights[i] for i in range(self.N)]) # Phat(y_k|y_0:k-1)
        
        # This if statement's only purpose is suppressing numpy warnings by explicitly saying log(0) = -inf. Can be replaced by the else branch alone. 
        # Numpy warnings are quite spammy when doing many experiments.
        if Pyk == 0: #np warnings are annoying
            likelihood -= np.inf
        else:
            likelihood += np.log(Pyk)


        
        #print(f"Running PF with {self.N} particles: ")#, end="")
        initial_mean, initial_variance = self.estimate(self.particles, self.weights)
        means.append(initial_mean)
        variances.append(initial_variance)

        for i in range(time-1): # For each of the measurements, do the routine of predict, update, estimate
            z = self.measurements[i+1]

            # The predict step
            self.particles = self.predict(self.model, self.particles)           

            # The likelihood calculation
            p_ykxk = lambda i: scipy.stats.multivariate_normal.pdf(z, mean=self.model.eval(self.particles[i]), cov=self.model.R) # p(yk|xki)
            Pyk = np.sum([p_ykxk(i) * self.weights[i] for i in range(self.N)]) # Phat(y_k|y_0:k-1)
            if Pyk == 0:
                likelihood -= np.inf
            else:
                likelihood += np.log(Pyk)

            # The update step
            self.weights = self.update(z, self.model, self.particles, self.weights)

            # The estimation step
            mean, var = self.estimate(self.particles, self.weights)

            # Keeping track of the results
            means.append(mean)
            variances.append(var)            
            
            # Is a resampling step necessary?
            if self.neff(self.weights) < len(self.particles) * self.neff_threshold:
                self.particles, self.weights = self.resample(self.particles, self.weights)

        if self.print_results: 
            # Printing some results. Highly customizable.

            print("Marginal likelihood: ", round(likelihood, 4))

            fig = plt.figure()
            # plt.title('Particle Filter Results', fontsize=16, y=1.08)
            means = np.array(means)
            variances = np.array(variances)
            measurements = np.array(self.measurements)
            times = range(time)
            for i in range(self.model.state_dim):
                ax = fig.add_subplot(self.model.state_dim, 1, i+1)
                # ax.set_title(f"dimension {i}")
                ax.plot(times, means[:, i], 'b--', label='PF means')

                # ax.plot(times, means[:, i] + variances[:, i], 'c--', label='PF variance')
                # ax.plot(times, means[:, i] - variances[:, i], 'c--')
                ax.fill_between(times, means[:, i] + variances[:, i], means[:, i] - variances[:, i], alpha=0.2)

                ax.plot(times, measurements[:, i], 'ko', label='measurements', markersize=3)
                ax.set_xlabel("timestep")
                ax.set_ylabel("value")
                ax.legend()
            

            fig.tight_layout()
            plt.savefig('PF_1D_run.pdf')

            plt.show()
        return means, variances, likelihood
    
    def simple_resample(self, particles, weights):
       """ The multinomial resampling algorithm. Detailed explanation in the report. """
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, N))
        return indexes

    def residual_resample(self, particles, weights):
        """ The residual resampling algorithm. Detailed explanation in the report. """

        N = len(weights)
        indexes = np.zeros(N, 'i')
        # take int(N*w) copies of each weight
        num_copies = (N*np.asarray(weights)).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]): # make n copies
                indexes[k] = i
                k += 1

        # use multinormial resample on the residual to fill up the rest.
        residual = (weights * N - num_copies) / N     # get fractional part
        residual /= sum(residual)     # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1. # ensures sum is exactly one
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, N-k))

        return indexes

    def stratified_resample(self, particles, weights):
        """ The stratified resampling algorithm. Detailed explanation in the report. """
        N = len(particles)
        # make N subdivisions, chose a random position within each one
        positions = (np.random.uniform(0, 1, N) + range(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    
    def systematic_resample(self, particles, weights):
        """ The systematic resampling algorithm. Detailed explanation in the report. """
        N = len(particles)

        # make N subdivisions, choose positions 
        # with a consistent random offset
        positions = (np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    

from filterpy.kalman import KalmanFilter as KF
class Kalman:
    """
    This is just a wrapper around the filterpy.kalman.KalmanFilter class to be used in our experiments, adding the likelihood functionality
    """
    def __init__(self, model, measurements, print_results=False):
        """
        model: SimpleModel or StochasticModelTauLeaping
        measurements: the precomputed measurements of the model, for e.g. [model.measure(t) for t in range(time)]
        (optional) print_results: useful for debugging purposes
        """
        self.kf = KF(dim_x=model.state_dim, dim_z=model.measurement_dim) # instantiating the underlying KF from filterpy
        self.measurements = measurements

        self.kf.x = model.states[0]
        self.kf.F = model.F
        self.kf.H = model.H
        self.kf.Q = model.Q
        self.kf.R = model.R

        self.model = model
        self.print_res = print_results
    
    def run(self):
        """
        Running the KF and computing the likelihood. Returns the posterior means, variances and overall likelihood
        """
        # Actually running the KF
        post_means, post_covs, prior_means, prior_covs = self.kf.batch_filter(self.measurements)

        # Computing the likelihood, according to Wikipedia's Kalman Filter > Marginal Likelihood equations and following their notations
        # https://en.wikipedia.org/wiki/Kalman_filter#Marginal_likelihood
        loglikelihood = 0
        for t, z in enumerate(self.measurements):
            Sk = self.model.H @ prior_covs[t] @ self.model.H.T + self.model.R
            yk = z - self.model.H @ prior_means[t]

            sign, logdet = np.linalg.slogdet(Sk)
            logdet *= sign
            loglikelihood -= 0.5 * (yk.T @ np.linalg.inv(Sk) @ yk + logdet + self.model.measurement_dim * np.log(2*np.pi))

        means = np.array(post_means)
        variances = np.array(list(map(lambda x: np.sqrt([x[i][i] for i in range(self.model.state_dim)]), post_covs)))
        
        if self.print_res:
            # Printing some results, highly customizable

            print("Loglikelihood: ", loglikelihood)

            fig = plt.figure()
            # plt.title('Kalman Filter Results', fontsize=16, y=1.08)
            
            measurements = np.array(self.measurements)
            times = range(len(self.measurements))
            for i in range(self.model.state_dim):
                ax = fig.add_subplot(self.model.state_dim, 1, i+1)
                # ax.set_title(f"dimension {i}")
                ax.plot(times, means[:, i], 'm--', color='red', label='KF means')

                # ax.plot(times, means[:, i] + variances[:, i], 'c--', label='KF variance')
                # ax.plot(times, means[:, i] - variances[:, i], 'c--')
                ax.fill_between(times, means[:, i] + variances[:, i], means[:, i] - variances[:, i], alpha=0.2)

                ax.plot(times, measurements[:, i], 'ko', label='measurements', markersize=3)
                ax.set_xlabel("timestep")
                ax.set_ylabel("value")
                ax.legend()

            fig.tight_layout()
            plt.savefig('KF_1D_run.pdf')

            plt.show()

            print("KF ran successfully")

        return means, variances, loglikelihood
        