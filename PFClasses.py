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

class SimpleModel:
    def __init__(self, x_0, F, H, Q, R):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.states = [x_0]
        self.measurements = [np.matmul(self.H, self.states[0]) + multivariate_normal(mean=[0] * len(self.R), cov=self.R)]
        self.state_dim = len(Q)
        self.measurement_dim = len(R)
                
    def step(self, particle):
        step_noise = lambda : multivariate_normal(mean=np.zeros(len(self.Q)), cov=self.Q)
        return np.matmul(self.F, particle) + step_noise()
    
    def measure(self, t):
        if t < len(self.measurements):
            return self.measurements[t]
        else:
            new_state = np.matmul(self.F, self.states[-1]) + multivariate_normal(mean=[0] * len(self.Q), cov=self.Q)
            new_measurement = np.matmul(self.H, new_state) + multivariate_normal(mean=[0] * len(self.R), cov=self.R)
            self.states.append(new_state)
            self.measurements.append(new_measurement)
            return self.measure(t)
    
    def eval(self, particle):
        return np.matmul(self.H, particle)


class Event:
    def __init__(self, rate_calculator, change_function):
        self.rate_calculator = rate_calculator
        self.change_function = change_function


class StochasticModelTauLeaping:
    def __init__(self, x_0, events, step_size, R):
        self.events = events
        self.x_0 = x_0
        self.step_size = step_size
        self.R = R
        self.states = [x_0]
        self.measurements = [x_0 + multivariate_normal(mean=[0] * len(self.R), cov=self.R)]
        self.state_dim = len(R)

    def step(self, particle):
        for e in self.events:
            Kj = np.random.poisson(self.step_size * e.rate_calculator(particle))
            for _ in range(Kj):
                particle = e.change_function(particle)
        return particle

    def measure(self, t):
        if t < len(self.measurements):
            return self.measurements[t]
        else:
            diff = t - len(self.measurements) + 1
            for _ in range(diff):
                new_state = self.step(self.states[-1])
                self.states.append(new_state)
                new_measurement = new_state + multivariate_normal(mean=[0] * len(self.R), cov=self.R)
                self.measurements.append(new_measurement)
            return self.measure(t)

    def eval(self, particle):
        return particle
    
    
class ParticleFilter:
    def __init__(self, model, N, measurements, neff_threshold=1/3, resampling_algo="stratified_resample", prior_cov_matrix=None, ranges=None, print_results=False):
        self.model = model
        self.N = N
        self.measurements = measurements
        self.neff_threshold = neff_threshold

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
        
        
        self.weights = np.ones(N) / N
        self.prior_cov_matrix = prior_cov_matrix

        self.used = False
        self.print_results = print_results
        
    def plot_particles(self):
        fig, axes = plt.subplots(nrows=self.model.state_dim)
        if self.model.state_dim == 1:
            axes = [axes]
        for i in range(self.model.state_dim):
            sns.distplot(self.particles[:, i], ax=axes[i])    
        
    def uniform_particles(self, N, model, ranges):
        particles = np.empty((N, model.state_dim))
        for i, (start, end) in enumerate(ranges):
            particles[:, i] = np.random.uniform(low=start, high=end, size=N)
        return particles
    
    def normal_particles(self, N, model, x_0, prior_cov_matrix):
        particles = np.empty((N, model.state_dim))
        for i in range(model.state_dim):
            particles[:, i] = np.random.normal(loc=x_0[i], scale=prior_cov_matrix[i][i], size=N)
        return particles
    
    def predict(self, model, particles):
        return np.array(list(map(model.step, particles)))
    
    def update(self, z, model, particles, weights): 
        observations = np.array(list(map(model.eval, particles)))
        
        weights *= scipy.stats.multivariate_normal.pdf(x=observations, mean=z, cov=self.model.R) 
        weights += 1.e-100 # avoid round off to zero
        weights /= np.sum(weights) # normalize

        return weights
    
    def estimate(self, particles, weights):
        mean = np.average(particles, weights=weights, axis=0)
        var = np.sqrt(np.average((particles - mean)**2, weights=weights, axis=0))
        return mean, var
    
    def resample(self, particles, weights):
        indexes = self.resampling_algo(particles, weights)
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights.fill(1.0/len(particles))
        return particles, weights
    
    def neff(self, weights):
        return 1. / np.sum(np.square(weights))
        
    def run(self):
        assert not self.used, "PF already ran. Please instantiate a new PF"
        self.used = True

        time = len(self.measurements)

        means = []
        variances = []

        likelihood = 0
        p_ykxk = lambda i: scipy.stats.multivariate_normal.pdf(self.model.states[0], mean=self.model.eval(self.particles[i]), cov=self.model.R) # p(yk|xki)
        Pyk = np.sum([p_ykxk(i) * self.weights[i] for i in range(self.N)]) # Phat(y_k|y_0:k-1)
        
        if Pyk == 0: #np warnings are annoying
            likelihood -= np.inf
        else:
            likelihood += np.log(Pyk)


        
        #print(f"Running PF with {self.N} particles: ")#, end="")
        initial_mean, initial_variance = self.estimate(self.particles, self.weights)
        means.append(initial_mean)
        variances.append(initial_variance)

        for i in range(time-1):
            z = self.measurements[i+1]

            self.particles = self.predict(self.model, self.particles)           

            p_ykxk = lambda i: scipy.stats.multivariate_normal.pdf(z, mean=self.model.eval(self.particles[i]), cov=self.model.R) # p(yk|xki)
            Pyk = np.sum([p_ykxk(i) * self.weights[i] for i in range(self.N)]) # Phat(y_k|y_0:k-1)
            if Pyk == 0:
                likelihood -= np.inf
            else:
                likelihood += np.log(Pyk)

            self.weights = self.update(z, self.model, self.particles, self.weights)

            mean, var = self.estimate(self.particles, self.weights)

            means.append(mean)
            variances.append(var)            
            
            if self.neff(self.weights) < len(self.particles) * self.neff_threshold:
                self.particles, self.weights = self.resample(self.particles, self.weights)
            else:
                pass
        pass#print(" end")

        if self.print_results:
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
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, N))
        return indexes

    def residual_resample(self, particles, weights):
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
    def __init__(self, model, measurements, print_results=False):
        self.kf = KF(dim_x=model.state_dim, dim_z=model.measurement_dim)
        self.measurements = measurements

        self.kf.x = model.states[0]
        self.kf.F = model.F
        self.kf.H = model.H
        self.kf.Q = model.Q
        self.kf.R = model.R

        self.model = model
        self.print_res = print_results
    
    def run(self):
        post_means, post_covs, prior_means, prior_covs = self.kf.batch_filter(self.measurements)
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
        