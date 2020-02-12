import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import scipy
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (17, 7)


class SimpleModel:
    def __init__(self, x_0, F, H, Q, R):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        
        self.states = [x_0]
        self.measurements = [np.matmul(self.H, self.states[0]) + np.random.multivariate_normal(mean=[0] * len(self.R), cov=self.R)]
                
        self.state_dim = len(Q)
        self.measurement_dim = len(R)
        
        self.step_error_covariance = Q
        self.measurement_error_covariance = R
        
    def step(self, particle):
        # x_n+1 = F*x_n + N(0, Q)
        step_noise = lambda : np.random.multivariate_normal(mean=np.zeros(len(self.Q)), cov=self.Q)
        return np.matmul(self.F, particle) + step_noise()
    
    def measure(self, t):
        if t < len(self.measurements):
            return self.measurements[t]
        else:
            new_state = np.matmul(self.F, self.states[-1]) + \
                            + np.random.multivariate_normal(mean=[0] * len(self.Q), cov=self.Q)
            new_measurement = np.matmul(self.H, new_state) + np.random.multivariate_normal(mean=[0] * len(self.R), cov=self.R)
            self.states.append(new_state)
            self.measurements.append(new_measurement)
            return self.measure(t)
    
    def eval(self, particle):
        return particle


class Event:
    def __init__(self, rate_calculator, change_function):
        self.rate_calculator = rate_calculator
        self.change_function = change_function


class StochasticModelTauLeaping:
    def __init__(self, x_0, events, step_size):
        self.events = events
        self.x_0 = x_0
        self.step_size = step_size

        self.states = [x_0]

    def step(self, particle):
        for e in self.events:
            Kj = np.random.poisson(self.step_size * e.rate_calculator(particle))
            for _ in range(Kj):
                particle = e.change_function(particle)
        return particle

    def measure(self, t):
        if t < len(self.states):
            return self.states[t]
        else:
            diff = t - len(self.states) + 1
            for _ in range(diff):
                new_state = self.step(self.states[-1])
                self.states.append(new_state)
            return self.measure(t)

    def eval(self, particle):
        return particle
    
    
class ParticleFilter:
    def __init__(self, model, N, resampling_algo="simple_resample", x_0=None, ranges=None, print_results=True):
        self.model = model
        self.N = N

        if resampling_algo == "simple_resample":
            self.resampling_algo = self.simple_resample
        else:
            assert False, f"Resampling algorithm {resampling_algo} unknown"
        
        
        if x_0 is None and ranges is None:
            assert False, "Please input either a starting point (for normally distributed particles)" \
                            "or some ranges (for uniformly distributed particles)"
        elif ranges is None:
            self.particles = self.normal_particles(N, model, x_0)
        else:
            assert len(ranges) == model.state_dim, "The number of ranges differs from the state dimension"
            self.particles = self.uniform_particles(N, model, ranges)
        
        self.weights = np.ones(N) / N
        self.used = False
        self.print_results = print_results
        
    def plot_particles(self):
        fig, axes = plt.subplots(nrows=self.model.state_dim)
        if self.model.state_dim == 1:
            axes = [axes]
        for i in range(self.model.state_dim):
            sns.distplot(self.particles[:, i], ax=axes[i], axlabel=f"dimension {i}")        
        
    def uniform_particles(self, N, model, ranges):
        particles = np.empty((N, model.state_dim))
        for i, (start, end) in enumerate(ranges):
            particles[:, i] = np.random.uniform(low=start, high=end, size=N)
        return particles
    
    def normal_particles(self, N, model, x_0):
        particles = np.empty((N, model.state_dim))
        for i in range(model.state_dim):
            particles[:, i] = np.random.normal(loc=x_0[i], scale=model.step_error_covariance[i][i], size=N)
        return particles
    
    def predict(self, model, particles):
        return np.array(list(map(model.step, self.particles)))
    
    def update(self, z, model, particles, weights): 
        observations = np.array(list(map(model.eval, particles)))
        
        weights = weights * scipy.stats.multivariate_normal.pdf(x=observations, mean=z, cov=model.measurement_error_covariance) 
        weights += 1.e-100 # avoid round off to zero
        weights /= np.sum(weights) # normalize

        return weights
    
    def estimate(self, particles, weights):
        mean = np.average(particles, weights=weights, axis=0)
        var = np.sqrt(np.average((particles - mean)**2, weights=weights, axis=0))
        return mean, var
    
    def resample(self, particles, weights):
        return self.resampling_algo(particles, weights)
    
    def neff(self, weights):
        return 1. / np.sum(np.square(weights))
        
    def run(self, time):
        assert not self.used, "PF already ran. Please instantiate a new PF"
        self.used = True

        measurements = []
        means = []
        variances = []
        likelihood = np.log(1) # log( p(y_0)??? )
        
        print(f"Running PF with {self.N} particles: ", end="")
        initial_mean, initial_variance = self.estimate(self.particles, self.weights)
        means.append(initial_mean)
        variances.append(initial_variance)
        measurements.append(self.model.states[0])

        for i in range(time-1):
            z = self.model.measure(i+1)

            self.particles = self.predict(self.model, self.particles)
            self.weights = self.update(z, self.model, self.particles, self.weights)
            mean, var = self.estimate(self.particles, self.weights)

            means.append(mean)
            variances.append(var)
            measurements.append(z)
            
            p_ykxk = lambda i: scipy.stats.multivariate_normal.pdf(z, mean=self.model.eval(self.particles[i]), cov=self.model.measurement_error_covariance) # p(yk|xki)
            Pyk = np.sum([p_ykxk(i) * self.weights[i] for i in range(self.N)]) # Phat(y_k|y_0:k-1)
            likelihood += np.log(Pyk)
            
            if self.neff(self.weights) < len(self.particles)/3:
                print("R", end="")
                self.particles, self.weights = self.resample(self.particles, self.weights)
            else:
                print(".", end="")
        print(" end")

        if self.print_results:
            print("Marginal likelihood: ", round(likelihood, 4))

            fig = plt.figure()
            plt.title('Particle Filter Results', fontsize=16, y=1.08)
            means = np.array(means)
            variances = np.array(variances)
            measurements = np.array(measurements)
            times = range(time)
            for i in range(self.model.state_dim):
                ax = fig.add_subplot(self.model.state_dim, 1, i+1)
                ax.set_title(f"dimension {i}")
                ax.plot(times, means[:, i], 'b--', label='PF means')
                ax.plot(times, means[:, i] + variances[:, i], 'c--', label='PF variance')
                ax.plot(times, means[:, i] - variances[:, i], 'c--')
                ax.plot(times, measurements[:, i], 'bo', label='measurements')
                ax.set_xlabel("timestep")
                ax.legend()

            fig.tight_layout()
            plt.show()

        return means, variances, likelihood
    
    def simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, N))
        return particles[indexes], np.ones(N)/N


from filterpy.kalman import KalmanFilter as KF
class Kalman:
    def __init__(self, model, print_results):
        self.kf = KF(dim_x=model.state_dim, dim_z=model.measurement_dim)
        self.kf.x = model.states[0]
        self.kf.F = model.F
        self.kf.H = model.H
        self.kf.Q = model.Q
        self.kf.R = model.R
        self.model = model
        self.print_res = print_results
    
    def run(self, steps):
        measurements = self.model.measurements[:steps]
        post_means, post_covs, prior_means, prior_covs = self.kf.batch_filter(measurements)
        loglikelihood = 0
        for t, z in enumerate(measurements):
            Sk = self.model.H @ prior_covs[t] @ self.model.H.T + self.model.R
            yk = z - self.model.H @ prior_means[t]
            loglikelihood -= 0.5 * (yk.T @ np.linalg.inv(Sk) @ yk + np.log(np.linalg.det(Sk)) + 2 * np.log(2*np.pi))

        means = np.array(post_means)
        variances = np.array(list(map(lambda x: np.sqrt([x[i][i] for i in range(self.model.state_dim)]), post_covs)))
        
        if self.print_res:
            print("Loglikelihood: ", loglikelihood)

            fig = plt.figure()
            plt.title('Kalman Filter Results', fontsize=16, y=1.08)
            
            measurements = np.array(measurements)
            times = range(len(measurements))
            for i in range(self.model.state_dim):
                ax = fig.add_subplot(self.model.state_dim, 1, i+1)
                ax.set_title(f"dimension {i}")
                ax.plot(times, means[:, i], 'b--', label='KF means')
                ax.plot(times, means[:, i] + variances[:, i], 'c--', label='KF variance')
                ax.plot(times, means[:, i] - variances[:, i], 'c--')
                ax.plot(times, measurements[:, i], 'bo', label='measurements')
                ax.set_xlabel("timestep")
                ax.legend()

            fig.tight_layout()
            plt.show()

        return means, variances
        