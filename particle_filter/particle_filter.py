import numpy as np
import matplotlib.pyplot as plt
import copy

class World:

    def __init__(self, size_x, size_y, landmarks):
        """
        Initialize world with given dimensions.
        :param size_x: Length world in x-direction (m)
        :param size_y: Length world in y-direction (m)
        :param landmarks: List with 2D-positions of landmarks
        """

        self.x_max = size_x
        self.y_max = size_y

        # Check if each element is a list
        if any(not isinstance(lm, list) for lm in landmarks):

           
            if len(landmarks) != 2:
                print("Invalid landmarks provided to World: {}".format(landmarks))
            else:
                self.landmarks = [landmarks]
        else:
            
            if any(len(lm) != 2 for lm in landmarks):
                print("Invalid landmarks provided to World: {}".format(landmarks))
            else:
                self.landmarks = landmarks


class Visualizer:
    """
    Class for visualing the world, the true robot pose and the discrete distribution that estimates the robot pose by
    means of a set of (weighted) particles.
    """

    def __init__(self, draw_particle_pose=False):
        """
        Initialize visualizer. By setting the flag to false the full 2D pose will be visualized. This makes
        visualization much slower hence is only recommended for a relatively low number of particles.

        :param draw_particle_pose: Flag that determines whether 2D positions (default) or poses must be visualized.
        """

        self.x_margin = 1
        self.y_margin = 1
        self.circle_radius_robot = 0.2  # 0.25
        self.draw_particle_pose = draw_particle_pose
        self.landmark_size = 7
        self.scale = 2  # meter / inch
        self.robot_arrow_length = 0.2  #0.5 / self.scale

    def draw_world(self, world, robot, particles, hold_on=False, particle_color='g'):
        """
        Draw the simulated world with its landmarks, the robot 2D pose and the particles that represent the discrete
        probability distribution that estimates the robot pose.

        :param world: World object (includes dimensions and landmarks)
        :param robot: True robot 2D pose (x, y, heading)
        :param particles: Set of weighted particles (list of [weight, [x, y, heading]]-lists)
        :param hold_on: Boolean that indicates whether figure must be cleared or nog
        :param particle_color: Color used for particles (as matplotlib string)
        """

        # Dimensions world
        x_min = -self.x_margin
        x_max = self.x_margin + world.x_max
        y_min = -self.y_margin
        y_max = self.y_margin + world.y_max

        # Draw world
        plt.figure(1, figsize=((x_max-x_min) / self.scale, (y_max-y_min) / self.scale))
        if not hold_on:
            plt.clf()
        plt.plot([0, world.x_max], [0, 0], 'k-')                      # lower line
        plt.plot([0, 0], [0, world.y_max], 'k-')                      # left line
        plt.plot([0, world.x_max], [world.y_max, world.y_max], 'k-')  # top line
        plt.plot([world.x_max, world.x_max], [0, world.y_max], 'k-')  # right line

        # Set limits axes
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        # No ticks on axes
        plt.xticks([])
        plt.yticks([])

        # Title
        plt.title("{} particles".format(len(particles)))

        # Add landmarks
        landmarks = np.array(world.landmarks)
        plt.plot(landmarks[:, 0], landmarks[:, 1], 'bs', linewidth=2, markersize=self.landmark_size)

        # Add particles
        if self.draw_particle_pose:
            # Warning: this is very slow for large numbers of particles
            radius_scale_factor = len(particles) / 10.
            for p in particles:
                self.add_pose2d(p[1][0], p[1][1], p[1][2], 1, particle_color, radius_scale_factor * p[0])
        else:
            # Convert to numpy array for efficiency reasons (drop weights)
            states = np.array([np.array(state_i[1]) for state_i in particles])
            plt.plot(states[:, 0], states[:, 1], particle_color+'.', linewidth=1, markersize=2)

        # Add robot pose
        self.add_pose2d(robot.x, robot.y, robot.theta, 1, 'r', self.circle_radius_robot)


        # Show
        plt.pause(0.05)

    def add_pose2d(self, x, y, theta, fig_num, color, radius):
        """
        Plot a 2D pose in given figure with given color and radius (circle with line indicating heading).

        :param x: X-position (center circle).
        :param y: Y-position (center circle).
        :param theta: Heading (line from center circle with this heading will be added).
        :param fig_num: Figure in which pose must be added.
        :param color: Color of the lines.
        :param radius: Radius of the circle.
        :return:
        """

        # Select correct figure
        plt.figure(fig_num)

        # Draw circle at given position (higher 'zorder' value means draw later, hence on top of other lines)
        circle = plt.Circle((x, y), radius, facecolor=color, edgecolor=color, alpha=0.4, zorder=20)
        plt.gca().add_patch(circle)

        # Draw line indicating heading
        plt.plot([x, x + radius * np.cos(theta)],
                 [y, y + radius * np.sin(theta)], color)


class Robot:

    def __init__(self, x, y, theta):
        """
        Initialize the robot with given 2D pose. In addition set motion uncertainty parameters.

        :param x: Initial robot x-position (m)
        :param y: Initial robot y-position (m)
        :param theta: Initial robot heading (rad)
        :param std_forward: Standard deviation zero mean additive noise on forward motions (m)
        :param std_turn: Standard deviation zero mean Gaussian additive noise on turn actions (rad)
        :param std_meas_distance: Standard deviation zero mean Gaussian additive measurement noise (m)
        :param std_meas_angle: Standard deviation zero mean Gaussian additive measurement noise (rad)
        """

        # Initialize robot pose
        self.x = x
        self.y = y
        self.theta = theta

        # Set standard deviations noise robot motion
        self.std_forward = .005
        self.std_turn = .002

        # Set standard deviation measurements
        self.std_meas_distance = .2
        self.std_meas_angle = .05

    def move(self, desired_distance, desired_rotation, world):
        # Compute relative motion (true motion is desired motion with some noise)
        distance_driven = self._get_gaussian_noise_sample(desired_distance, self.std_forward)
        angle_rotated = self._get_gaussian_noise_sample(desired_rotation, self.std_turn)

        # Update robot pose
        self.theta += angle_rotated
        self.x += distance_driven * np.cos(self.theta)
        self.y += distance_driven * np.sin(self.theta)

        # Cyclic world assumption (i.e. crossing right edge -> enter on left hand side)
        self.x = np.mod(self.x, world.x_max)
        self.y = np.mod(self.y, world.y_max)

        # Angles in [0, 2*pi]
        self.theta = np.mod(self.theta, 2*np.pi)

    def measure(self, world):
        """
        Perform a measurement. The robot is assumed to measure the distance to and angles with respect to all landmarks
        in meters and radians respectively. While doing so, the robot experiences zero mean additive Gaussian noise.

        :param world: World containing the landmark positions.
        :return: List of lists: [[dist_to_landmark1, angle_wrt_landmark1], dist_to_landmark2, angle_wrt_landmark2], ...]
        """

        # Loop over measurements
        measurements = []
        for lm in world.landmarks:
            dx = self.x - lm[0]
            dy = self.y - lm[1]

            # Measured distance perturbed by zero mean additive Gaussian noise
            z_distance = self._get_gaussian_noise_sample(np.sqrt(dx * dx + dy * dy), self.std_meas_distance)

            # Measured angle perturbed by zero mean additive Gaussian noise
            z_angle = self._get_gaussian_noise_sample(np.arctan2(dy, dx), self.std_meas_angle)

            # Store measurement
            measurements.append([z_distance, z_angle])

        return measurements

    @staticmethod
    def _get_gaussian_noise_sample(mu, sigma):
        """
        Get a random sample from a 1D Gaussian distribution with mean mu and standard deviation sigma.

        :param mu: mean of distribution
        :param sigma: standard deviation
        :return: random sample from distribution with given parameters
        """
        return np.random.normal(loc=mu, scale=sigma, size=1)[0]

class Resampler:
    """
    Resample class that implements different resampling methods.
    """

    def __init__(self):
        self.initialized = True


    def cumulative_sum(self, weights):
    
        return np.cumsum(weights).tolist()
    
    def naive_search(self, cumulative_list, x):
    
        m = 0
        while cumulative_list[m] < x:
             m += 1
        return m
    
    def resample(self, samples, N):
        """
        Particles are sampled with replacement proportional to their weight and in arbitrary order. This leads
        to a maximum variance on the number of times a particle will be resampled, since any particle will be
        resampled between 0 and N times.

        Computational complexity: O(N log(M)

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """

        # Get list with only weights
        weights = [weighted_sample[0] for weighted_sample in samples]

        # Compute cumulative sum
        Q = self.cumulative_sum(weights)

        # As long as the number of new samples is insufficient
        n = 0
        new_samples = []
        while n < N:

            # Draw a random sample u
            u = np.random.uniform(1e-6, 1, 1)[0]

            # Naive search (alternative: binary search)
            m = self.naive_search(Q, u)

            # Add copy of the state sample (uniform weights)
            new_samples.append([1.0/N, copy.deepcopy(samples[m][1])])

            # Added another sample
            n += 1

        return new_samples



class ParticleFilter:
    
    def __init__ (self, Ns, xylimits):
        self.Ns = Ns
        self.particles = []
        self.state_dimension = 3  #(x,y,theta)
        self.x_max = xylimits [0]
        self.y_max = xylimits [1]
        self.process_noise  = [.1, .2]
        self.measurement_noise = [.4, .3]
        self.resampler = Resampler()
        

    def init_particles_uniform(self):
        weight = 1 / self.Ns
        self.particles = [[weight, [  np.random.uniform(0, self.x_max, 1)[0],np.random.uniform(0, self.y_max, 1)[0], np.random.uniform(0, 2 * np.pi, 1)[0]]] for i in range (self.Ns) ]
        
    def propagate_sample(self, sample, forward_motion, angular_motion):
        """
        Propagate an individual sample with a simple motion model that assumes the robot rotates angular_motion rad and
        then moves forward_motion meters in the direction of its heading. Return the propagated sample (leave input
        unchanged).

        :param sample: Sample (unweighted particle) that must be propagated
        :param forward_motion: Forward motion in meters
        :param angular_motion: Angular motion in radians
        :return: propagated sample
        """
        # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
        propagated_sample = copy.deepcopy(sample)
        propagated_sample[2] += np.random.normal(angular_motion, self.process_noise[1], 1)[0]

        # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
        forward_displacement = np.random.normal(forward_motion, self.process_noise[0], 1)[0]

        # 2. move forward
        propagated_sample[0] += forward_displacement * np.cos(propagated_sample[2])
        propagated_sample[1] += forward_displacement * np.sin(propagated_sample[2])

        # Make sure we stay within cyclic world
        return propagated_sample


    def compute_likelihood(self, sample, measurement, landmarks):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: List with measurements, for each landmark [distance_to_landmark, angle_wrt_landmark], units
        are meters and radians
        :param landmarks: Positions (absolute) landmarks (in meters)
        :return Likelihood
        """

        # Initialize measurement likelihood
        likelihood_sample = 1.0

        # Loop over all landmarks for current particle
        for i, lm in enumerate(landmarks):

            # Compute expected measurement assuming the current particle state
            dx = sample[0] - lm[0]
            dy = sample[1] - lm[1]
            expected_distance = np.sqrt(dx*dx + dy*dy)
            expected_angle = np.arctan2(dy, dx)

            # Map difference true and expected distance measurement to probability
            p_z_given_x_distance = \
                np.exp(-(expected_distance-measurement[i][0]) * (expected_distance-measurement[i][0]) /
                       (2 * self.measurement_noise[0] * self.measurement_noise[0]))

            # Map difference true and expected angle measurement to probability
            p_z_given_x_angle = \
                np.exp(-(expected_angle-measurement[i][1]) * (expected_angle-measurement[i][1]) /
                       (2 * self.measurement_noise[1] * self.measurement_noise[1]))

            # Incorporate likelihoods current landmark
            likelihood_sample *= p_z_given_x_distance * p_z_given_x_angle

        # Return importance weight based on all landmarks
        return likelihood_sample


    def normalize_weights(self, weighted_samples):
        """
        Normalize all particle weights.
        """

        # Compute sum weighted samples
        sum_weights = 0.0
        for weighted_sample in weighted_samples:
            sum_weights += weighted_sample[0]

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

            # Set uniform weights
            return [[1.0 / len(weighted_samples), weighted_sample[1]] for weighted_sample in weighted_samples]

        # Return normalized weights
        return [[weighted_sample[0] / sum_weights, weighted_sample[1]] for weighted_sample in weighted_samples]



    def update(self, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Loop over all particles
        new_particles = []
        for par in self.particles:

            # Propagate the particle state according to the current particle
            propagated_state = self.propagate_sample(par[1], .25, .02)

            # Compute current particle's weight
            weight = par[0] * self.compute_likelihood(propagated_state, measurements, landmarks)

            # Store
            new_particles.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        
        self.particles = self.resampler.resample(self.particles, self.Ns)


steps = 30
N_particles = 1000

world = World(10.0, 10.0, [[2.0, 2.0], [2.0, 8.0], [9.0, 2.0], [8, 9]])
visual = Visualizer()
robot = Robot(world.x_max * .75, world.y_max / 5, 3.14/2)
pFilter = ParticleFilter(N_particles, (world.x_max, world.y_max))
pFilter.init_particles_uniform()

#print (pFilter.particles)

for i in range(steps):

        # Simulate robot motion (required motion will not exactly be achieved)
    robot.move(desired_distance= .25,
                   desired_rotation=.02,
                   world=world)

        # Simulate measurement
    measurements = robot.measure(world)

        # Update SIR particle filter
    pFilter.update(  measurements=measurements, landmarks=world.landmarks)

        # Visualization
    visual.draw_world(world, robot, pFilter.particles, hold_on=False, particle_color='g')
    plt.pause(0.05)



