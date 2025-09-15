from franky import *

robot = Robot("10.1.38.23")  # Replace this with your robot's IP

# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
robot.relative_dynamics_factor = 0.05

# A Cartesian velocity motion with linear (first argument) and angular (second argument)
# components
m_cv1 = CartesianVelocityMotion(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]))

# With target elbow velocity
m_cv2 = CartesianVelocityMotion(
    RobotVelocity(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]), elbow_velocity=-0.2)
)

# Cartesian velocity motions also support multiple waypoints. Unlike in Cartesian position
# control, a Cartesian velocity waypoint is a target velocity to be reached. This particular
# example first accelerates the end-effector, holds the velocity for 1s, then reverses
# direction for 2s, reverses direction again for 1s, and finally stops. It is important not to
# forget to stop the robot at the end of such a sequence, as it will otherwise throw an error.
m_cv4 = CartesianVelocityWaypointMotion(
    [
        CartesianVelocityWaypoint(
            Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
            hold_target_duration=Duration(1000),
        ),
        CartesianVelocityWaypoint(
            Twist([-0.2, 0.1, -0.1], [-0.1, 0.1, -0.2]),
            hold_target_duration=Duration(2000),
        ),
        CartesianVelocityWaypoint(
            Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
            hold_target_duration=Duration(1000),
        ),
        CartesianVelocityWaypoint(Twist()),
    ]
)

# Stop the robot in Cartesian velocity control mode.
m_cv6 = CartesianVelocityStopMotion()

# robot.move(m_cv1)
# robot.move(m_cv2)
robot.move(m_cv4)
# robot.move(m_cv6)