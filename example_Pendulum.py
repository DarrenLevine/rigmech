from matplotlib import animation
import matplotlib.pyplot as plt
from rigmech import rigmech, np
import copy

# initialize the rigmech (with an optional name for the save file)
# and some simulation parameters
Pendulum = rigmech("Pendulum")
ForceRecompile = False
MotorCount = 3  # 1= single pendulum, 2= double pendulum, etc.
dt = 1. / 60.
substeps = 10
UseTorqueController = True

# if the save file couldn't load, the robot will be reconstructed
if ForceRecompile or not Pendulum.load():

    # Add a method of rotational movement (a joint)
    # as well as an attached mass (a link) relative to each joint
    for jnt in range(MotorCount):
        Pendulum.addJoint(name=f"motor{jnt}",
                          type="continuous",
                          parent=f"mass{jnt}" if jnt > 0 else None,
                          child=f"mass{jnt+1}",
                          axis_xyz=[0, 0, 1],
                          origin_xyz=[1 if jnt > 0 else 0, 0, 0],
                          friction=0.05)
        Pendulum.addLink(name=f"mass{jnt+1}",
                         origin_xyz=[1, 0, 0],
                         mass=1)
    Pendulum.generateEqns(Simplify=True)
    Pendulum.save()

# Optionally, customize the model parameters without re-running generateEqns()
Pendulum.global_syms["qFrict"] = [0.5 for _ in range(MotorCount)]


def getRobotLines(q, isJoint):
    ''' Use the xyz_com, and xyz_coj lambda functions to turn
    joint positional states (q) into x,y world frame positions'''
    xline = [0. for _ in range(MotorCount)]
    yline = [0. for _ in range(MotorCount)]
    for cnt, jnt in enumerate(Pendulum.Joints.keys()):
        if isJoint:  # coj = center of joint coordinate frame
            xyz = Pendulum.joint_syms[jnt]["func_xyz_coj"](*q)
        else:  # com = center of mass coordinate frame
            xyz = Pendulum.joint_syms[jnt]["func_xyz_com"](*q)
        xline[cnt] = xyz[0, 0]
        yline[cnt] = xyz[1, 0]
    return xline, yline


# set up the plot
fig, ax = plt.subplots()
linejnt, = ax.plot([0], [0], '-o', label="joint")
linemass, = ax.plot([0], [0], '--s', label="mass")
if UseTorqueController:
    linegoal, = ax.plot([0], [0], '--', label=None, color='r')
    linegoalx, = ax.plot([0], [0], 'o', label="goal", color='r')
ax.set(xlabel='X', ylabel='Y',
       xlim=[-MotorCount, MotorCount],
       ylim=[-MotorCount, MotorCount],
       title=f'{"Torque Controlled " if UseTorqueController else ""}Pendulum')
ax.set_aspect('equal')
plt.legend()

# initialize states
qForceJointsZero = [0. for _ in range(MotorCount)]  # tau
xyzrpyAccels = [0., -9.81, 0., 0., 0., 0.]  # x y z, wx, wy, wz
q = [-0.01 for _ in range(MotorCount)]  # joint angles in radians
q[0] += np.pi/2
dq = [0. for _ in range(MotorCount)]  # joint velocities in radians/sec
time_elapsed = 0
# pick a goal, with NaN's for the null space
Goal = np.array([0.5, 1., np.nan, np.nan, np.nan, -45*np.pi/180])
# pick how important each goal element is relative to the others
Importance = np.array([0.9, 0.9, 0, 0, 0, 1])


def CreateNewGoal(error, dq, dt):
    '''If the error is small, create a new goal, or
    if give up if it's taking too long'''
    global Goal, time_elapsed
    time_elapsed += dt
    error_threshold = 0.01
    too_long_threshold = 1.
    enorm = np.linalg.norm(error)
    if enorm < error_threshold or \
        time_elapsed > too_long_threshold or \
            np.linalg.norm(dq) < 0.01:
        if enorm < error_threshold:
            print(f"Reached goal in {time_elapsed} seconds.")
        else:
            print(f"Gave up, could not reach goal.")
        Goal[0] = (np.random.rand()-0.5)*2*0.8
        Goal[1] = (np.random.rand()-0.5)*2*0.8
        Goal[-1] = np.random.rand()*2*np.pi*2
        time_elapsed = 0


def Controller(dt, q, dq, ExtAccels):
    '''A very simple finite horizon controler (with no shooting
    optimization), where we pick an initial gain*Torque proportional in
    the direction of the solution, include some compensators for
    friction/momentum/etc. and then integrate the response over a finite
    horizon to adjust the final Torque applied to include compensation
    for the immediate frequency response, acheiving a self tuning PI
    like controler.'''
    global Goal

    # friction compensation
    qForceCompensators = np.array(dq)*np.array(Pendulum.global_syms["qFrict"])
    # momentum compensation
    qForceCompensators -= np.dot(Pendulum.global_syms["func_Mq"](*q), dq)/dt
    # gravity compensation
    qForceCompensators -= Pendulum.global_syms[
        "func_qFext"](*q, *ExtAccels).T[0, :]

    def limitTorque(Torque):
        max_t = 5E3  # limit the abs torque to this value
        Torque[Torque > max_t] = max_t
        Torque[Torque < -max_t] = -max_t

    limitTorque(qForceCompensators)

    # craft an error function
    LastJointName = next(reversed(Pendulum.Joints.keys()))
    fJ = Pendulum.joint_syms[LastJointName]["func_J_com"]
    fMq = Pendulum.global_syms["func_Mq"]
    fxyz = Pendulum.joint_syms[LastJointName]["func_xyz_com"]
    fwxyz = Pendulum.joint_syms[LastJointName]["func_Wxyz_com"]
    ctrl_dof = ~np.isnan(Goal)
    GoalArray = np.array([Goal]).T
    GoalArray[np.isnan(GoalArray)] = 0
    Imptc = np.array([Importance[ctrl_dof]]).T
    Imptc = Imptc/np.linalg.norm(Imptc)  # normalize

    def cartesianError_func(_q):
        Wxyz_ee = fwxyz(*_q)
        cartesianErr = GoalArray - np.concatenate((fxyz(*_q), Wxyz_ee))
        cartesianErr[3:] = rigmech.QuatAngleDiff(Wxyz_ee, GoalArray[3:])
        cartesianErr = np.multiply(cartesianErr[ctrl_dof], Imptc)
        return cartesianErr

    def qError_func(_q):
        cartesianErr = cartesianError_func(_q)
        # inertia matrix in task space (Mx) is used to
        # translate the cartesian error vector back into
        # joint space error (qError) while compensating
        # for mass
        J = fJ(*_q)[ctrl_dof]
        Mx_inv = np.dot(J, np.dot(np.linalg.inv(fMq(*_q)), J.T))
        Mx_dot_cErr = np.linalg.lstsq(Mx_inv, cartesianErr, rcond=.005)[0]
        qError = np.dot(J.T, Mx_dot_cErr).T[0]
        return qError

    # integrate the error along a future limited horizon
    gain = 80/dt
    qControlTorque = qError_func(q)*gain
    horizon_size = 3
    if horizon_size > 0:
        qcopy = copy.deepcopy(q)
        dqcopy = copy.deepcopy(dq)
        for _ in range(horizon_size):
            qForceJoints = qForceCompensators + qControlTorque
            qcopy, dqcopy, _ = Pendulum.ForwardDynamics(
                dt/horizon_size, qcopy, dqcopy, qForceJoints, ExtAccels)
            # adjust conrol torque component
            qControlTorque = (qControlTorque + qError_func(qcopy)*gain)/2

    CreateNewGoal(cartesianError_func(q), dq, dt)
    qForceJoints = qForceCompensators + qControlTorque
    limitTorque(qForceJoints)
    return qForceJoints


def update_plot(num):
    '''for each re-draw step, calculate the ForwardDynamics for
    x substeps and then update the plot lines'''
    global q, dq, linemass, linejnt, linegoal, linegoalx, Goal

    # optionally apply a joint force (not updated in substeps
    # loop, to simulate a lower bandwidth controller of dt)
    if UseTorqueController:
        qForceJoints = Controller(dt, q, dq, xyzrpyAccels)
        # add some noise to make things harder for the controller
        qForceJoints += np.random.randn(qForceJoints.size)
    else:
        qForceJoints = qForceJointsZero

    # simulate
    for _ in range(substeps):
        q, dq, _ = Pendulum.ForwardDynamics(
            dt/substeps, q, dq, qForceJoints, xyzrpyAccels)

    # update plot
    linejnt.set_data(getRobotLines(q, True))
    linemass.set_data(getRobotLines(q, False))
    if UseTorqueController:
        gx = Goal[0]
        gy = Goal[1]
        cr = np.cos(Goal[-1])*0.45
        sr = np.sin(Goal[-1])*0.45
        linegoal.set_data(([gx - cr, gx], [gy - sr, gy]))
        linegoalx.set_data(([gx], [gy]))
        return linejnt, linemass, linegoal, linegoalx
    return linejnt, linemass


# start the animation
SaveAsGif = False
anim = animation.FuncAnimation(fig, update_plot, interval=dt, blit=True,
                               frames=100, repeat=~SaveAsGif)

if SaveAsGif:
    anim.save('animation.gif', writer='imagemagick', fps=int(1/dt))
else:
    plt.show()
