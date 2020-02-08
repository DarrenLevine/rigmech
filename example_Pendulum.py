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
    linegoalx, = ax.plot([0], [0], 'o', label=None, color='r')
    ax.plot([0], [0], '--o', label="goal", color='r')
ax.set(xlabel='X', ylabel='Y', xlim=[-3, 3],
       ylim=[-3, 3], title='Pendulum')
ax.set_aspect('equal')
plt.legend()

# initialize state
qForceJointsZero = [0. for _ in range(MotorCount)]  # tau
xyzrpyAccels = [0., -9.81, 0., 0., 0., 0.]  # x y z, wx, wy, wz
q = [1.01 for _ in range(MotorCount)]  # joint angles in radians
dq = [0. for _ in range(MotorCount)]  # joint velocities in radians/sec
stepcount = 0
# pick a goal, with NaN's for the null space
Goal = np.array([0.5, 1., np.nan, np.nan, np.nan, -45*np.pi/180])


def CreateNewGoal(enorm):
    '''If the error is small, create a new goal, or
    if give up if it's taking too long'''
    global Goal, stepcount
    stepcount += 1
    if enorm < 0.01 or stepcount > 200:
        if enorm < 0.01:
            print(f"Reached goal in {stepcount} steps.")
        else:
            print(f"Gave up, could not reach goal.")
        Goal[0] = (np.random.rand()-0.5)*2*1
        Goal[1] = (np.random.rand()-0.5)*2*1
        Goal[-1] = (np.random.rand()-0.5)*2*2*np.pi
        stepcount = 0


def Controller(dt, q, dq, ExtAccels):
    '''A very simple finite horizon controler, where we pick an initial
    gain*Torque that decays in the direction of the solution, and then
    integrate the response over a finite horizon to adjust the Torque
    to include the frequency adjusted response, acheiving a self tuning
    PI like controler.'''
    global Goal

    # friction compensation
    qForceJoints = np.array(dq)*np.array(Pendulum.global_syms["qFrict"])
    # momentum compensation
    qForceJoints -= np.dot(Pendulum.global_syms["func_Mq"](*q), dq)/dt
    # gravity compensation
    qForceJoints += - Pendulum.global_syms[
        "func_qFext"](*q, *ExtAccels).T[0, :]

    # craft an error function
    LastJointName = next(reversed(Pendulum.Joints.keys()))
    fJ = Pendulum.joint_syms[LastJointName]["func_J_com"]
    fxyz = Pendulum.joint_syms[LastJointName]["func_xyz_com"]
    fwxyz = Pendulum.joint_syms[LastJointName]["func_Wxyz_com"]
    ctrl_dof = ~np.isnan(Goal)
    targ = np.array([Goal[ctrl_dof]]).T

    def qError_func(_q):
        J = fJ(*_q)[ctrl_dof]
        cartesianErr = targ - np.concatenate((fxyz(*_q), fwxyz(*_q)))[ctrl_dof]
        qError = np.linalg.lstsq(J, cartesianErr, rcond=0.005)[0].T[0]
        normErr = sum(np.square(qError))**0.5
        return qError, normErr

    # integrate the error along a future limited horizon
    gain = 10/dt
    error, enorm = qError_func(q)
    qForceJoints += error*gain/(enorm**0.25)
    qcopy = copy.deepcopy(q)
    dqcopy = copy.deepcopy(dq)
    horizon_size = 3
    # note: ((h+1)**0.5) reduces the importance of
    # steps farther in the future, as their accuracy decreases
    # since we're not recalculating the compensators
    for h in range(horizon_size):
        qcopy, dqcopy, _ = Pendulum.ForwardDynamics(
            dt, qcopy, dqcopy, qForceJoints, ExtAccels)
        er, en = qError_func(qcopy)
        qForceJoints += er*gain/(en**0.25)/((h+1)**0.5)
    CreateNewGoal(enorm)
    return qForceJoints


def update_plot(num):
    '''for each re-draw step, calculate the ForwardDynamics for
    x substeps and then update the plot lines'''
    global q, dq, linemass, linejnt, linegoal, linegoalx, Goal

    # optionally apply a joint force (not updated in substeps
    # to simulate a lower bandwidth controller)
    if UseTorqueController:
        qForceJoints = Controller(dt, q, dq, xyzrpyAccels)
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
        cr = np.cos(Goal[-1])*0.3
        sr = np.sin(Goal[-1])*0.3
        linegoal.set_data(([gx - cr, gx], [gy - sr, gy]))
        linegoalx.set_data(([gx], [gy]))
        return linejnt, linemass, linegoal, linegoalx
    return linejnt, linemass


# start the animation
animation.FuncAnimation(fig, update_plot, interval=dt, blit=True)
plt.show()
