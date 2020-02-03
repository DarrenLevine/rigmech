from matplotlib import animation
import matplotlib.pyplot as plt
from rigmech import rigmech

# initialize the rigmech (with an optional name for the save file)
# and some simulation parameters
Pendulum = rigmech("Pendulum")
ForceRecompile = False
MotorCount = 3  # 1= single pendulum, 2= double pendulum, etc.
dt = 1. / 60.
substeps = 10

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
                          origin_xyz=[0, 1 if jnt > 0 else 0, 0],
                          friction=0.05)
        Pendulum.addLink(name=f"mass{jnt+1}",
                         origin_xyz=[0, 1, 0],
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
linemass, = ax.plot([0], [0], 's', label="mass")
ax.set(xlabel='X', ylabel='Y', xlim=[-3, 3],
       ylim=[-3, 3], title='Pendulum')
ax.set_aspect('equal')
plt.legend()

# initialize state
qForceJoints = [0. for _ in range(MotorCount)]  # tau
ExtForces = [0., -9.81, 0., 0., 0., 0.]  # x y z, wx, wy, wz
q = [0.01 for _ in range(MotorCount)]  # joint angles in radians
dq = [0. for _ in range(MotorCount)]  # joint velocities in radians/sec


def update_plot(num):
    '''for each re-draw step, calculate the ForwardDynamics for
    x substeps and then update the plot lines'''
    global q, dq, linemass, linejnt
    for _ in range(substeps):
        q, dq, _ = Pendulum.ForwardDynamics(
            dt/substeps, q, dq, qForceJoints, ExtForces)
    linejnt.set_data(getRobotLines(q, True))
    linemass.set_data(getRobotLines(q, False))
    return linejnt, linemass


# start the animation
animation.FuncAnimation(fig, update_plot, interval=dt, blit=True)
plt.show()
