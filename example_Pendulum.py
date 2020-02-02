from matplotlib import animation
import matplotlib.pyplot as plt
from rigmech import rigmech

ForceRecompile = False
Pendulum = rigmech("Pendulum")
Dof = 3
if ForceRecompile or not Pendulum.load():
    for jnt in range(Dof):
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

fig, ax = plt.subplots()


def getRobotLines(q, isJoint):
    xline, yline = [], []
    for jnt in Pendulum.Joints.keys():
        if isJoint:  # coj = center of joint coordinate frame
            xyz = Pendulum.joint_syms[jnt]["func_xyz_coj"](*q)
        else:  # com = center of mass coordinate frame
            xyz = Pendulum.joint_syms[jnt]["func_xyz_com"](*q)
        xline += [xyz[0, 0]]
        yline += [xyz[1, 0]]
    return xline, yline


qForceJoints = [0. for _ in range(Dof)]  # tau
ExtForces = [0., -9.81, 0., 0., 0., 0.]  # x y z, wx, wy, wz
q = [0.01 for _ in range(Dof)]  # joint angles in radians
dq = [0. for _ in range(Dof)]  # joint velocities in radians/sec
dt = 1. / 60.
substeps = 10


def update(num):
    global q, dq, linemass, linejnt
    for _ in range(substeps):
        q, dq, _ = Pendulum.ForwardDynamics(
            dt/substeps, q, dq, qForceJoints, ExtForces)
    jx, jy = getRobotLines(q, True)
    mx, my = getRobotLines(q, False)
    linejnt.set_data(jx, jy)
    linemass.set_data(mx, my)
    return linejnt, linemass


linejnt, = ax.plot([0], [0], '-o', label="joint")
linemass, = ax.plot([0], [0], 's', label="mass")
ax.set(xlabel='X', ylabel='Y', xlim=[-3, 3],
       ylim=[-3, 3], title='Pendulum')
ax.set_aspect('equal')
plt.legend(loc='upper right')
animation.FuncAnimation(fig, update, interval=dt, blit=True)
plt.show()
