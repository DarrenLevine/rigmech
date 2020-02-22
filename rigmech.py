"""
MIT License

Copyright (c) 2019 Darren V Levine (https://github.com/DarrenLevine)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# This code is based on work from Travis DeWolf's blog studywolf.wordpress.com
# and Modern Robotics text

import numpy as np
import sympy as sp
import xml.etree.ElementTree as ET
from collections import OrderedDict
import json
import enum
import cloudpickle
import os


class rigmech:
    """ Rigid Body Mechanisms (rigmech)
    * Imports from urdf and/or creates generic models of rigid bodies.
    * Calculates analytic forms of common dynamics equations such as Jacobians
      and transforms.
    * Provides several static helper methods for common transforms.
    * Uses syntax similar to a urdf file.

    Initialize a rigmech:
        Robot = rigmech()

    Add joints (constraints) and links (masses) to the mechanism:
        Robot.addJoint( **joint_parameters )
        Robot.addLink( **link_parameters )

    Or load a mechanism directly from a urdf file (can still be
    modified after loading):
        Robot.loadURDF( urdf_file_name )

    Which gives access to:
        Robot.Joints       - (OrderedDict): Joint parameters (same as in urdf)
        Robot.Links        - (OrderedDict): link parameters (same as in urdf)

    Once the mechanism is set up, the Jacobians and transformations
    can be generated (using sympy):
        Robot.generateEqns()

    generateEqns() gives access to:
        Robot.joint_syms   - (dict): symbols and equations defined relative to
                                     each joint
        Robot.global_syms  - (dict): symbols and equations relating to the
                                     global mechanism

    In addition, since for large mechanisms, the equation calculations can
    take awhile (especially if Simplify is used), the rigmech instance's
    data can be saved and loaded, here is a usage example:
        Robot = rigmech("your_robot_name")
        if not Robot.load(): # if couldn't load from your_robot_name.bin
            Robot.loadURDF("your_robot.urdf")
            Robot.generateEqns(Simplify=True)
            Robot.save() # saves data to your_robot_name.bin

    """

    def __init__(self, name="robot"):
        # note: joints and link containers are OrderedDict so that they
        # can deterministically coorespond to q numbers
        self.Joints = OrderedDict()
        self.Links = OrderedDict()
        self.joint_syms = OrderedDict()
        self.global_syms = {}
        self.sym_prefix = ""
        if not name.isidentifier():
            raise ValueError(
                f"rigmech name must be valid identifier (name.isidentifier())"
            )
        self.name = name
        self.save_filename = name + ".bin"

    def save(self):
        """ saves the data portion of the rigmech object to a user specified
        file"""
        data = (
            self.Joints,
            self.Links,
            self.joint_syms,
            self.global_syms,
            self.name,
            self.sym_prefix,
        )
        cloudpickle.dump(data, open(self.save_filename, "wb"))

    def load(self):
        """ loads the data portion of the rigmech object to a user specified
        file"""
        if os.path.isfile(self.save_filename):
            data = cloudpickle.load(open(self.save_filename, "rb"))
            (
                self.Joints,
                self.Links,
                self.joint_syms,
                self.global_syms,
                self.name,
                self.sym_prefix,
            ) = data
            return True
        return False

    class _RBField(enum.Enum):
        UNSUPPORTED = 1
        REQUIRED = 2

    FIELD_UNSUPPORTED = _RBField.UNSUPPORTED
    FIELD_REQUIRED = _RBField.REQUIRED

    _JointTypes = ["revolute", "continuous",
                   "prismatic", "fixed", "floating", "planar"]
    _DefaultJointFields = {
        "name": FIELD_REQUIRED,
        "type": FIELD_REQUIRED,
        "origin_xyz": [0, 0, 0],
        "origin_rpy": [0, 0, 0],
        "parent": None,
        "child": FIELD_REQUIRED,
        "axis_xyz": [1, 0, 0],
        "damping": 0,
        "friction": 0,
        "limit_lower": None,
        "limit_upper": None,
        "calibration": FIELD_UNSUPPORTED,
        "mimic": FIELD_UNSUPPORTED,
        "safety_controller": FIELD_UNSUPPORTED,
    }
    _DefaultLinkFields = {
        "name": FIELD_REQUIRED,
        "origin_xyz": [0, 0, 0],
        "origin_rpy": [0, 0, 0],
        "mass": FIELD_REQUIRED,
        "inertia": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "visual": FIELD_UNSUPPORTED,
        "collision": FIELD_UNSUPPORTED,
    }

    @staticmethod
    def _check_field_inputs(funcname, DefaultFields, PassedFields):
        """ checks link and joint field definitions for common input errors"""
        for key, val in DefaultFields.items():
            if val is rigmech.FIELD_REQUIRED:
                if PassedFields.get(key) is None:
                    raise ValueError(
                        f"{key} is a required input to {funcname}()")
            elif val is rigmech.FIELD_UNSUPPORTED:
                if not PassedFields.get(key) is None:
                    raise ValueError(
                        f"{funcname}({key}) is a currently unsupported")
        for key in PassedFields.keys():
            if key not in DefaultFields.keys():
                raise ValueError(f"{key} is not an argument for {funcname}()")
        DefaultFieldsWithoutUF = {
            k: v for k, v in DefaultFields.items()
            if v is not rigmech.FIELD_UNSUPPORTED
        }
        return {**DefaultFieldsWithoutUF, **PassedFields}

    def _insertJoint(self, new_joint_key_value, before_joint_name=None):
        key, value = new_joint_key_value
        if before_joint_name is None:
            self.Joints[key] = value
        else:
            new_dict = OrderedDict()
            for k, v in self.Joints.items():
                if k == before_joint_name:
                    new_dict[key] = value
                new_dict[k] = v
            self.Joints = new_dict

    def addJoint(self, **kwargs):
        """Add a constraint (joint) between bodies (links).

        Args:
            name (str): Specifies a unique name of the joint
            type (str): Specifies the type of joint, which can be one of the
                        following:
                "revolute" - a hinge joint that rotates along the axis and has
                    a limited range specified by the upper and lower limits.
                "continuous" - a continuous hinge joint that rotates around
                    the axis and has no upper and lower limits.
                "prismatic" - a sliding joint that slides along the axis, and
                    has a limited range specified by the upper and lower
                    limits.
                "fixed" - This is not really a joint because it cannot move.
                    All degrees of freedom are locked. This type of joint does
                    not require the axis, calibration, dynamics, limits or
                    safety_controller.
                "floating" - This joint allows motion for all 6 degrees of
                    freedom.
                "planar" - This joint allows motion in a plane perpendicular
                    to the axis.
            origin_xyz (list, (optional)): Represents the [x,y,z] offset. All
                positions are specified in meters. Default of [0,0,0].
            origin_rpy (list, (optional)): Represents the rotation around
                fixed axis:
                first roll around x, then pitch around y and finally yaw
                around z. All angles are specified in radians. Default of
                [0,0,0].
            parent (str): The name of the link that is the parent of this link
                in the robot tree structure.
            child (str): The name of the link that is the child link.
            axis_xyz (list, (optional)): Represents the [x,y,z] components of
                a vector. The vector should be normalized. The joint axis
                specified in the joint frame. This is the axis of rotation for
                revolute joints, the axis of translation for prismatic joints,
                and the surface normal for planar joints. The axis is specified
                in the joint frame of reference. Fixed and floating joints do
                not use the axis field.
            damping (float, (optional)): The physical damping value of the
                joint (N*s/m for prismatic joints, N*m*s/rad for revolute
                joints). Defaults to 0.
            friction (float, (optional)): The physical static friction value
                of the joint (N for prismatic joints, N*m for revolute joints).
                Defaults to 0.
            limit_lower (float, (optional)): An attribute specifying the lower
                joint limit (radians for revolute joints, meters for prismatic
                joints). Omit if joint is continuous. Defaults to None.
            limit_upper (float, (optional)): An attribute specifying the
                upper joint limit (radians for revolute joints, meters for
                prismatic joints). Omit if joint is continuous. Defaults to
                None.
            insert_before (str, (optional)): Inserts this joint into the list
                before the insert_before joint name. Does not effect dynamics,
                simply a convenience to be able to force a specific order when
                itterating through the Joints OrderedDict.
        """
        insert_before = kwargs.get("insert_before")
        if insert_before is not None:
            kwargs.pop("insert_before")
        JointArgs = rigmech._check_field_inputs(
            "addJoint", self._DefaultJointFields, kwargs
        )
        if not JointArgs["type"] in self._JointTypes:
            raise ValueError(
                f"Joint type {JointArgs['type']} is not a recognized type:\
                    {self._JointTypes}"
            )
        # insert the joint (optionally before another)
        self._insertJoint((JointArgs["name"], JointArgs), insert_before)

    def _expandPlanar(self, joint_name):
        plane_norm = self.Joints[joint_name]["axis_xyz"]
        dof_vect = (1 - np.array(plane_norm)).tolist()
        axes = [
            ([dof_vect[0], 0, 0], "x", "prismatic"),
            ([0, dof_vect[1], 0], "y", "prismatic"),
            ([0, 0, dof_vect[2]], "z", "prismatic"),
        ]
        axes = [v for v in axes if any([np.abs(s) > 1e-6 for s in v[0]])]
        self._replaceJointWith(joint_name, axes)

    def _expandFloating(self, joint_name):
        axes = [
            ([1, 0, 0], "x", "prismatic"),
            ([0, 1, 0], "y", "prismatic"),
            ([0, 0, 1], "z", "prismatic"),
            ([1, 0, 0], "Wx", "continuous"),
            ([0, 1, 0], "Wy", "continuous"),
            ([0, 0, 1], "Wz", "continuous"),
        ]
        self._replaceJointWith(joint_name, axes)

    def _replaceJointWith(self, joint_name, axes):
        original_joint = self.Joints[joint_name]
        child_linkname = original_joint["child"]
        original_origin = {
            "origin_xyz": original_joint["origin_xyz"],
            "origin_rpy": original_joint["origin_rpy"],
        }
        jnt_list = [joint_name]
        link_list = [original_joint["parent"]]
        for axis, deg, jtype in axes:
            jnt_list += [joint_name + "_j" + deg]
            link_list += [child_linkname + "_l" + deg]
            new_joint = {
                **original_joint,
                **{
                    "name": jnt_list[-1],
                    "type": jtype,
                    "parent": "_temp_",
                    "child": "_temp_",
                    "axis_xyz": axis,
                    "insert_before": joint_name,
                },
                **original_origin,
            }
            self.addJoint(**new_joint)
            self.addLink(name=link_list[-1], mass=0)
            # make sure the origin xyz,rpy transform is only applied to the
            # first frame
            original_origin["origin_xyz"] = [0, 0, 0]
            original_origin["origin_rpy"] = [0, 0, 0]

        # remove the last automatically link, and replace it with the original
        # child
        self.Links.pop(link_list[-1])
        link_list[-1] = child_linkname

        # sort out the linkages
        for cnt, jnt in enumerate(jnt_list[1:]):
            self.Joints[jnt]["parent"] = link_list[cnt]
            self.Joints[jnt]["child"] = link_list[cnt + 1]

        # remove the orginal joint this is meant to replace (only
        # at the end of this process, since it is reference in "insert_before")
        self.Joints.pop(joint_name)

    def addLink(self, name=None, **kwargs):
        """Add a rigid body (link)

        Args:
            name (str): The name of the link itself.
            origin_xyz (list,(optional)): Represents the [x,y,z] offset.
                Default of [0,0,0]. This is the pose of the inertial
                reference frame, relative to the link reference frame.
                The origin of the inertial reference frame needs to be at
                the center of gravity. The axes of the inertial reference
                frame do not need to be aligned with the principal axes of the
                inertia. origin_rpy (list,(optional)): Represents the fixed
                axis roll, pitch and yaw angles in radians.
            mass (str): The mass of the link is represented by the value
                attribute of this element.
            inertia (list 3x3 or 6x6,(optional)): The 3x3 rotational inertia
                matrix [[ixx,ixy,ixz],[iyx,iyy,iyz],[izx,izy,izz]],
                represented in the inertia frame. The rotational inertia
                matrix must be symmetric. Defaults to eye. Or, the full 6x6
                inertia matrix if available.
        """
        if isinstance(name, rigmech):
            self.sym_prefix = name.sym_prefix + "_"
            self.addLink(
                name=name.name,
                mass=name.global_syms["mass"],
                inertia=name.global_syms["Mq"],
                origin_xyz=name.global_syms["xyz_com"],
            )
        else:
            kwargs["name"] = name
            LinkArgs = rigmech._check_field_inputs(
                "addLink", self._DefaultLinkFields, kwargs
            )
            self.Links[LinkArgs["name"]] = LinkArgs

    def loadURDF(self, filename):
        """ loads a mechanism from a .urdf file by calling self.addJoint()
        and self.addLink() for each joint and link in the file """

        # format the default values for joints and links
        d_jnt, d_lnk = {}, {}
        for k, v in self._DefaultJointFields.items():
            d_jnt[k] = str(v).replace(", ", " ").replace(
                "[", "").replace("]", "")
        for k, v in self._DefaultLinkFields.items():
            d_lnk[k] = str(v).replace(", ", " ").replace(
                "[", "").replace("]", "")
        mt = self._DefaultLinkFields["inertia"]

        # define some other helpful formatting functions
        def to_list(xmlstr):
            return eval(f'[{xmlstr.replace(" ",",")}]')

        def eget(v, n, d):
            return eval(v.get(n, d))

        # how to convert the urdf xml into a pythonic datastructure
        def _pythonize_urdf(root, data={}):
            for field in root:
                if isinstance(data, dict) and field.tag not in data.keys():
                    data[field.tag] = []
                data[field.tag] += [field.attrib]
                data[field.tag][-1] = _pythonize_urdf(
                    field, data[field.tag][-1])
            for key in data.keys():
                if isinstance(data[key], list) and len(data[key]) == 1:
                    data[key] = data[key][0]
            return data

        # convert the xml into a pythonic dict/list datastructure
        urdf = _pythonize_urdf(ET.parse(filename).getroot())

        # add the links
        for link in urdf.get("link", []):
            im = link["inertial"].get("inertia", {})
            imat = [
                [
                    eget(im, "ixx", mt[0][0]),
                    eget(im, "ixy", mt[0][1]),
                    eget(im, "ixz", mt[0][2]),
                ],
                [
                    eget(im, "ixy", mt[1][0]),
                    eget(im, "iyy", mt[1][1]),
                    eget(im, "iyz", mt[1][2]),
                ],
                [
                    eget(im, "ixz", mt[2][0]),
                    eget(im, "iyz", mt[2][1]),
                    eget(im, "izz", mt[2][2]),
                ],
            ]
            self.addLink(
                **{
                    "name": link["name"],
                    "origin_xyz": to_list(
                        link["inertial"]
                        .get("origin", {})
                        .get("xyz", d_lnk["origin_xyz"])
                    ),
                    "origin_rpy": to_list(
                        link["inertial"]
                        .get("origin", {})
                        .get("rpy", d_lnk["origin_rpy"])
                    ),
                    "mass": eval(link["inertial"]["mass"]["value"]),
                    "inertia": imat,
                }
            )

        # add the joints
        for joint in urdf.get("joint", []):
            self.addJoint(
                **{
                    "name": joint["name"],
                    "type": joint["type"],
                    "origin_xyz": to_list(
                        joint.get("origin", {}).get("xyz", d_jnt["origin_xyz"])
                    ),
                    "origin_rpy": to_list(
                        joint.get("origin", {}).get("rpy", d_jnt["origin_rpy"])
                    ),
                    "parent": joint["parent"]["link"],
                    "child": joint["child"]["link"],
                    "axis_xyz": to_list(
                        joint.get("axis", {}).get("xyz", d_jnt["axis_xyz"])
                    ),
                    "damping": eval(
                        joint.get("dynamics", {}).get(
                            "damping", d_jnt["damping"])
                    ),
                    "friction": eval(
                        joint.get("dynamics", {}).get(
                            "friction", d_jnt["friction"])
                    ),
                    "limit_lower": eval(
                        joint.get("limit", {}).get(
                            "lower", d_jnt["limit_lower"])
                    ),
                    "limit_upper": eval(
                        joint.get("limit", {}).get(
                            "upper", d_jnt["limit_upper"])
                    ),
                }
            )
        return urdf

    @staticmethod
    def Tform(xyz, R):
        """ forms a homogeneous transformation matrix T from a displacement
        and rotation matrix """
        xyz = sp.Matrix([xyz[0], xyz[1], xyz[2]]
                        )  # enforce orientation and type
        T = sp.Matrix(R).row_join(xyz).col_join(sp.Matrix([[0, 0, 0, 1]]))
        return T

    @staticmethod
    def R(rpy):
        """ Calculates a Rotation matrix from roll pitch and yaw angles
        using a Rx_roll*Ry_pitch*Rz_yaw rotational convention"""
        Rx_roll = sp.Matrix(
            [
                [1, 0, 0],
                [0, sp.cos(rpy[0]), -sp.sin(rpy[0])],
                [0, sp.sin(rpy[0]), sp.cos(rpy[0])],
            ]
        )
        Ry_pitch = sp.Matrix(
            [
                [sp.cos(rpy[1]), 0, sp.sin(rpy[1])],
                [0, 1, 0],
                [-sp.sin(rpy[1]), 0, sp.cos(rpy[1])],
            ]
        )
        Rz_yaw = sp.Matrix(
            [
                [sp.cos(rpy[2]), -sp.sin(rpy[2]), 0],
                [sp.sin(rpy[2]), sp.cos(rpy[2]), 0],
                [0, 0, 1],
            ]
        )
        R = Rx_roll * Ry_pitch * Rz_yaw
        return R

    @staticmethod
    def T(xyz, rpy, Tbase=None):
        """ Calculates a homogeneous transformation matrix T
        using a Rx_roll*Ry_pitch*Rz_yaw rotational convention"""
        R = rigmech.R(rpy)
        T = rigmech.Tform(xyz, R)
        if Tbase is not None:
            T = Tbase * T
        return T

    @staticmethod
    def applyTx(T, xyz=[0, 0, 0]):
        """ Apply rotation and translation transformations to xyz"""
        return (T * sp.Matrix([xyz[0], xyz[1], xyz[2], 1]))[:3, 0]

    @staticmethod
    def applyTw(T, Wxyz=[0, 0, 0]):
        """ Apply only rotation transformations to xyz"""
        return (T * sp.Matrix([Wxyz[0], Wxyz[1], Wxyz[2], 0]))[:3, 0]

    @staticmethod
    def T2Rxyz(T):
        """ Seperates a homogeneous transformation matrix T into its rotation
        matrix R and translational xyz components"""
        R = T[:3, :3]
        xyz = T[:3, 3]
        return R, xyz

    @staticmethod
    def T_inv(T):
        """ inverts a homogeneous transformation matrix T """
        R, xyz = rigmech.T2Rxyz(T)
        R_inv = R.T
        xyz_inv = -R_inv * xyz
        T_inv = R_inv.row_join(xyz_inv).col_join(sp.Matrix([[0, 0, 0, 1]]))
        return T_inv

    @staticmethod
    def toQuat(roll_pitch_yaw):
        """Converts a [roll,pitch,yaw] array into
        a quaternion"""
        cos_r = np.cos(roll_pitch_yaw[0] * 0.5)
        sin_r = np.sin(roll_pitch_yaw[0] * 0.5)
        cos_p = np.cos(roll_pitch_yaw[1] * 0.5)
        sin_p = np.sin(roll_pitch_yaw[1] * 0.5)
        cos_y = np.cos(roll_pitch_yaw[2] * 0.5)
        sin_y = np.sin(roll_pitch_yaw[2] * 0.5)
        w = cos_y * cos_p * cos_r + sin_y * sin_p * sin_r
        x = cos_y * cos_p * sin_r - sin_y * sin_p * cos_r
        y = sin_y * cos_p * sin_r + cos_y * sin_p * cos_r
        z = sin_y * cos_p * cos_r - cos_y * sin_p * sin_r
        return np.array([w, x, y, z])

    @staticmethod
    def QuatConj(wxyz):
        """Conjugates a 4 element array quaternion"""
        return np.array([wxyz[0], -wxyz[1], -wxyz[2], -wxyz[3]])

    @staticmethod
    def QuatMag(wxyz):
        """Returns the magnitude of a 4 element array quaternion"""
        return np.sqrt(np.sum(np.square(wxyz)))

    @staticmethod
    def QuatNormalize(wxyz):
        """Returns the normalized version of the input quaternion"""
        return wxyz/rigmech.QuatMag(wxyz)

    @staticmethod
    def QuatMult(q1, q2):
        """Multiplies two quaternions"""
        w = -q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3] + q1[0] * q2[0]
        x = q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1]
        y = -q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2]
        z = q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0] + q1[0] * q2[3]
        return np.array([w, x, y, z])

    @staticmethod
    def QuatAngleDiff(Wxyz1, Wxyz2):
        """Calculates the euler angle error/difference using
        quaternion operations"""
        q1 = rigmech.QuatNormalize(rigmech.toQuat(Wxyz1))
        q2 = rigmech.QuatNormalize(rigmech.toQuat(Wxyz2))
        q3 = rigmech.QuatMult(q2, rigmech.QuatConj(q1))
        WxyzChange = q3[1:] * np.sign(q3[0])
        return WxyzChange

    def T_joint_chain(self, joint_name):
        """ Calculates a homogeneous transformation matrix T from a kinematic
        chain's base joint to the specified joint """
        if self.joint_syms[joint_name].get("T_joint") is None:
            # go up the parent chain of transformations
            parent_joint_name = self.global_syms["Jname2parentJname"].get(
                joint_name)
            if parent_joint_name is None:
                self.joint_syms[joint_name]["T_joint"] = \
                    self.joint_syms[joint_name]["Tlocal_joint"]
            else:
                self.joint_syms[joint_name]["T_joint"] = (
                    self.T_joint_chain(parent_joint_name)
                    * self.joint_syms[joint_name]["Tlocal_joint"]
                )
        return self.joint_syms[joint_name]["T_joint"]

    def W_joint_chain(self, joint_name):
        """ Sums the rotational transformations from a kinematic
        chain's base joint to the specified joint """
        if self.joint_syms[joint_name].get("W") is None:
            # go up the parent chain of transformations
            parent_joint_name = self.global_syms["Jname2parentJname"].get(
                joint_name)
            if parent_joint_name is None:
                self.joint_syms[joint_name]["W"] = \
                    self.joint_syms[joint_name]["q_rpy"]
            else:
                self.joint_syms[joint_name]["W"] = (
                    self.W_joint_chain(parent_joint_name)
                    + self.joint_syms[joint_name]["q_rpy"]
                )
        return self.joint_syms[joint_name]["W"]

    def _preprocess_heirarchy(self, FloatingBase):
        ResolveLinks = True
        while ResolveLinks:

            # expand floating (6-dof) and planer joints (2-dof) into
            # serparate joints each with a single dof
            name_types = [(name, val["type"])
                          for name, val in self.Joints.items()]
            for joint_name, joint_type in name_types:
                if joint_type == "floating":
                    self._expandFloating(joint_name)
                if joint_type == "planar":
                    self._expandPlanar(joint_name)

            # create a look-up-table between link names and parent joint names
            Lname2parentJname = {}
            for joint_name in self.Joints:
                child_link = self.Joints[joint_name]["child"]
                if child_link in Lname2parentJname.keys():
                    raise ValueError(
                        "Cannot handle possible kinematic loops (a child link\
                            with multiple parent joints)"
                    )
                Lname2parentJname[child_link] = joint_name

            ResolveLinks = False
            linknames_list = list(self.Links.keys())
            for link_name in linknames_list:
                if Lname2parentJname.get(link_name) is None:
                    ResolveLinks = True
                    jtype = "floating" if FloatingBase else "fixed"
                    automatic_joint = link_name + "_joint"
                    print(
                        f'Warning: link "{link_name}" had no parent joint,' +
                        f'automatically adding {jtype} joint ' +
                        f'"{automatic_joint}" for pybullet compatibility'
                    )
                    parent_lnk = None
                    first_joint = next(iter(self.Joints.items()))[0]
                    self.addJoint(
                        **{
                            "name": automatic_joint,
                            "type": jtype,
                            "parent": parent_lnk,
                            "child": link_name,
                            "origin_xyz": -np.array(
                                self.Links[link_name]["origin_xyz"]
                            ),
                            "origin_rpy": -np.array(
                                self.Links[link_name]["origin_rpy"]
                            ),
                            "insert_before": first_joint,
                        }
                    )

        # create a look-up-table between joint names and parent link names
        Jname2parentJname = {}
        for joint_name in self.Joints:
            parent_link = self.Joints[joint_name]["parent"]
            Jname2parentJname[joint_name] = Lname2parentJname.get(parent_link)
        return Lname2parentJname, Jname2parentJname

    def generateEqns(
        self, Simplify=False, Lambdify=True, FloatingBase=False,
        backend="numpy"
    ):
        """ Finalizes the mechanism by populating self.global_syms and
        self.joint_syms with helpful rigid body equations and symbols
        using sympy, such as jacobians and frame transformations.
        Also post-processes the mechanism structure for any necessary
        additions such as seperating multi-dof joints into single-dof joints.

        Args:
            Simplify (bool): Simplify any symbolic expressions.Default is
                False (since it can take a long time).
            Lambdify (bool): Lambdify any symbolic expressions.Default is
                False (since it can take a long time).
            FloatingBase (bool): If not base joint is specified, one will
                automatically be added, if this parameter is true, that
                joint will be a "floating" joint, otherwise it will be
                "fixed".
            backend (str): The sympy backend to use if Lambdify = True.
                Defaults to "numpy".
        """
        self.joint_syms = OrderedDict()
        self.global_syms = {}
        self.global_syms["Jname2q"] = {}
        self.global_syms["q2Jname"] = {}
        _Lname2parentJname, _Jname2parentJname = self._preprocess_heirarchy(
            FloatingBase
        )
        self.global_syms["Lname2parentJname"] = _Lname2parentJname
        self.global_syms["Jname2parentJname"] = _Jname2parentJname

        # record the number of degrees of freedom
        degrees_of_freedom = sum(
            [self.Joints[jnt]["type"] != "fixed" for jnt in self.Joints]
        )
        self.global_syms["dof"] = degrees_of_freedom

        # joint positions q
        self.global_syms["q"] = [
            sp.Symbol(f"{self.sym_prefix}q{j}")
            for j in range(degrees_of_freedom)
        ]

        # joint velocities dq
        self.global_syms["dq"] = [
            sp.Symbol(f"{self.sym_prefix}dq{j}")
            for j in range(degrees_of_freedom)
        ]

        # joint user forces tau
        self.global_syms["qTau"] = [
            sp.Symbol(f"{self.sym_prefix}qTau{j}")
            for j in range(degrees_of_freedom)
        ]

        # [x,y,z] translations (meaning relative to useage)
        self.global_syms["xyz"] = [
            sp.Symbol(f"{self.sym_prefix}x"),
            sp.Symbol(f"{self.sym_prefix}y"),
            sp.Symbol(f"{self.sym_prefix}z"),
        ]
        zero_xyz = [(s, 0) for s in self.global_syms["xyz"]]

        # [Wx,Wy,Wz] rotations (meaning relative to useage)
        self.global_syms["Wxyz"] = [
            sp.Symbol(f"{self.sym_prefix}Wx"),
            sp.Symbol(f"{self.sym_prefix}Wy"),
            sp.Symbol(f"{self.sym_prefix}Wz"),
        ]
        zero_Wxyz = [(s, 0) for s in self.global_syms["Wxyz"]]

        # translational and rotational accelerations [Ax,Ay,Az,AWx,AWy,AWz]
        # (meaning relative to useage)
        self.global_syms["extAccel"] = [
            sp.Symbol(f"{self.sym_prefix}Ax"),
            sp.Symbol(f"{self.sym_prefix}Ay"),
            sp.Symbol(f"{self.sym_prefix}Az"),
            sp.Symbol(f"{self.sym_prefix}AWx"),
            sp.Symbol(f"{self.sym_prefix}AWy"),
            sp.Symbol(f"{self.sym_prefix}AWz"),
        ]

        #
        # create terms for each joint/link combo in the local isolated
        # reference frame (terms that need no other connected joint terms)
        #
        q_indx = 0
        for j_name in self.Joints:
            joint = self.Joints[j_name]
            clink = self.Links[joint["child"]]
            joint_type = joint["type"]

            # initialize an eqn dict for this joint (and link)
            self.joint_syms[j_name] = {}
            E = self.joint_syms[j_name]

            # joint (and link) mass
            E["mass"] = clink["mass"]

            # joint (and link) specific inertia matrix
            Inertia = sp.Matrix(clink["inertia"])
            if Inertia.shape == (3, 3):
                E["M"] = sp.Matrix(
                    [
                        [clink["mass"], 0, 0, 0, 0, 0],
                        [0, clink["mass"], 0, 0, 0, 0],
                        [0, 0, clink["mass"], 0, 0, 0],
                        [0, 0, 0, Inertia[0, 0], Inertia[0, 1], Inertia[0, 2]],
                        [0, 0, 0, Inertia[1, 0], Inertia[1, 1], Inertia[1, 2]],
                        [0, 0, 0, Inertia[2, 0], Inertia[2, 1], Inertia[2, 2]],
                    ]
                )
            elif Inertia.shape == (6, 6):
                E["M"] = Inertia
            else:
                raise ValueError(
                    f"inertia shape must be 3x3 or 6x6, not {Inertia.shape}")

            # re-record (for convenience) the local q and dq, joint and joint
            # velocity terms, in their joint symbol containers
            if joint_type == "fixed":
                E["q"] = 0
                E["dq"] = 0
                E["qTau"] = 0
            else:
                E["q"] = self.global_syms["q"][q_indx]
                E["dq"] = self.global_syms["dq"][q_indx]
                E["qTau"] = self.global_syms["qTau"][q_indx]
                q_indx += 1
                self.global_syms["q2Jname"][E["q"]] = j_name
            self.global_syms["Jname2q"][j_name] = E["q"]

            # process each joint type and apply the relevant q to a rpy,xyz
            # transform
            E["q_rpy"] = sp.Matrix([0, 0, 0])
            E["q_xyz"] = sp.Matrix([0, 0, 0])
            if joint_type == "revolute" or joint_type == "continuous":
                E["q_rpy"] = E["q"] * sp.Matrix(joint["axis_xyz"])
            elif joint_type == "prismatic":
                E["q_xyz"] = E["q"] * sp.Matrix(joint["axis_xyz"])
            elif joint_type == "fixed":
                pass
            elif joint_type == "floating":
                raise ValueError(
                    "no direct floating joint support (should have been" +
                    " replaced by 3 prismatic, 3 continuous)"
                )
            elif joint_type == "planar":
                raise ValueError(
                    "no direct planar joint support (should have been" +
                    " replaced by 2 prismatic)"
                )

            # creating homogeneous transformation matrix T, in joint and mass
            # spaces for various tranforms.
            #
            # The chain of transformations is diagramed as:
            # ... parent joint --> joint origin --> joint actuated --> ... etc.
            #     actuated     |                                   |
            #                   --> parent link                     --> link
            #

            # parent joint's actuateed frame to joint's actuated frame
            E["Tlocal_joint"] = rigmech.T(
                joint["origin_xyz"], joint["origin_rpy"]
            ) * rigmech.T(E["q_xyz"], E["q_rpy"])

            # joint's actuated frame to the child link's inertial frame
            E["T_joint2cLink"] = rigmech.T(
                clink["origin_xyz"], clink["origin_rpy"])

            # parent joint's actuateed frame to child link's frame
            E["Tlocal_link"] = E["Tlocal_joint"] * E["T_joint2cLink"]

            # inverse transformations
            E["Tlocal_joint_inv"] = rigmech.T_inv(E["Tlocal_joint"])
            E["Tlocal_link_inv"] = rigmech.T_inv(E["Tlocal_link"])

            print(f"rigmech: Calculated {j_name} isolated.")
        #
        # create non-isolated terms for each joint (terms that require
        # information about other connected joints)
        #

        for j_name in self.Joints:
            E = self.joint_syms[j_name]

            # T: transforms from base to joint or mass, for forward transform
            # calculations
            E["T_joint"] = self.T_joint_chain(j_name)
            E["T_link"] = E["T_joint"] * E["T_joint2cLink"]

            # T_inv: transforms for forward inverse transform calculations
            E["T_inv_joint"] = rigmech.T_inv(E["T_joint"])
            E["T_inv_link"] = rigmech.T_inv(E["T_link"])

            # xyz: translation from base to joint or link frame
            E["xyz_joint"] = rigmech.applyTx(
                E["T_joint"], E["q_xyz"]+sp.Matrix(self.global_syms["xyz"]))
            E["xyz_link"] = rigmech.applyTx(
                E["T_link"], E["q_xyz"]+sp.Matrix(self.global_syms["xyz"]))
            E["xyz_coj"] = E["xyz_joint"].subs(zero_xyz)  # center of joint
            E["xyz_com"] = E["xyz_link"].subs(zero_xyz)  # center of mass

            # Wxyz: rotation from base to joint or link frame
            E["W"] = self.W_joint_chain(j_name)
            E["Wxyz_joint"] = rigmech.applyTw(
                E["T_joint"], E["W"]+sp.Matrix(self.global_syms["Wxyz"]))
            E["Wxyz_link"] = rigmech.applyTw(
                E["T_link"], E["W"]+sp.Matrix(self.global_syms["Wxyz"]))
            E["Wxyz_coj"] = E["Wxyz_joint"].subs(zero_Wxyz)  # coj orientation
            E["Wxyz_com"] = E["Wxyz_link"].subs(zero_Wxyz)  # com orientation

            # calculate the d[x(i) y(i) z(i) Wx(i) Wy(i) Wz(i)]/dq(j)
            # a.k.a. jacobian components for the current joint/link frame
            # (i) with respect to all the other joints (j) to form a
            # complete Jacobian matrix
            E["J_joint"] = sp.Matrix()
            E["J_link"] = sp.Matrix()
            for jnm in self.Joints:
                jnm_q = self.joint_syms[jnm]["q"]
                if jnm_q is not 0:

                    # joints:
                    dxyz_dq__joint = E["xyz_joint"].diff(jnm_q)
                    dWxyz_dq__joint = E["Wxyz_joint"].diff(jnm_q)
                    new_row = dxyz_dq__joint.col_join(dWxyz_dq__joint)
                    E["J_joint"] = E["J_joint"].row_join(new_row)

                    # links:
                    dxyz_dq__link = E["xyz_link"].diff(jnm_q)
                    dWxyz_dq__link = E["Wxyz_link"].diff(jnm_q)
                    new_row = dxyz_dq__link.col_join(dWxyz_dq__link)
                    E["J_link"] = E["J_link"].row_join(new_row)

            # evaluate the link frame Jacobian at xyz = [0,0,0] and
            # Wxyz = [0,0,0] to get the center of mass (COM) Jacobian
            E["J_com"] = E["J_link"].subs(zero_xyz + zero_Wxyz)
            # evaluate the joint frame Jacobian at xyz = [0,0,0] and
            # Wxyz = [0,0,0] to get the center of joint (COJ) Jacobian
            E["J_coj"] = E["J_joint"].subs(zero_xyz + zero_Wxyz)

            # Mq: joint space inertia matrix of single joint
            E["Mq"] = E["J_com"].T * E["M"] * E["J_com"]

            # qFext: joint space matrix of the forces due to external
            # accelerations (such as gravity) on single joint
            E["qFext"] = E["J_com"].T * E["M"] * \
                sp.Matrix(self.global_syms["extAccel"])

            print(f"rigmech: Calculated {j_name} non-isolated.")

        #
        # create terms common to entire mechanism
        #

        # Mq: joint space inertia matrix of entire mechanism
        self.global_syms["Mq"] = sp.zeros(degrees_of_freedom)
        for j_name in self.Joints:
            self.global_syms["Mq"] += self.joint_syms[j_name]["Mq"]

        # qFext: joint space matrix of the forces due to external
        # accelerations (such as gravity) on entire mechanism
        self.global_syms["qFext"] = sp.zeros(degrees_of_freedom, 1)
        for j_name in self.Joints:
            self.global_syms["qFext"] += self.joint_syms[j_name]["qFext"]

        # qFrict: joint friction in a convenient list
        self.global_syms["qFrict"] = [
            self.Joints[jnt]["friction"]
            for jnt in self.Joints
            if not self.joint_syms[jnt]["q"] is 0
        ]

        # xyz_com: xyz center of mass of entire mechanism
        total_mass = 0.0
        weighted_mass = sp.Matrix([0, 0, 0])
        for j_name in self.Joints:
            E = self.joint_syms[j_name]
            total_mass += E["mass"]
            weighted_mass += E["xyz_com"] * E["mass"]
        self.global_syms["xyz_com"] = weighted_mass / total_mass
        self.global_syms["mass"] = total_mass

        # Cq(q,dq) joint space Coriolis matrix (coriolis and centrifugal terms)
        # of entire mechanism
        i_max, j_max = self.global_syms["Mq"].shape
        Mq = self.global_syms["Mq"]
        q = self.global_syms["q"]
        dq = self.global_syms["dq"]
        Cq = sp.zeros(i_max, j_max)
        for k in range(len(q)):
            for i in range(i_max):
                for j in range(i_max):
                    if not dq[k] is 0:
                        dmij_dqk = 0 if q[k] is 0 else Mq[i, j].diff(q[k])
                        dmik_dqj = 0 if q[j] is 0 else Mq[i, k].diff(q[j])
                        dmkj_dqi = 0 if q[i] is 0 else Mq[k, j].diff(q[i])
                        Cq[i, j] += (dmij_dqk + dmik_dqj - dmkj_dqi) * dq[k]
        Cq = 0.5 * Cq
        self.global_syms["Cq"] = Cq

        # forces due to coriolis matrix in joint space
        self.global_syms["qFCoriolis"] = Cq * sp.Matrix(dq)

        print(f"rigmech: Calculated global_syms.")

        if Simplify:
            print(f"rigmech: starting simplify()")
            self.simplify()

        if Lambdify:
            print(f"rigmech: starting lambdify()")
            self.lambdify(backend)

        print(f"rigmech: done")

        return self.joint_syms, self.global_syms

    def simplify(self):
        """ simplify all the symbolic terms in self.joint_syms, and
        self.global_syms (in place)"""
        for key, val in self.global_syms.items():
            if isinstance(val, (sp.Expr, sp.Matrix)):
                simp = sp.simplify(val)
                self.global_syms[key] = simp
                print(f"global_syms[{key}] pre-simplified:\n{val}\n")
                print(f"global_syms[{key}] simplified:\n{simp}\n")
        for j_name in self.Joints:
            for key, val in self.joint_syms[j_name].items():
                if isinstance(val, (sp.Expr, sp.Matrix)):
                    simp = sp.simplify(val)
                    self.joint_syms[j_name][key] = simp
                    print(
                        f"joint_syms[{j_name}][{key}] pre-simplify:\n{val}\n")
                    print(
                        f"joint_syms[{j_name}][{key}] simplified:\n{simp}\n")

    def lambdify(self, backend="numpy"):
        """ turns any symbolic terms into lambda functions, with
        the prefix addition "func_"

        Args:
            backend (str, optional): Backend selection to pass to sympy's
            lambdify. Defaults to 'numpy'
        """

        # wont work without joints
        if len(self.Joints) == 0:
            return

        # possible input states:
        q_syms = self.global_syms["q"]
        dq_syms = self.global_syms["dq"]
        xyz_syms = self.global_syms["xyz"]
        Wxyz_syms = self.global_syms["Wxyz"]
        Axyz_syms = self.global_syms["extAccel"]

        def ordered_inputs_to_expr(expression):
            start_syms = [
                str(x).replace("_", "")[0]
                for x in list(expression.free_symbols)
            ]
            inputs = []
            # at a minimum, always include joint state as a common input
            inputs += q_syms
            if "d" in start_syms:
                inputs += dq_syms
            if "x" in start_syms or "y" in start_syms or "z" in start_syms:
                inputs += xyz_syms
            if "W" in start_syms:
                inputs += Wxyz_syms
            if "A" in start_syms:
                inputs += Axyz_syms
            # add any missing symbols (from external rigmechs)
            missing_symbols = list(
                set(list(expression.free_symbols)) - set(inputs))
            ms_list = []
            if missing_symbols:
                ms_indxs = np.argsort([str(x) for x in missing_symbols])
                for j in ms_indxs:
                    ms_list += [missing_symbols[j]]
            return q_syms + ms_list + inputs[len(q_syms):]

        first_joint = next(iter(self.Joints.items()))[0]
        filter_out = ["q", "dq", "q_rpy", "q_xyz", "qTau"]
        jsym_names = [
            key
            for key, val in self.joint_syms[first_joint].items()
            if key not in filter_out and isinstance(val, (sp.Expr, sp.Matrix))
        ]
        glsym_names = [
            key
            for key, val in self.global_syms.items()
            if isinstance(val, (sp.Expr, sp.Matrix))
        ]

        # define lambdas relative to joints
        for jsym_name in jsym_names:
            for jn in self.Joints:
                expression = self.joint_syms[jn][jsym_name]
                input_list = ordered_inputs_to_expr(expression)
                self.joint_syms[jn]["func_" + jsym_name] = sp.lambdify(
                    input_list, expression, backend
                )

        # define lambdas relative to global mechanism
        for glsym_name in glsym_names:
            expression = self.global_syms[glsym_name]
            input_list = ordered_inputs_to_expr(expression)
            if glsym_name in ['qFCoriolis'] and not dq_syms[0] in input_list:
                input_list += dq_syms
            self.global_syms["func_" + glsym_name] = sp.lambdify(
                input_list, expression, backend
            )

    def ForwardDynamics(
            self,
            dt,
            q,
            dq,
            qForceJoints,
            extAccels=[0., 0., -9.81, 0., 0., 0.],
            Friction=True,
            Quadratic=True,
            rcond=.005
    ):
        """Simulates a forward euler step in time for the machanism.

        Args:
            dt (float): Time step size
            q (list of floats): initial joint position vector (in joint space)
            dq (list of floats): initial joint velocity vector (in joint space)
            qForceJoints (list of floats): joint force vector (in joint space)
            extAccels (list of 6 floats): fictitious translational and
                rotational accelerations (in cartesian space) [x,y,z,Wx,Wy,Wz]
                defaults to -z gravity vector ([0., 0., -9.81, 0., 0., 0.])
            Friction (bool, optional): If true, includes a the influence of
                global_syms["qFrict"]. Defaults to True.
            Quadratic (bool, optional): If true, includes a the influence of
                global_syms["qFCoriolis"]. Defaults to True.
            rcond (float, optional): Cut-off ratio for small singular values in Mq. 

        Returns:
            q (list of floats): final joint position vector (in joint space)
            dq (list of floats): final joint velocity vector (in joint space)
            ddq (list of floats): final joint accel vector (in joint space)
        """
        q = np.array(q).reshape(len(q), 1).copy()
        shape = (len(dq), 1)
        dq = np.array(dq).reshape(shape).copy()
        dqlst = dq.T.tolist()[0]
        qlst = q.T.tolist()[0]
        qForceExternal = self.global_syms["func_qFext"](*qlst, *extAccels)
        qForceJoints = np.array(qForceJoints).reshape(len(qForceJoints), 1)
        qForces = qForceJoints + qForceExternal
        if Quadratic:  # include qudratic force contributions
            qForceQuadratic = self.global_syms[
                "func_qFCoriolis"](*qlst, *dqlst)
            qForces -= qForceQuadratic
        if Friction:  # include friction force contributions
            dForceFriction = -dq * \
                np.array(self.global_syms["qFrict"]).reshape(shape)
            qForces += dForceFriction
        Mq = self.global_syms["func_Mq"](*qlst)
        # accel (ddq) = Force (qForces) / mass (Mq)
        ddq = np.linalg.lstsq(Mq, qForces, rcond=rcond)[0]
        # euler stepping
        q[: len(dq)] += dt * dq
        dq += dt * np.array(ddq)
        return q.T.tolist()[0], dq.T.tolist()[0], ddq.T.tolist()[0]

#
# Misc. helper functions:
#


def printjson(toprint, exception_conversions={}):
    """print anything in json format. Can override printing for custom
    types using a dictionary of types to lambda conversion functions.
    Examples:
        printjson(data)
        printjson(data,{sp.DenseSparseMatrix: lambda x: f"matrix  = {x}" })
    """

    def serialize(x):
        conversion = exception_conversions.get(type(x))
        if conversion is None:
            try:
                return json.dumps(x)
            except Exception:
                return str(x)
        else:
            return conversion(x)

    print(json.dumps(toprint, indent=4, default=serialize))
