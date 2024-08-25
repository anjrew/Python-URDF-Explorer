from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
from typing import List

from urdfpy.urdf import Joint, Link, Transmission


class Angles(BaseModel):
    degrees_45: float = Field(..., description="45 degrees in radians")
    degrees_90: float = Field(..., description="90 degrees in radians")


class WheelMount(BaseModel):
    length_m: float = Field(..., description="Vertical height of the mount in meters")
    diameter_m: float = Field(..., description="Diameter of the mount in meters")
    mass_kg: float = Field(..., description="Mass of the mount in kilograms")


class Chassis(BaseModel):
    length_m: float = Field(..., description="Chassis length in meters")
    width_m: float = Field(..., description="Chassis width in meters")
    height_m: float = Field(..., description="Chassis height in meters")
    mass_kg: float = Field(..., description="Chassis mass in kilograms")
    wheelbase: float = Field(..., description="Wheelbase in meters")
    wheel_x_offset_from_edge_m: float = Field(
        ..., description="Wheel X offset from front and back edge in meters"
    )
    wheel_z_base_offset_m: float = Field(
        ..., description="Wheel Z base offset in meters"
    )
    steering_column_y_center_offset_m: float = Field(
        ...,
        description="Steering column Y offset from the side of the chassis in meters",
    )
    wheel_mounts: WheelMount = Field(
        ..., description="Details for the wheel mounts properties"
    )


class Wheel(BaseModel):
    diameter_m: float = Field(..., description="Wheel diameter in meters")
    width_m: float = Field(..., description="Wheel width in meters")
    mass_kg: float = Field(..., description="Wheel mass in kilograms")


class Tire(BaseModel):
    diameter_m: float = Field(..., description="Tire diameter in meters")
    depth_m: float = Field(..., description="Tire depth in meters")
    width_m: float = Field(..., description="Tire width in meters")


class SteeringColumn(BaseModel):
    max_angle_deg: float = Field(..., description="Maximum steering angle in degrees")
    effort_Nm: float = Field(
        ...,
        description="Maximum torque that can be applied to the steering column in Newton-meters",
    )
    rads_per_sec: float = Field(
        ...,
        description="Maximum angular velocity of the steering column in radians per second",
    )
    front_camber_deg: float = Field(..., description="Camber angle in degrees")


class CarGeometry(BaseModel):
    rotational_center_offset_m: float = Field(
        ..., description="Rotational center offset in meters"
    )


class LidarBase(BaseModel):
    height_m: float = Field(..., description="Lidar base height in meters")
    width_m: float = Field(..., description="Lidar base width in meters")


class LidarScanner(BaseModel):
    diameter_m: float = Field(..., description="Lidar scanner diameter in meters")
    height_m: float = Field(..., description="Lidar scanner height in meters")


class Lidar(BaseModel):
    base: LidarBase
    scanner: LidarScanner
    weight_kg: float = Field(..., description="Lidar weight in kilograms")


class LidarMount(BaseModel):
    height_m: float = Field(..., description="Lidar mount height in meters")
    width_m: float = Field(..., description="Lidar mount width in meters")
    length_offset_m: float = Field(
        ..., description="Lidar mount length offset in meters"
    )
    mass_kg: float = Field(..., description="Lidar mount mass in kilograms")


class Mounts(BaseModel):
    lidar: LidarMount


class AxleProperties(BaseModel):
    length_m: float = Field(..., description="Length of the axle in meters")
    diameter_m: float = Field(..., description="Diameter of the axle in meters")
    mass_kg: float = Field(..., description="Mass of the axle in kilograms")
    wheel_inset_m: float = Field(
        ...,
        description="For far the wheel is inset from the edge of the axle in meters",
    )

@dataclass
class ControlledJoint():
    joint: Joint
    transmission: Transmission


class Boticar(BaseModel):
    chassis: Chassis
    wheel: Wheel
    axle: AxleProperties
    steering_column: SteeringColumn
    tire: Tire
    geometry: CarGeometry
    lidar: Lidar
    mounts: Mounts


class WheelAssemblyURDFComponents:
    wheel: Link
    axle: Link
    steering_column: Link
    wheel_axle_joint: Joint
    axle_column_joint: Joint
    column_chassis_joint: Joint

    def get_joints(self) -> List[Joint]:
        return [
            self.wheel_axle_joint,
            self.axle_column_joint,
            self.column_chassis_joint,
        ]

    def get_links(self) -> List[Link]:
        return [self.wheel, self.axle, self.steering_column]


class TransmissionType(str, Enum):
    SIMPLE = "transmission_interface/SimpleTransmission"
    DIFFERENTIAL = "transmission_interface/DifferentialTransmission"


class HardwareInterfaceType(str, Enum):
    EFFORT_JOINT = "hardware_interface/EffortJointInterface"
    POSITION_JOINT = "hardware_interface/PositionJointInterface"
    VELOCITY_JOINT = "hardware_interface/VelocityJointInterface"
