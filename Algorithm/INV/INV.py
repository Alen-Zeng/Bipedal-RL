import math

class Leg_INV:
    def __init__(self) -> None:
        self.l1 = 0.2175
        self.l2 = 0.180
        self.ori_H = 0.296
        self.theta1_offset = 0.6733
        self.theta2_offset = -1.4897
        self.theta3_offset = 0.8163952

    def inv_leg(self,H):
        xp=self.ori_H-H
        cos_theta2 = (pow(xp,2)-pow(self.l1,2)-pow(self.l2,2))/(2*self.l1*self.l2)
        sin_theta2 = -math.sqrt(1-pow(cos_theta2,2))
        theta2 = math.atan2(sin_theta2,cos_theta2)

        k1 = self.l1+self.l2*cos_theta2
        k2 = self.l2*sin_theta2
        theta1 = math.atan2(0,xp)-math.atan2(k2,k1)

        theta3 = -(theta1+theta2)

        theta1 -= self.theta1_offset
        theta2 -= self.theta2_offset
        theta3 -= self.theta3_offset

        return theta1,theta2,theta3
