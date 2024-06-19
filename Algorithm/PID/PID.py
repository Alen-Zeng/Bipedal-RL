import numpy as np

class DualLoopPID:
    def __init__(self, kp_pos, ki_pos, kd_pos, kp_vel, ki_vel, kd_vel, integrator_threshold_pos, integrator_threshold_vel):
        # 外环（位置环）的PID参数
        self.kp_pos = np.array(kp_pos)
        self.ki_pos = np.array(ki_pos)
        self.kd_pos = np.array(kd_pos)
        
        # 内环（速度环）的PID参数
        self.kp_vel = np.array(kp_vel)
        self.ki_vel = np.array(ki_vel)
        self.kd_vel = np.array(kd_vel)

        # 位置环的积分和微分项
        self.integral_pos = np.zeros_like(kp_pos)
        self.prev_error_pos = np.zeros_like(kp_pos)

        # 速度环的积分和微分项
        self.integral_vel = np.zeros_like(kp_vel)
        self.prev_error_vel = np.zeros_like(kp_vel)

        # 积分分离阈值
        self.integrator_threshold_pos = integrator_threshold_pos
        self.integrator_threshold_vel = integrator_threshold_vel

    def position_control(self, position_setpoint, position_feedback):
        # 位置环的误差计算
        error_pos = position_setpoint - position_feedback
        
        # 积分项（带积分分离）
        within_threshold = np.abs(error_pos) < self.integrator_threshold_pos
        self.integral_pos[within_threshold] += error_pos[within_threshold]
        self.integral_pos[~within_threshold] = 0
        
        # 微分项
        derivative_pos = error_pos - self.prev_error_pos
        
        # PID控制输出
        velocity_setpoint = (self.kp_pos * error_pos +
                             self.ki_pos * self.integral_pos +
                             self.kd_pos * derivative_pos)
        
        # 更新前一误差
        self.prev_error_pos = error_pos
        
        return velocity_setpoint

    def velocity_control(self, velocity_setpoint, velocity_feedback):
        # 速度环的误差计算
        error_vel = velocity_setpoint - velocity_feedback
        
        # 积分项（带积分分离）
        within_threshold = np.abs(error_vel) < self.integrator_threshold_vel
        self.integral_vel[within_threshold] += error_vel[within_threshold]
        self.integral_vel[~within_threshold] = 0
        
        # 微分项
        derivative_vel = error_vel - self.prev_error_vel
        
        # PID控制输出
        torque_output = (self.kp_vel * error_vel +
                         self.ki_vel * self.integral_vel +
                         self.kd_vel * derivative_vel)
        
        # 更新前一误差
        self.prev_error_vel = error_vel
        
        return torque_output

    def control(self, position_setpoint, position_feedback, velocity_feedback):
        # 计算速度设定值
        velocity_setpoint = self.position_control(position_setpoint, position_feedback)
        
        # 计算力矩输出
        torque_output = self.velocity_control(velocity_setpoint, velocity_feedback)
        
        return torque_output
