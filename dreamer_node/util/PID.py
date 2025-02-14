import numpy as np


class PID:
    def __init__(self, kp=0, ki=0, kd=0, target_output_value=0):
        self.Kp = kp  # proportional gain
        self.Ki = ki  # integral gain
        self.Kd = kd  # derivative gain
        self.I = (
            0  # integral term (initialized to zero, accumulated over successive calls)
        )
        self.target_output_value = target_output_value
        self.previous_input_value = np.nan

    def change_tuning_parameters(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.Kp = kp
        if ki is not None:
            self.Ki = ki
            self.I = 0
        if kd is not None:
            self.Kd = kd

    def calculate(self, input_value, dt):
        error = self.target_output_value - input_value  # steady-state error
        if dt <= 0:
            raise ValueError("dt>0 required")
        P = self.Kp * error
        self.I += self.Ki * error * dt
        D = (
            0
            if np.isnan(self.previous_input_value)
            else (self.Kd * (self.previous_input_value - input_value) / dt)
        )
        self.previous_input_value = input_value
        return (P + self.I + D) if (self.Kp + self.Ki + self.Kd > 0) else (input_value)
