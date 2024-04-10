#!/usr/bin/env python

import numpy as np 

class PID:

    def __init__(self, p, i, d, sat, dt):
        self.p = p
        self.i = i 
        self.d = d
        self.sat = sat 
        self.dt = dt # set dt to 0 to disable dt 

        self.integral = 0
        self.prev_err = 0
        self.prev_t = -1.0

    def regulate(self, err, t):
            
            derr_dt = 0.0
            if self.dt == 0:
                dt = t - self.prev_t
            else:
                dt = self.dt

            if self.prev_t > 0.0 and dt > 0.0:
                derr_dt = (err - self.prev_err)/dt
                # self.integral += 0.5*(err + self.prev_err)*dt # more accuracy in the error value
                # general formula 
                self.integral += err*dt 

            u = self.p*err + self.d*derr_dt + self.i*self.integral

            self.prev_err = err
            self.prev_t = t

            if (np.linalg.norm(u) > self.sat):
                # controller is in saturation: limit outpt, reset integral
                u = self.sat*u/np.linalg.norm(u)
                self.integral = 0.0

            return u