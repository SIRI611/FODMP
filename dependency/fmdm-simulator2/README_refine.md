1. add initial speed of the ball 

   ```python
   # set velocity -> set random velocity
   ball_id = target_obj_info['obj_id']
   bullet_client.resetBaseVelocity(
       ball_id,
       linearVelocity=[-1.0, 0.5, 0.0],   # m/s in +x (change as you like)
       angularVelocity=[0.0, 0.0, 0.0]
   )
   ```

2. xdefine the friction of the table when init the env

   ```python
   # set friction for table
   self.bullet_client.changeDynamics(self.table_id, -1,
                             lateralFriction=0.1,
                             spinningFriction=0.0,
                             rollingFriction=0.0)
   ```

3. add expert stop_object policy 

   ```python
      def stop_object(self, obj_info, grasping_z=0.002, dist_thres=0.005, max_approach_steps=60,
                       v_ee_max=0.5,     # EE max XY speed (m/s): tune to your controller
                       latency=0.05,     # control/obs latency margin (s)
                       lead=0.02,        # meet the ball slightly ahead along its heading (m)
                       g=9.81):
           """
           Intercept the rolling ball and stop it: plan to the earliest reachable point
           along its friction-slowed trajectory.
           Returns: ([dx, dy, dz, discrete_skill_index], done_mark)
           """
           pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_info['obj_id'])
           vel, _ = self.bullet_client.getBaseVelocity(obj_info['obj_id'])
           x, y, z = pos
           vx, vy, vz = vel
          	v0 = float((vx**2 + vy**2) ** 0.5) # 2) Effective friction μ from table dynamics (fallback to a sane default)
           try:
               di = self.bullet_client.getDynamicsInfo(self.table_id, -1)
               lat_mu = float(di[1])
               roll_mu = float(di[6]) if len(di) > 6 and di[6] is not None else 0.0
               spin_mu = float(di[7]) if len(di) > 7 and di[7] is not None else 0.0
               mu_eff = max(1e-4, 0.7*lat_mu + 0.2*roll_mu + 0.1*spin_mu)
           except Exception:
               mu_eff = 0.15  # default; tune to match observed decel
   
           a = max(1e-6, mu_eff * g)  # deceleration magnitude
   
           # Early exit: ball basically stopped → go to its current XY
           if v0 < 1e-4:
               target_xy = np.array([x, y], dtype=float)
           else:
               dhat = np.array([vx, vy], dtype=float) / v0          # heading
               t_stop = v0 / a                                       # time to stop
               # parametric position along path under constant decel
               def pos_at_t(t):
                   t = float(np.clip(t, 0.0, t_stop))
                   s = max(0.0, v0*t - 0.5*a*t*t)                   # arc length traveled
                   return np.array([x, y]) + dhat * s
   
               # 3) choose earliest reachable intercept time t
               # current EE XY
               ee_states = self.franka_robot.get_end_effector_states()
               ex, ey, ez = map(float, ee_states['position'])
               dt_plan = 0.02
               t = 0.0
               best_t = None
               while t <= t_stop + 1e-9:
                   px, py = pos_at_t(t)
                   dist = float(np.hypot(px - ex, py - ey))
                   t_needed = dist / max(1e-6, v_ee_max) + latency
                   if t_needed <= t:
                       best_t = t
                       break
                   t += dt_plan
   
               if best_t is None:
                   # can’t meet before stop → aim at stop point
                   s_stop = v0*v0 / (2.0*a)
                   target_xy = np.array([x, y]) + dhat * s_stop
               else:
                   # lead slightly along heading to meet the front of the ball
                   target_xy = pos_at_t(best_t) + dhat * lead
   
           # 4) Move toward intercept point at a blocking height
           intercept_z = grasping_z + 0.08
           goal_xyz = np.array([target_xy[0], target_xy[1], intercept_z], dtype=float)
   
           delta_xyz, _, dist_linear, _ = self.franka_robot.approach_target(goal_xyz)
           dx, dy, dz = delta_xyz
   
           # Your existing done logic
           discrete_skill_index = 0
           done_mark = False
           self.robot_approach_target_attempt_step += 1
           if self.robot_approach_target_attempt_step >= max_approach_steps or dist_linear <= dist_thres:
               self.robot_approach_target_attempt_step = 0
               done_mark = True
   
           return [dx, dy, dz, discrete_skill_index], done_mark
   ```