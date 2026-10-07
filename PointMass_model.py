import crocoddyl
import numpy as np
import mim_solvers
from PointMass_utils import Costs, check_collision


def set_solver_callbacks(solver, enabled):
    """Configure verbose callbacks across mim-solvers API versions."""
    callbacks = []
    if enabled:
        callback_type = getattr(
            mim_solvers, "CallbackVerbose", crocoddyl.CallbackVerbose
        )
        callbacks.append(callback_type())
    solver.setCallbacks(callbacks)


class DifferentialActionModelPointMass(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, cost_model, w):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), nu = 2, nr = cost_model.nr
        )  
        # nu = 2 {Fx, Fy} 
        # nr = 2 {Trans, Obs}
        self.unone = np.zeros(self.nu)
        self.m = 1.0 # Dynamics (so to say !!)
        self.cost_model = cost_model
        self.costWeights = w.copy()

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        # Getting the state and control variables
        X, Y, Xdot, Ydot = x[0], x[1], x[2], x[3]
        fx = u[0]
        fy = u[1]

        # Shortname for system parameters
        m = self.m

        # Defining the equation of motions
        Xddot = fx / m
        Yddot = fy / m
        data.xout = np.matrix([Xddot, Yddot]).T
        data.r = self.cost_model.residuals(x, u) 
        data.cost = 0.5 * np.sum(self.costWeights * np.asarray(data.r) ** 2)


def get_results_from_model(cost_set, xs_init, us_init, T, w, dt, max_iter, with_callback = True):
    PM_DAM_running = DifferentialActionModelPointMass(cost_set, w[:cost_set.nr])
    PM_DAM_terminal = DifferentialActionModelPointMass(cost_set, w[cost_set.nr:])
    PM_ND_R = crocoddyl.DifferentialActionModelNumDiff(PM_DAM_running, False)
    PM_ND_T = crocoddyl.DifferentialActionModelNumDiff(PM_DAM_terminal, False)
    PM_IAM = crocoddyl.IntegratedActionModelEuler(PM_ND_R, dt)
    PM_IAM_T = crocoddyl.IntegratedActionModelEuler(PM_ND_T, 0.0)
    problem = crocoddyl.ShootingProblem(xs_init[0], [PM_IAM] * T, PM_IAM_T)
    # Creating the SQP solver
    sqp = mim_solvers.SolverSQP(problem)
    set_solver_callbacks(sqp, with_callback)
    sqp.termination_tolerance = 1e-5
    # xs_init = [x0 for i in range(T+1)]
    # us_init = [u0 for i in range(T)]

    # Solving this problem
    done = sqp.solve(xs_init, us_init, max_iter)
    xs = np.stack(sqp.xs.tolist().copy())
    us = np.stack(sqp.us.tolist().copy())
    # print(done)
    return xs, us, sqp

def test_model_full(cost_set, obs_set, samples, xlims, ylims, T, w, dt, max_iter, with_callback = True):
    xs = []
    us = []
    x_list = np.linspace(xlims[0],xlims[1],samples)
    y_list = np.linspace(ylims[0],ylims[1],samples)
    print('Collecting {} trajectories'.format(samples**2))
    c = 0
    for x_ in x_list:
        for y_ in y_list:
            c += 1
            col = False
            x0 = np.array([x_, y_, 0.0, 0.0])
            u0 = np.array([0.0, 0.0])
            for obs in obs_set:
                if np.linalg.norm(x0[:2] - np.array([obs.x, obs.y])) < obs.R:
                    col = True
            if not col:
                xs_init = [x0 for i in range(T+1)]
                us_init = [u0 for i in range(T)]
                xs_, us_, _ = get_results_from_model(cost_set, xs_init, us_init, T, w, dt, max_iter, with_callback = with_callback)
                xs.append(xs_.copy())
                us.append(us_.copy())
            else:
                print('Trajectory {} Rejected'.format(c))
            if np.mod(c,10) == 0:
                print('{} Trajectories Collected'.format(c))

    return xs, us

def reset_weights(solver, w_run, w_term):
    T = solver.problem.T
    for i in range(T):
        solver.problem.runningModels[i].differential.model.costWeights = w_run
    solver.problem.terminalModel.differential.model.costWeights = w_term
    set_solver_callbacks(solver, False)
    return solver
