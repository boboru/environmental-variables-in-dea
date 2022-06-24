import numpy as np
import gurobipy as gp
from gurobipy import GRB

class EfficiencyCalculator:
    def __init__(self, X, Y):
        # reshape the input and output data to avoid one-dimension array
        if len(X.shape) == 1:
            X = np.reshape(X, (len(X), 1))
        if len(Y.shape) == 1:
            Y = np.reshape(Y, (len(X), 1))
            
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.q = Y.shape[1]
    
    def set_environmental_variables(self, Z):
        if len(Z.shape) == 1:
            Z = np.reshape(Z, (len(Z), 1))
        
        self.Z = Z
        self.r = Z.shape[1]
        
    def set_bandwidth(self, h):
        self.h = h
        
    def set_kernel(self, kernel='triangular'):
        self.kernel = kernel
        
        def triangular(z):
            if abs(z) < 1:
                return 1 - abs(z)
            else:
                return 0

        if self.kernel == 'triangular':
            self.K = triangular
        
    
    def get_FDH_efficiency(self, x, y, X, Y, z=None, Z=None, h=None):
        selected = np.all(Y >= y, axis=1)
        X_sub = X[selected]  # input data of obs., whose output >= the given DMU
        if z is not None and Z is not None and h is not None:
            Z_sub = Z[selected]
            X_sub = X_sub[np.all(np.abs(Z_sub-z) <= h, axis=1)]

        return np.min(np.max(X_sub / x, axis=1))
    
    def get_VRS_efficiency(self, x, y, X, Y, z=None, Z=None, h=None):
        I = self.p
        J = self.q
        
        model = gp.Model('VRS_Input_Oriented_Dual')
        model.setParam('OutputFlag', False)  # mute
        
        # conditional (use subset of input and output data)
        if z is not None and Z is not None and h is not None:
            filter_ = np.all(np.abs(Z-z) <= h, axis=1)
            X = X[filter_]
            Y = Y[filter_]
        
        K = len(X)
        
        lambda_ = model.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name='lambda')
        theta_p = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='theta_p')
        theta_pp = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='theta_pp')
        model.update()


        model.setObjective(theta_p - theta_pp, GRB.MINIMIZE)
        for i in range(I):
            model.addConstr(gp.quicksum((lambda_[k] * X[k][i]) for k in range(K)) <= (theta_p - theta_pp) * x[i])
        for j in range(J):
            model.addConstr(gp.quicksum((lambda_[k] * Y[k][j]) for k in range(K)) >= y[j])

        model.addConstr(gp.quicksum(lambda_[k] for k in range(K)) == 1)

        model.optimize()
            
        return model.objVal
    
    def get_full_efficiency(self, dmu, conditional, method='FDH'):
        x = self.X[dmu]
        y = self.Y[dmu]
        
        if method == 'FDH':
            if not conditional:
                return self.get_FDH_efficiency(x, y, self.X, self.Y)
            else:
                z = self.Z[dmu]
                return self.get_FDH_efficiency(x, y, self.X, self.Y, z=z, Z=self.Z, h=self.h)
        elif method == 'VRS':
            if not conditional:
                return self.get_VRS_efficiency(x, y, self.X, self.Y)
            else:
                z = self.Z[dmu]
                return self.get_VRS_efficiency(x, y, self.X, self.Y, z=z, Z=self.Z, h=self.h)

            
    def get_partial_efficiency(self, dmu, conditional, m, B, method='FDH'):
        """
            m: the sample size of each bootstrap
            B: bootstrap size
        """
        if method == 'FDH':
            x = self.X[dmu]
            y = self.Y[dmu]
            if not conditional:
                eff_bs = []
                for b in range(B):
                    # draw random samples with replacement
                    selected = np.all(self.Y >= y, axis=1)
                    X_sub = self.X[selected]  # input data of obs., whose output >= the given DMU
                    Y_sub = self.Y[selected]
                    m_idx = np.random.choice(len(X_sub), size=m, replace=True)
                    eff_b = self.get_FDH_efficiency(x, y, X_sub[m_idx], Y_sub[m_idx])
                    eff_bs.append(eff_b)
                
                return np.mean(eff_bs)
            else:
                z = self.Z[dmu][0]  # assuming univariate
                eff_bs = []
                for b in range(B):
                    # draw random samples with replacement
                    selected = np.all(self.Y >= y, axis=1)
                    X_sub = self.X[selected]  # input data of obs., whose output >= the given DMU
                    Y_sub = self.Y[selected]
                    Z_sub = self.Z[selected]
                    
                    # set the sampling probability
                    # FIX: assuming univariate Z
                    
                    denominator = np.sum([self.K((z - z_i)/self.h) for z_i in Z_sub.flatten()])
                    p = np.array([self.K((z - z_i)/self.h) for z_i in Z_sub.flatten()])
                    p = p / denominator
                    
                    m_idx = np.random.choice(len(X_sub), size=m, replace=True, p=p)
                    eff_b = self.get_FDH_efficiency(x, y, X_sub[m_idx], Y_sub[m_idx])
                    eff_bs.append(eff_b)
                
                return np.mean(eff_bs)
            
        else:  # VRS
            if not conditional:
                eff_bs = []
                for b in range(B):
                    # draw random samples with replacement
                    selected = np.all(self.Y >= y, axis=1)
                    X_sub = self.X[selected]  # input data of obs., whose output >= the given DMU
                    Y_sub = self.Y[selected]
                    m_idx = np.random.choice(len(X_sub), size=m, replace=True)
                    
                    X = X_sub
                    Y = Y_sub
                    
                    I = self.p
                    J = self.q
                    K = len(X)

                    model = gp.Model('VRS_Input_Oriented_Dual')
                    model.setParam('OutputFlag', False)  # mute

                    lambda_ = model.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name='lambda')
                    theta_p = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='theta_p')
                    theta_pp = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='theta_pp')
                    model.update()

                    model.setObjective(theta_p - theta_pp, GRB.MINIMIZE)
                    for i in range(I):
                        model.addConstr(gp.quicksum((lambda_[k] * X[k][i]) for k in range(K)) <= (theta_p - theta_pp) * x[i])

                    model.addConstr(gp.quicksum(lambda_[k] for k in range(K)) == 1)
                    model.optimize()  
                    eff_bs.append(model.objVal)
                return np.mean(eff_bs)            
            else:  # conditional
                eff_bs = []
                for b in range(B):
                    # draw random samples with replacement
                    selected = np.all(self.Y >= y, axis=1)
                    X_sub = self.X[selected]  # input data of obs., whose output >= the given DMU
                    Y_sub = self.Y[selected]
                    Z_sub = self.Z[selected]
                    
                    # set the sampling probability
                    # FIX: assuming univariate Z
                    
                    denominator = np.sum([self.K((z - z_i)/self.h) for z_i in Z_sub.flatten()])
                    p = np.array([self.K((z - z_i)/self.h) for z_i in Z_sub.flatten()])
                    p = p / denominator
                    
                    m_idx = np.random.choice(len(X_sub), size=m, replace=True, p=p)
                    
                    X = X_sub
                    Y = Y_sub
                    
                    I = self.p
                    J = self.q
                    K = len(X)

                    model = gp.Model('VRS_Input_Oriented_Dual')
                    model.setParam('OutputFlag', False)  # mute

                    lambda_ = model.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name='lambda')
                    theta_p = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='theta_p')
                    theta_pp = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='theta_pp')
                    model.update()

                    model.setObjective(theta_p - theta_pp, GRB.MINIMIZE)
                    for i in range(I):
                        model.addConstr(gp.quicksum((lambda_[k] * X[k][i]) for k in range(K)) <= (theta_p - theta_pp) * x[i])

                    model.addConstr(gp.quicksum(lambda_[k] for k in range(K)) == 1)
                    model.optimize()  
                    eff_bs.append(model.objVal)
                return np.mean(eff_bs)
