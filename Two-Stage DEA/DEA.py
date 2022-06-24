from gurobipy import *



# TODO solve DEA_VRS models with LP technique by Gurobi Software (package)
def VRS(DMU, X, Y, orientation, dual):
    I = 1
    O = 1
    E = []

    if (orientation == "input" and dual == False):

        for r in DMU:
            try:

                # Initialize LP model
                m = Model("VRS_model")
                m.setParam('OutputFlag', 0)  # Muting the optimize function

                # The decision variable
                v, u, u0 = {}, {}, {}

                # Add decision variables
                for i in range(I):
                    v[r, i] = m.addVar(vtype=GRB.CONTINUOUS, name="v_%s%d" % (r, i))

                for j in range(O):
                    u[r, j] = m.addVar(vtype=GRB.CONTINUOUS, name="u_%s%d" % (r, j))
                u0[r] = m.addVar(lb=-1000, vtype=GRB.CONTINUOUS, name="u0_%s" % r)

                m.update()

                # Add objective function
                m.setObjective(quicksum(u[r, j] * Y[r][j] for j in range(O)) - u0[r], GRB.MAXIMIZE)

                # Add constraints
                m.addConstr(quicksum(v[r, i] * X[r][i] for i in range(I)) == 1)
                for k in DMU:
                    m.addConstr(
                        quicksum(u[r, j] * Y[k][j] for j in range(O)) - quicksum(v[r, i] * X[k][i] for i in range(I)) -
                        u0[r] <= 0)

                # Start optimize the formulation
                m.optimize()

                # Print efficiency
                E.append(m.objVal)

            except GurobiError:
                print('GurobiError reported')
        return (E)


    elif (orientation == "input" and dual == True):
        # TODO solve dual of input-oriented VRS DEA model with LP technique by Gurobi Software (package)
        for r in DMU:
            try:

                # The decision variables
                theta, λ = {}, {}

                # Initialize LP model
                m = Model("Dual_of_CRS_model")
                m.setParam('OutputFlag', False)  # Muting the optimize function

                # Add decision variables
                for k in DMU:
                    λ[k] = m.addVar(vtype=GRB.CONTINUOUS, name="λ_%s" % k)
                theta[r] = m.addVar(vtype=GRB.CONTINUOUS, lb=-1000, name="theta_%s" % r)

                m.update()

                # Add objective function
                m.setObjective(theta[r], GRB.MINIMIZE)

                # Add constraints
                for i in range(I):
                    m.addConstr(quicksum(λ[k] * X[k][i] for k in DMU) <= theta[r] * X[r][i])
                for j in range(O):
                    m.addConstr(quicksum(λ[k] * Y[k][j] for k in DMU) >= Y[r][j])
                m.addConstr(quicksum(λ[k] for k in DMU) == 1, name='sum of λ')

                # Start optimize the formulation
                m.optimize()

                # Print efficiency
                E.append(m.objVal)

            except GurobiError:
                print('GurobiError reported')
        return (E)

            # TODO solve output-oriented DEA_VRS model with LP technique by Gurobi Software (package)
    elif (orientation == "output" and dual == False):
        v0_v = {}

        for r in DMU:
            try:

                # Initialize LP model
                m = Model("VRS_output_model")
                m.setParam('OutputFlag', 0)  # Muting the optimize function

                # The decision variable
                v, u, v0 = {}, {}, {}

                # Add decision variables
                for i in range(I):
                    v[r, i] = m.addVar(vtype=GRB.CONTINUOUS, name="v_%s%d" % (r, i))

                for j in range(O):
                    u[r, j] = m.addVar(vtype=GRB.CONTINUOUS, name="u_%s%d" % (r, j))
                v0[r] = m.addVar(lb=-1000, vtype=GRB.CONTINUOUS, name="v0_%s" % r)

                m.update()

                # Add objective function
                m.setObjective(quicksum(v[r, i] * X[r][i] for i in range(I)) + v0[r], GRB.MINIMIZE)

                # Add constraints
                m.addConstr(quicksum(u[r, j] * Y[r][j] for j in range(O)) == 1)
                for k in DMU:
                    m.addConstr(
                        quicksum(v[r, i] * X[k][i] for i in range(I)) - quicksum(u[r, j] * Y[k][j] for j in range(O)) +
                        v0[r] >= 0)

                # Start optimize the formulation
                m.optimize()

                # Print efficiency
                E.append(m.objVal)

            except GurobiError:
                print('GurobiError reported')
        return (E)

    elif (orientation == "output" and dual == True):
        # TODO solve dual of output-oriented VRS DEA model with LP technique by Gurobi Software (package)
        for r in DMU:
            try:

                # The decision variables
                theta, λ = {}, {}

                # Initialize LP model
                m = Model("Dual_of_output-oriented_VRS_model")
                m.setParam('OutputFlag', False)  # Muting the optimize function

                # Add decision variables
                for k in DMU:
                    λ[k] = m.addVar(vtype=GRB.CONTINUOUS, name="λ_%s" % k)
                theta[r] = m.addVar(vtype=GRB.CONTINUOUS, lb=-1000, name="theta_%s" % r)

                m.update()

                # Add objective function
                m.setObjective(theta[r], GRB.MAXIMIZE)

                # Add constraints
                for j in range(O):
                    m.addConstr(quicksum(λ[k] * Y[k][j] for k in DMU) >= theta[r] * Y[r][j])
                for i in range(I):
                    m.addConstr(quicksum(λ[k] * X[k][i] for k in DMU) <= X[r][i])
                m.addConstr(quicksum(λ[k] for k in DMU) == 1, name='sum of λ')

                # Start optimize the formulation
                m.optimize()

                # Print efficiency
                E.append(m.objVal)

            except GurobiError:
                print('GurobiError reported')
        return (E)