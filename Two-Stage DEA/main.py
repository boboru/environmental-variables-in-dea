import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from truncreg import Truncreg
import DEA

#=============================================== Monte-Carlo ===============================================
#trial
trial = 100

beta1_mean_list = []
beta1_std_list = []
beta2_mean_list = []
beta2_std_list = []
sigma_mean_list = []
sigma_std_list = []

for t in range(trial):
    #=============================================== DGP ===============================================
    # sample size
    n = 100 #400

    # beta
    beta_1 = 0.5
    beta_2 = 0.5

    # generate environment variable z
    np.random.seed(seed=3*t)
    mu_z = 2
    sigma_z = 2
    z = norm.rvs(mu_z,sigma_z,size=n)
    zn = norm.rvs(mu_z,sigma_z,size=n)

    # generate error term
    np.random.seed(seed=3*t+1)
    sigma_e = 1
    error = norm.rvs(0,sigma_e,size=n)

    # generate inefficiency U
    delta = beta_1 + z*beta_2 + error
    for i in range(n): # left-truncated
        if delta[i] < 1:
            delta[i] = 1

    # generate input x
    np.random.seed(seed=3*t+3)
    x = uniform.rvs(6,16,size=n)

    # generate output y
    y = (delta**-1) * (x**(3/4))


    #=============================================== DEA ===============================================
    # DataFrame
    DMU_index = list(range(0,n))
    DMU = [str(x) for x in DMU_index]
    df_Y = pd.DataFrame({'Y':y},index=DMU)
    df_X = pd.DataFrame({'X':x},index=DMU)
    Y = df_Y.T.to_dict('list')
    X = df_X.T.to_dict('list')

    # input-oriented VRS (dual)
    d = DEA.VRS(DMU, X, Y, orientation="output", dual=True)

    #=============================================== Truncated Regression ===============================================
    # DataFrame
    #df = pd.DataFrame({'D': d, 'D_trunc':d, 'Z':z})
    df = pd.DataFrame({'D': d, 'D_trunc':d, 'Z':z, 'Zn':zn})

    # Truncated Data
    left = 1
    cond = (df.loc[:,'D'] <= left)
    df.loc[cond,'D_trunc'] = np.nan
    d_trunc = df['D_trunc']

    # Fitting Truncated Regression
    formula_trunc = 'D_trunc ~ Z'
    res_trunc = Truncreg.from_formula(formula_trunc,left=1,data=df).fit()
    print(res_trunc.summary().tables[1])
    df_trunc = pd.read_html(res_trunc.summary().tables[1].as_html(),header=0,index_col=0)[0]
    beta1_h = df_trunc.coef[0]
    beta2_h = df_trunc.coef[1]
    sigma_h = np.mean(res_trunc.resid**2)

    #=============================================== Bootstrap Estimation ===============================================
    B = 100
    beta1_b_list = []
    beta2_b_list = []
    sigma_b_list = []
    for b in range(B):
        #1
        np.random.seed(seed=b)
        eps = norm.rvs(0, sigma_h, size=n)
        for i in range(n):
            if eps[i] < 1-z[i]*beta2_h - beta1_h:
                eps[i] = 1-z[i]*beta2_h - beta1_h
            else:
                eps[i] = eps[i]

        #2
        d_s = beta1_h + z*beta2_h + eps

        #3
        # DataFrame
        df_b = pd.DataFrame({'D': d_s, 'D_trunc': d_s, 'Z': z})

        # Truncated Data
        left = 1
        cond = (df_b.loc[:, 'D'] <= left)
        df_b.loc[cond, 'D_trunc'] = np.nan
        d_trunc = df_b['D_trunc']

        # Fitting Truncated Regression
        formula_trunc = 'D_trunc ~ Z'
        res_b_trunc = Truncreg.from_formula(formula_trunc, left=1, data=df_b).fit()
        df_b_trunc = pd.read_html(res_b_trunc.summary().tables[1].as_html(), header=0, index_col=0)[0]
        beta1_b = df_b_trunc.coef[0]
        beta2_b = df_b_trunc.coef[1]
        sigma_b = np.mean(res_b_trunc.resid ** 2)
        beta1_b_list.append(beta1_b)
        beta2_b_list.append(beta2_b)
        sigma_b_list.append(sigma_b)
        print('b=',b)
    beta1_mean_list.append(np.mean(beta1_b_list))
    beta1_std_list.append(np.std(beta1_b_list))
    beta2_mean_list.append(np.mean(beta2_b_list))
    beta2_std_list.append(np.std(beta2_b_list))
    sigma_mean_list.append(np.mean(sigma_b_list))
    sigma_std_list.append(np.std(sigma_b_list))
print(beta1_mean_list, beta1_std_list)
print(beta2_mean_list, beta2_std_list)
print(sigma_mean_list, sigma_std_list)

alpha = 0.95 # 0.9 0.95, 0.975, 0.995 (0.8, 0.9, 0.95, 0.99)
beta1_lb_list = np.add(beta1_mean_list, [-norm.ppf(alpha, loc=0, scale=1)*i for i in beta1_std_list])
beta1_ub_list = np.add(beta1_mean_list, [norm.ppf(alpha, loc=0, scale=1)*i for i in beta1_std_list])
beta2_lb_list = np.add(beta2_mean_list, [-norm.ppf(alpha, loc=0, scale=1)*i for i in beta2_std_list])
beta2_ub_list = np.add(beta2_mean_list, [norm.ppf(alpha, loc=0, scale=1)*i for i in beta2_std_list])
sigma_lb_list = np.add(sigma_mean_list, [-norm.ppf(alpha, loc=0, scale=1)*i for i in sigma_std_list])
sigma_ub_list = np.add(sigma_mean_list, [norm.ppf(alpha, loc=0, scale=1)*i for i in sigma_std_list])
beta1_in = 0
beta2_in = 0
sigma_in = 0
for i in range(trial):
    if beta1_lb_list[i] < 0.5 and beta1_ub_list[i] > 0.5:
        beta1_in += 1
    if beta2_lb_list[i] < 0.5 and beta2_ub_list[i] > 0.5:
        beta2_in += 1
    if sigma_lb_list[i] < 1 and sigma_ub_list[i] > 1:
        sigma_in += 1
print(beta1_in)
print(beta2_in)
print(sigma_in)
