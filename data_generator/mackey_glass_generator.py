
def mackeyglass_eq(x_t, x_t_minus_tau, a, b):
    x_dot = -b * x_t + a*x_t_minus_tau/(1 + x_t_minus_tau**10.0)
    return x_dot

def mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b):
    k1 = deltat*mackeyglass_eq(x_t,x_t_minus_tau,a,b)
    k2 = deltat*mackeyglass_eq(x_t + 0.5*k1, x_t_minus_tau, a, b)
    k3 = deltat*mackeyglass_eq(x_t + 0.5*k2, x_t_minus_tau, a, b)
    k4 = deltat*mackeyglass_eq(x_t+k3, x_t_minus_tau, a, b)
    x_t_plus_deltat = (x_t + k1/6 + k2/3 + k3/3 + k4/6)
    return x_t_plus_deltat

def mackeyglass():
    a = 0.2
    b = 0.1
    tau = 30
    x0 = 1.2
    deltat = 0.1
    sample_n = 12000
    interval = 1

    time = 0
    index = 1
    history_length = int(tau/deltat)
    x_history = [0 for i in range(history_length+1)]
    x_t = x0

    X = [0 for i in range(sample_n + 1)]
    T = [0 for i in range(sample_n + 1)]

    for i in range(1, sample_n+1):
        X[i] = x_t
        if (i-1)%interval == 0:
            print((i-1)/interval,x_t)
        if tau==0:
            x_t_minus_tau = 0.0
        else:
            x_t_minus_tau = x_history[index]

        x_t_plus_deltat = mackeyglass_rk4(x_t,x_t_minus_tau, deltat, a, b)

        if tau!=0:
            x_history[index] = x_t_plus_deltat
            index = (index % history_length) + 1
        time += deltat
        T[i] = time
        x_t = x_t_plus_deltat
    return X

writePath = '/home/minh/PycharmProjects/MultivariateESN/data_backup/Mackey'
no_data = 4000

X = mackeyglass()
with open(writePath, 'w') as f:
    for i in range(1,no_data+1):
        f.write(str(X[i]) + ',' + str(X[i + 1]) + '\n')
