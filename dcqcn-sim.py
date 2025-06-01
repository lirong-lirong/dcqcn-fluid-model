import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# Global parameters (converted to a class for better encapsulation)
class Params:
    def __init__(self):
        # Network parameters
        self.packetSize = 8e3  # 8000 bits
        self.C = 100e9 / self.packetSize  # packets per second, 100Gbps
        self.numFlows = 2
        self.taustar = 85e-6  # feedback delay
        
        # DCQCN parameters
        self.Rai = 40e6 / self.packetSize  # packets per second
        self.B = 10e6 * 8 / self.packetSize  # packets
        self.timer = 55e-6  # seconds
        self.F = 5
        self.g = 1/256
        self.tau = 50e-6  # seconds
        self.tauprime = 55e-6  # seconds
        
        # ECN/PFC parameters
        self.queueLength = 10e5 * 8 / self.packetSize  # packets
        self.Kmin = 5e3 * 8 / self.packetSize  # packets
        self.Kmax = 200e3 * 8 / self.packetSize  # packets
        self.pmax = 0.5  # 50%
        self.pfcThreshold = 800e3 * 8 / self.packetSize  # packets
        self.pfcPauseTime = 65535 * 512 / 40e9  # seconds
        self.pfcStartTime = 0.0
        
        # Simulation parameters
        self.numCalls = 0
        self.lossNumPackets = 0
        
        # Proactively generate CNP
        self.S = 80e9 / self.packetSize
        self.inner_term = (1/self.B) + (1/(self.S*self.timer))
        self.Psend = ( (self.Rai / (self.tauprime * self.S**2)) * (self.inner_term)**2 )**(1/3)
        
        self.S2 = 40e9 / self.packetSize
        self.inner_term2 = (1/self.B) + (1/(self.S2*self.timer))
        self.Psend2 = ( (self.Rai / (self.tauprime * self.S2**2)) * (self.inner_term2)**2 )**(1/3)
        
        
params = Params()

# History function (constant initial conditions)
def history(t):
    initVal = np.zeros(3 * params.numFlows + 3)
    initVal[0:3*params.numFlows:3] = params.C  # rc initial
    initVal[1:3*params.numFlows:3] = params.C  # rt initial
    initVal[2:3*params.numFlows:3] = 1.0       # alpha initial
    return initVal

# DCQCN model (delay differential equations)
def model(Y, t):
    # Unpack current state
    x = Y(t)
    xlag = Y(t - params.taustar) if t > params.taustar else history(t - params.taustar)
    
    dx = np.zeros(3 * params.numFlows + 3)
    qlag = xlag[-2]  # queue at time t - taustar    # S0
    
    # Calculate marking probability
    p = calculate_p(qlag, params.Kmin, params.Kmax)
    # p = params.Psend
    
    # Compute derivatives for each flow
    for i in range(0, 3*params.numFlows, 3):
        prevRC = xlag[i]
        currRC = x[i]
        currRT = x[i+1]
        currAlpha = x[i+2]
        if i == 0:
            p_ = min((p + params.Psend), 1)
            # p_ = p
            # p_ = min((params.Psend), 1)
        else:
            # p_ = p
            p_ = min((p + params.Psend2), 1)
        a, b, c, d, e = intermediate_terms(p_, prevRC, currRC, i)
        dx[i] = rc_delta(currRC, currRT, currAlpha, prevRC, a, b, d)
        dx[i+1] = rt_delta(currRC, currRT, prevRC, a, c, e)
        dx[i+2] = alpha_delta(currAlpha, prevRC, p_)
    
    # Compute queue derivatives
    rates = x[0:3*params.numFlows:3]
    S1currentQueue = x[-3]
    S0currentQueue = x[-2]
    dx[-3], dx[-2], dx[-1] = queue_delta(S1currentQueue, S0currentQueue, rates, t)
    
    params.numCalls += 1
    if params.numCalls % 10000 == 0:
        print(f"{t} {params.numCalls} {p}")
    
    return dx

# Helper functions
def calculate_p(q, kmin, kmax):
    if q <= kmin:
        return 0.0
    elif q <= kmax:
        return (q - kmin) / (kmax - kmin) * params.pmax
    else:
        return 1.0
    

def intermediate_terms(p, prevRC, currRC, i):
    if p == 0:
        a = 0.0
        b = 1 / params.B
        c = b
        d = 1 / (params.timer * prevRC) if prevRC > 0 else 0.0
        e = d
    elif p == 1:
        a = 1.0
        b = 0.0
        c = 0.0
        d = 0.0
        e = 0.0
    else:   # ECN or Delay: Lessons Learnt from Analysis of DCQCN and TIMELY 公式(11)
        a = 1 - (1 - p) ** (params.tau * prevRC)
        b = p / ((1 - p) ** (-params.B) - 1)
        c = b * (1 - p) ** (params.F * params.B)
        d = p / ((1 - p) ** (-params.timer * prevRC) - 1)
        if np.isinf(d):
            d = 1 / (params.timer * prevRC) if prevRC > 0 else 0.0
        e = d * (1 - p) ** (params.F * params.timer * prevRC)
    return a, b, c, d, e

def rc_delta(currRC, currRT, currAlpha, prevRC, a, b, d):
    term1 = -currRC * currAlpha * a / (2 * params.tau)
    term2 = (currRT - currRC) * prevRC * b / 2
    term3 = (currRT - currRC) * prevRC * d / 2
    delta = term1 + term2 + term3
    if currRC >= params.C and delta > 0:
        return 0.0
    return delta

def rt_delta(currRC, currRT, prevRC, a, c, e):
    term1 = -(currRT - currRC) * a / params.tau
    term2 = params.Rai * prevRC * c
    term3 = params.Rai * prevRC * e
    delta = term1 + term2 + term3
    if currRT >= params.C and delta > 0:
        return 0.0
    return delta

def alpha_delta(currentAlpha, prevRC, p):
    term = (1 - (1 - p) ** (params.tauprime * prevRC)) - currentAlpha
    return params.g * term / params.tauprime

def queue_delta(S1currentQueue, S0currentQueue, rates, t):
    total_rate = np.sum(rates)
    S1_delta = 0.0
    S0_delta = 0.0
    loss_delta = 0.0
    
    # Handle PFC states
    if params.pfcStartTime != 0.0:
        if t - params.pfcStartTime < params.pfcPauseTime:
            S1_delta = total_rate
            S0_delta = -params.C
        else:
            if S0currentQueue >= params.pfcThreshold:
                params.pfcStartTime = t
                S1_delta = total_rate
                S0_delta = -params.C
            else:
                S1_delta = total_rate - params.C
                S0_delta = total_rate - params.C
    else:
        if S0currentQueue < params.pfcThreshold:
            S1_delta = total_rate - params.C
            S0_delta = total_rate - params.C
        else:
            params.pfcStartTime = t
            S1_delta = total_rate
            S0_delta = -params.C
    
    # Prevent queues from going negative
    if S0currentQueue <= 0 and S0_delta < 0:
        S0_delta = 0.0
    if S1currentQueue <= 0 and S1_delta < 0:
        S1_delta = 0.0
    
    # Handle queue overflow
    if S1currentQueue >= params.queueLength and S1_delta > 0:
        S1_delta = 0.0
        loss_delta = total_rate
    if S0currentQueue >= params.queueLength and S0_delta > 0:
        S0_delta = 0.0
    
    return S1_delta, S0_delta, loss_delta

def utilization(t, rates, q, C):
    sent = 0.0
    tmin = t[0]
    tmax = t[-1]
    total_max = C * (tmax - tmin)
    err = 0
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if q[i] > 1:
            rate_sum = C
        else:
            rate_sum = np.sum(rates[:, i])
            if rate_sum > C:
                rate_sum = C
                err += 1
        sent += rate_sum * dt
    
    return sent / total_max, err

# def plot_sol(t, queueS0, queueS1, lossPacketNum, rates, sim_length):
#     plt.figure(figsize=(10, 12))
    
#     plt.subplot(4, 1, 1)
#     plt.plot(t, rates.T * params.packetSize / 1e9, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, 1.1 * np.max(rates) * params.packetSize / 1e9)
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Throughput (Gbps)')
#     plt.title('Per-flow Throughput')
    
#     plt.subplot(4, 1, 2)
#     plt.plot(t, queueS0, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, 1.1 * np.max(queueS0))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Queue (Packets)')
#     plt.title('S0 Queue (ECN & PFC)')
    
#     plt.subplot(4, 1, 3)
#     plt.plot(t, queueS1, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, 1.1 * np.max(queueS1))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Queue (Packets)')
#     plt.title('S1 Queue (ECN & PFC)')
    
#     plt.subplot(4, 1, 4)
#     plt.plot(t, lossPacketNum, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, max(1, 1.1 * np.max(lossPacketNum)))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Packet Loss')
#     plt.title('S1 Packet Loss')
    
#     plt.tight_layout()
#     plt.savefig("1")
#     plt.show()

# def plot_sol(t, queueS0, queueS1, lossPacketNum, rates, sim_length):
#     plt.figure(figsize=(12, 16))
#     num_flows = rates.shape[0]  # 获取流的数量
    
#     # 1. 每个流的吞吐量（单独子图）
#     # plt.subplot(4, 1, 1)
#     for i in range(num_flows):
#         plt.subplot(num_flows + 3, 1, i + 1)  # 前num_flows个位置给每个流
#         plt.plot(t, rates[i, :] * params.packetSize / 1e9, linewidth=2)
#         plt.xlim(0, sim_length)
#         plt.ylim(0, 1.1 * np.max(rates[i, :]) * params.packetSize / 1e9)
#         plt.xlabel('Time (seconds)')
#         plt.ylabel('Throughput (Gbps)')
#         plt.title(f'Flow {i+1} Throughput')
    
#     # 2. S0队列
#     plt.subplot(num_flows + 3, 1, num_flows + 1)
#     # plt.subplot(4, 1, 2)
#     plt.plot(t, queueS0, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, 1.1 * np.max(queueS0))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Queue (Packets)')
#     plt.title('S0 Queue (ECN & PFC)')
    
#     # 3. S1队列
#     plt.subplot(num_flows + 3, 1, num_flows + 2)
#     # plt.subplot(4, 1, 3)
#     plt.plot(t, queueS1, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, 1.1 * np.max(queueS1))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Queue (Packets)')
#     plt.title('S1 Queue (ECN & PFC)')
    
#     # 4. 丢包
#     plt.subplot(num_flows + 3, 1, num_flows + 3)
#     # plt.subplot(4, 1, 4)
#     plt.plot(t, lossPacketNum, linewidth=2)
#     plt.xlim(0, sim_length)
#     plt.ylim(0, max(1, 1.1 * np.max(lossPacketNum)))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Packet Loss')
#     plt.title('S1 Packet Loss')
    
#     plt.tight_layout()
#     plt.savefig("dcqcn_simulation_results.png", dpi=300)
#     plt.show()

def plot_sol(t, queueS0, queueS1, lossPacketNum, rates, sim_length):
    plt.figure(figsize=(12, 16))
    num_flows = rates.shape[0]  # 获取流的数量
    
    # 1. 所有流的吞吐量（合并到一张子图上）
    plt.subplot(4, 1, 1)
    # 创建不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, num_flows))
    
    for i in range(num_flows):
        plt.plot(t, rates[i, :] * params.packetSize / 1e9, 
                 linewidth=2, color=colors[i], 
                 label=f'Flow {i+1}')
    
    plt.xlim(0, sim_length)
    max_rate = np.max(rates) * params.packetSize / 1e9
    plt.ylim(0, 1.1 * max_rate)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (Gbps)')
    plt.grid(True)
    plt.title('All Flows Throughput')
    plt.legend(loc='upper right')  # 添加图例
    
    # 2. S0队列
    plt.subplot(4, 1, 2)
    plt.plot(t, queueS0, linewidth=2)
    plt.xlim(0, sim_length)
    plt.ylim(0, 1.1 * np.max(queueS0))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Queue (Packets)')
    plt.title('S0 Queue (ECN & PFC)')
    
    # 3. S1队列
    plt.subplot(4, 1, 3)
    plt.plot(t, queueS1, linewidth=2)
    plt.xlim(0, sim_length)
    plt.ylim(0, 1.1 * np.max(queueS1))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Queue (Packets)')
    plt.title('S1 Queue (ECN & PFC)')
    
    # 4. 丢包
    plt.subplot(4, 1, 4)
    plt.plot(t, lossPacketNum, linewidth=2)
    plt.xlim(0, sim_length)
    plt.ylim(0, max(1, 1.1 * np.max(lossPacketNum)))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Packet Loss')
    plt.title('S1 Packet Loss')
    
    plt.tight_layout()
    plt.savefig("dcqcn_simulation_results.png", dpi=300)
    plt.show()

# Main simulation
def main():
    sim_length = 0.2  # s    0.5
    
    t_eval = np.linspace(0, sim_length, 10000)         # 100000
    # t_eval = np.arange(0, sim_epoch) * params.taustar
    print(f"psend: ", params.Psend)

    # Solve DDE
    sol = ddeint(model, history, t_eval)
    sol = sol.T  # Transpose to match MATLAB shape
    
    # Extract results
    rates = sol[0:3*params.numFlows:3, :]
    queueS1 = sol[-3, :]
    queueS0 = sol[-2, :]
    lossPacketNum = sol[-1, :]
    
    # Calculate utilization
    util, err = utilization(t_eval, rates, queueS0, params.C)
    print(f"Utilization: flows={params.numFlows} util={util:.6f} err={err}")
    
    # Plot results
    plot_sol(t_eval, queueS0, queueS1, lossPacketNum, rates, sim_length)

if __name__ == "__main__":
    main()