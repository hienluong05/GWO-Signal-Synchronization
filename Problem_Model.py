import numpy as np

# ==========================================
# PHẦN A: CÁC HÀM BENCHMARK (Cho Task 3.2, 4.2)
# ==========================================
def F1_Sphere(x):
    return np.sum(x ** 2)

def F2_Rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def F3_Rastrigin(x):
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

def get_benchmark_info(func_name):
    """Hàm lấy thông tin cho các bài toán Benchmark toán học"""
    if func_name == "F1":
        return F1_Sphere, -100, 100, 30
    elif func_name == "F2":
        return F2_Rosenbrock, -30, 30, 30
    elif func_name == "F3":
        return F3_Rastrigin, -5.12, 5.12, 30
    else:
        return None, 0, 0, 0

# ==========================================
# PHẦN B: BÀI TOÁN ỨNG DỤNG (SIGNAL SYNC)
# ==========================================
class SignalSyncProblem:
    def __init__(self, num_symbols=100, Fs=20, noise_power=5.0):
        """
        Khởi tạo bài toán:
        1. Sinh tín hiệu gốc
        2. Sinh đáp án bí mật (tau, phi)
        3. Tạo tín hiệu thu R(t)
        """
        self.num_symbols = num_symbols
        self.Fs = Fs
        self.noise_power = noise_power

        # --- 1. Tạo tín hiệu gốc S(t) ---
        symbols = np.random.randint(0, 4, num_symbols)
        phase_map = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
        s_symbols = np.exp(1j * phase_map[symbols])
        self.S_t = np.repeat(s_symbols, Fs)

        # --- 2. Tạo đáp án bí mật ---
        max_delay = len(self.S_t) // 2
        self.tau_true = np.random.randint(0, 50)
        self.phi_true = np.random.uniform(0, 2 * np.pi)

        # --- 3. Tạo tín hiệu thu R(t) ---
        S_delayed = np.roll(self.S_t, self.tau_true)
        S_distorted = S_delayed * np.exp(1j * self.phi_true)

        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(self.S_t)) + 1j * np.random.randn(len(self.S_t)))
        self.R_t = S_distorted + noise

        print(f"\n[Problem Initialized] Target: Tau={self.tau_true}, Phi={self.phi_true:.4f}")

    def get_bounds(self):
        """Trả về giới hạn tìm kiếm [lb, ub]"""
        # Tau: tìm từ 0 đến 100 mẫu (hoặc tùy độ dài)
        # Phi: tìm từ 0 đến 2*pi
        lb = [0, 0]
        ub = [100, 2 * np.pi]
        return lb, ub

    def fitness_function(self, position):
        """
        Hàm chấm điểm. position: [tau_guess, phi_guess]
        """
        tau_guess = int(position[0])
        phi_guess = position[1]

        S_guess_delayed = np.roll(self.S_t, tau_guess)
        cross_corr = np.sum(self.R_t * np.conjugate(S_guess_delayed))
        compensated_corr = cross_corr * np.exp(-1j * phi_guess)
        fitness = -1 * np.real(compensated_corr)

        return fitness