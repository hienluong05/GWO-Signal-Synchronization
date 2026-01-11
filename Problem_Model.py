import numpy as np

class SignalSyncProblem:
    def __init__(self, num_symbols=100, Fs=20, noise_power=0.1):
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
        phase_map = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        s_symbols = np.exp(1j * phase_map[symbols])
        self.S_t = np.repeat(s_symbols, Fs) # Lưu vào self để dùng sau này
        
        # --- 2. Tạo đáp án bí mật ---
        # Giới hạn trễ tối đa là 1/2 độ dài tín hiệu để dễ tìm
        max_delay = len(self.S_t) // 2 
        self.tau_true = np.random.randint(0, 50) 
        self.phi_true = np.random.uniform(0, 2*np.pi)
        
        # --- 3. Tạo tín hiệu thu R(t) ---
        S_delayed = np.roll(self.S_t, self.tau_true)
        S_distorted = S_delayed * np.exp(1j * self.phi_true)
        
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(self.S_t)) + 1j * np.random.randn(len(self.S_t)))
        self.R_t = S_distorted + noise # Lưu vào self
        
        # In đáp án ra để lát nữa so sánh (Debug)
        print(f"\n[Problem Initialized] Target: Tau={self.tau_true}, Phi={self.phi_true:.4f}")

    def get_bounds(self):
        """Trả về giới hạn tìm kiếm cho GWO [lb, ub]"""
        # Tau: tìm từ 0 đến 100 mẫu (hoặc tùy độ dài)
        # Phi: tìm từ 0 đến 2*pi
        lb = [0, 0]
        ub = [100, 2 * np.pi] 
        return lb, ub

    def fitness_function(self, position):
        """
        Hàm chấm điểm cho con sói.
        position: [tau_guess, phi_guess]
        """
        tau_guess = int(position[0])
        phi_guess = position[1]
        
        # 1. Tái tạo trễ (trên tín hiệu gốc S_t lưu trong self)
        S_guess_delayed = np.roll(self.S_t, tau_guess)
        
        # 2. Tính tương quan chéo (Cross-Correlation)
        # Công thức: Sum(R * conj(S_guess))
        cross_corr = np.sum(self.R_t * np.conjugate(S_guess_delayed))
        
        # 3. Bù pha (Phase Compensation) - Logic cực quan trọng đã sửa
        compensated_corr = cross_corr * np.exp(-1j * phi_guess)
        
        # 4. Lấy phần thực và đổi dấu (Max Correlation -> Min Fitness)
        fitness = -1 * np.real(compensated_corr)
        
        return fitness