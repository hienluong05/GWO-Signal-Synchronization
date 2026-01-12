import numpy as np
from Problem_Model import get_benchmark_info, SignalSyncProblem

class BenchmarkTask:
    def __init__(self, name):
        self.name = name
    
    def get_config(self):
        raise NotImplementedError

    def get_targets(self):
        return {} # Mặc định không có target (cho các hàm toán học)

class MathTask(BenchmarkTask):
    def get_config(self):
        func, lb, ub, dim = get_benchmark_info(self.name)
        return func, list(np.full(dim, lb)), list(np.full(dim, ub)), dim

class SignalTask(BenchmarkTask):
    def __init__(self):
        super().__init__("SignalSyncProblem")
        # Giữ 1 instance cố định để tất cả thuật toán cùng giải 1 bài toán giống nhau
        self.problem_instance = SignalSyncProblem()
        
    def get_config(self):
        # Trả về hàm fitness của instance đã tạo
        lb, ub = self.problem_instance.get_bounds()
        return self.problem_instance.fitness_function, lb, ub, 2

    def get_targets(self):
        return {
            "True Tau": self.problem_instance.tau_true,
            "True Phi": self.problem_instance.phi_true
        }