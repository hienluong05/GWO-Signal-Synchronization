import numpy as np

class GWO_hybrid:
    def __init__(self, objective_function, lb, ub, dim, pop_size, max_iter):
        # Giống GWO gốc
        self.func = objective_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.positions = np.random.uniform(0, 1, (pop_size, dim)) * (self.ub - self.lb) + self.lb
        
        # Phần đặc trưng của hybrid: vận tốc
        self.velocity = np.zeros((pop_size, dim))
        
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        
    def optimize(self):
        convergence_curve = []
        
        # Cấu hình hằng số cho công thức lai ghép
        # Hệ số gia tốc của alpha, beta, delta
        c1 = 0.5
        c2 = 0.5
        c3 = 0.5
        
        for t in range(self.max_iter):
            self.positions = np.clip(self.positions, self.lb, self.ub)
            
            for i in range(self.pop_size):
                fitness = self.func(self.positions[i])
                
                if fitness < self.alpha_score:
                    self.delta_pos = self.beta_pos.copy()
                    self.delta_score = self.beta_score
                    
                    self.beta_pos = self.alpha_pos.copy()
                    self.beta_score = self.alpha_score
                    
                    self.alpha_pos = self.positions[i].copy()
                    self.alpha_score = fitness
                    
                elif fitness < self.beta_score:
                    self.delta_pos = self.beta_pos.copy()
                    self.delta_score = self.beta_score
                    
                    self.beta_pos = self.positions[i].copy()
                    self.beta_score = fitness
                    
                if fitness < self.delta_score:
                    self.delta_pos = self.positions[i].copy()
                    self.delta_score = fitness
            
            # Cấu hình trọng số quán tính cho sói (giúp chúng giữ đà đi cũ)    
            # w giảm tuyến tính từ 0.9 xuống 0.4 theo chuẩn PSO   
            w = 0.9 - 0.5 * (t / self.max_iter)
              
            for i in range(self.pop_size):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                r3 = np.random.random(self.dim)
                term1 = c1 * r1 * (self.alpha_pos - self.positions[i])
                term2 = c2 * r2 * (self.beta_pos - self.positions[i])
                term3 = c3 * r3 * (self.delta_pos - self.positions[i])
                
                V_old = self.velocity[i]
                V_new = w * V_old + term1 + term2 + term3
                self.velocity[i] = V_new
                
                pos_old = self.positions[i]
                pos_new = pos_old + V_new
                self.positions[i] = pos_new

            convergence_curve.append(self.alpha_score)
            
        return self.alpha_pos, self.alpha_score, convergence_curve