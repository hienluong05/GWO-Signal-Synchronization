import numpy as np

class GWO:
    def __init__(self, objective_function, lb, ub, dim, pop_size, max_iter):
        self.func = objective_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.positions = np.random.uniform(0, 1, (pop_size, dim)) * (self.ub - self.lb) + self.lb
        
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        
    def optimize(self):
        convergence_curve = []
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
                    
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.pop_size):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A_alpha = 2 * a * r1 - a
                C_alpha = 2 * r2
                dis_to_alpha = abs(C_alpha * self.alpha_pos - self.positions[i])
                new_pos_follow_alpha = self.alpha_pos - A_alpha * dis_to_alpha
                    
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A_beta = 2 * a * r1 - a
                C_beta = 2 * r2
                dis_to_beta = abs(C_beta * self.beta_pos - self.positions[i])
                new_pos_follow_beta = self.beta_pos - A_beta * dis_to_beta
                    
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A_delta = 2 * a * r1 - a
                C_delta = 2 * r2
                dis_to_delta = abs(C_delta * self.delta_pos - self.positions[i])
                new_pos_follow_delta = self.delta_pos - A_delta * dis_to_delta
                    
                new_pos = (new_pos_follow_alpha + new_pos_follow_beta + new_pos_follow_delta) / 3
                self.positions[i] = new_pos
                    
            print(f"Iter {t}: Best Fitness = {self.alpha_score:.5f}")
            convergence_curve.append(self.alpha_score)
        
        return self.alpha_pos, self.alpha_score, convergence_curve
            
                
        