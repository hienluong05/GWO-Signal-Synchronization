import numpy as np
from GWO_standard import GWO          # Import "Bộ não"
from Problem_Model import SignalSyncProblem # Import "Đề bài"

# 1. Khởi tạo bài toán
problem = SignalSyncProblem(num_symbols=100, Fs=20, noise_power=0.1)

# 2. Lấy cấu hình từ bài toán
lb, ub = problem.get_bounds()
dim = 2             # 2 biến: Tau và Phi
pop_size = 50       # 50 con sói
max_iter = 100      # 100 vòng lặp

print("--- Bắt đầu thả sói đi săn ---")

# 3. Khởi tạo GWO
my_gwo = GWO(problem.fitness_function, lb, ub, dim, pop_size, max_iter)

# 4. Chạy tối ưu
best_pos, best_score, curve = my_gwo.optimize()

# 5. So sánh kết quả
tau_est = int(best_pos[0])
phi_est = best_pos[1]

print("\n================ KẾT QUẢ ================")
print(f"{'Tham số':<10} | {'Thực tế (Target)':<20} | {'GWO Tìm được':<20} | {'Sai số'}")
print("-" * 65)
print(f"{'Tau':<10} | {problem.tau_true:<20} | {tau_est:<20} | {abs(problem.tau_true - tau_est)}")
print(f"{'Phi':<10} | {problem.phi_true:<20.4f} | {phi_est:<20.4f} | {abs(problem.phi_true - phi_est):.4f}")
print("=========================================")

# Kiểm tra xem sói có tìm đúng không
if abs(problem.tau_true - tau_est) <= 1 and abs(problem.phi_true - phi_est) < 0.1:
    print(">> THÀNH CÔNG: Sói đã bắt được mồi! (Đồng bộ thành công)")
else:
    print(">> THẤT BẠI: Sói bị lạc hướng. Cần chỉnh lại tham số.")