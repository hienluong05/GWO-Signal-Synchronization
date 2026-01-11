import numpy as np
from igwo import IGWO 

# Hàm Parabol lệch: Min tại (5, -5) với giá trị 25
def fitness_function_offset(position):
    x = position[0]
    y = position[1]
    
    # Tính giá trị
    val = (x - 5)**2 + (y + 5)**2 + 25
    return val

# Cấu hình tìm kiếm
# Lưu ý: Không gian tìm kiếm phải bao trùm được điểm (5, -5)
lb = [-10, -10] 
ub = [10, 10]
dim = 2
pop_size = 50      # Tăng số lượng sói lên chút cho nhanh tìm thấy
max_iter = 100     # Tăng số vòng lặp

# Chạy GWO (Logic MIN)
my_gwo = IGWO(fitness_function_offset, lb, ub, dim, pop_size, max_iter)
best_pos, best_score, _ = my_gwo.optimize()

print("\n=== KẾT QUẢ KIỂM TRA ===")
print(f"Vị trí tìm được : {best_pos}")
print(f"  -> Kỳ vọng    : [ 5. -5.]")
print("-" * 30)
print(f"Giá trị Min     : {best_score:.5f}")
print(f"  -> Kỳ vọng    : 25.00000")