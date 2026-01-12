# --- IMPORT MODULES ---
from standard.gwo_standard import GWO
from variants.chaotic_gwo_variant.chaoticgwo import ChaoticGwo
from variants.hybird_gwo_pso_variant.hybridgwo import HybridGwo
from variants.igwo_variant.igwo import IGWO

from ExperimentRunner import ExperimentRunner
from Visualizer import Visualizer
from TaskWrapper import MathTask, SignalTask

# --- CONFIGURATION ---
CONFIG = {
    "pop_size": 30,
    "max_iter": 500,
    "runs": 20
}

# 1. Đăng ký các thuật toán tham chiến
algorithms = {
    "Standard GWO": (GWO, {}),
    "Chaotic GWO (Logistic)": (ChaoticGwo, {"map_type": "logistic"}),
    "Chaotic GWO (Tent)": (ChaoticGwo, {"map_type": "tent"}),
    "Hybrid GWO-PSO": (HybridGwo, {}),
    "IGWO": (IGWO, {})
}

# 2. Đăng ký các bài toán cần giải
tasks = [
    MathTask("F1"),
    MathTask("F2"),
    MathTask("F3"),
    SignalTask()  # Bài toán thực tế
]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 3. Khởi tạo Engine chạy thực nghiệm
    runner = ExperimentRunner(algorithms, tasks, CONFIG)

    # 4. Chạy và lấy kết quả
    results = runner.run()

    # 5. Hiển thị kết quả (View)
    viz = Visualizer(results, save_dir="report")
    viz.export_csv()
    viz.plot_convergence()
    viz.plot_boxplot()

    print("\n[DONE] Đã hoàn thành! Kiểm tra thư mục 'report' để lấy số liệu.")