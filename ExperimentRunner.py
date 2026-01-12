import numpy as np

class ExperimentRunner:
    def __init__(self, algorithms, tasks, config):
        self.algorithms = algorithms
        self.tasks = tasks
        self.pop_size = config.get('pop_size', 30)
        self.max_iter = config.get('max_iter', 500)
        self.runs = config.get('runs', 30)
        self.results = {}

    def run(self):
        for task in self.tasks:
            print(f"\n>>> Running Task: {task.name}")
            self.results[task.name] = {}

            # Lấy config và targets (nếu có)
            func, lb, ub, dim = task.get_config()
            targets = task.get_targets() # Lấy đáp án đúng

            for algo_name, (AlgoClass, params) in self.algorithms.items():
                print(f"    Running {algo_name}...", end=" ")

                convergence_matrix = []
                best_scores = []

                # Biến theo dõi nghiệm tốt nhất toàn cục (trong 30 lần chạy)
                global_best_score = float("inf")
                global_best_pos = None

                for _ in range(self.runs):
                    optimizer = AlgoClass(func, lb, ub, dim, self.pop_size, self.max_iter, **params)
                    # optimize trả về: best_pos, best_score, curve
                    run_best_pos, run_best_score, curve = optimizer.optimize()

                    # Padding curve nếu cần
                    if len(curve) < self.max_iter:
                        curve = np.pad(curve, (0, self.max_iter - len(curve)), 'edge')

                    convergence_matrix.append(curve)
                    best_scores.append(run_best_score)

                    # Cập nhật Global Best nếu lần chạy này tốt hơn các lần trước
                    if run_best_score < global_best_score:
                        global_best_score = run_best_score
                        global_best_pos = run_best_pos

                # Lưu kết quả
                self.results[task.name][algo_name] = {
                    "avg_curve": np.mean(convergence_matrix, axis=0),
                    "best_scores": best_scores,
                    "mean": np.mean(best_scores),
                    "std": np.std(best_scores),
                    "best_pos": global_best_pos,
                    "targets": targets
                }
                print(f"Done. (Mean: {np.mean(best_scores):.4e})")

        return self.results