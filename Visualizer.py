import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os

class Visualizer:
    def __init__(self, results_data, save_dir="report"):
        self.results = results_data
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def export_csv(self, filename="benchmark_report.csv"):
        rows = []
        for task_name, algos in self.results.items():
            for algo_name, metrics in algos.items():
                row_data = {
                    "Task": task_name,
                    "Algorithm": algo_name,
                    "Mean Fitness": metrics["mean"],
                    "Std Dev": metrics["std"],
                    "Best Fitness": np.min(metrics["best_scores"])
                }

                best_pos = metrics["best_pos"]
                targets = metrics.get("targets", {})

                if task_name == "SignalSyncProblem":
                    row_data["Pred Tau"] = int(best_pos[0])
                    row_data["True Tau"] = targets.get("True Tau", "-")

                    row_data["Pred Phi"] = round(best_pos[1], 8)
                    row_data["True Phi"] = round(targets.get("True Phi", 0), 8)

                    if targets:
                        row_data["Error Tau"] = abs(row_data["Pred Tau"] - row_data["True Tau"])
                        row_data["Error Phi"] = abs(row_data["Pred Phi"] - row_data["True Phi"])
                else:
                    short_pos = str(np.round(best_pos[:3], 4)) + "..."
                    row_data["Best Position"] = short_pos

                rows.append(row_data)

        df = pd.DataFrame(rows)
        file_path = os.path.join(self.save_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"[Visualizer] Đã xuất CSV chi tiết: {file_path}")

    def plot_convergence(self):
        print("[Visualizer] Đang vẽ Convergence curve...")
        for task_name, algos in self.results.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for algo_name, metrics in algos.items():
                curve = metrics["avg_curve"]
                if "Signal" not in task_name:
                    curve = np.maximum(curve, 1e-300)
                ax.plot(curve, label=algo_name, linewidth=2)

            ax.set_title(f"Convergence: {task_name}", fontsize=14)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Fitness")
            ax.grid(True, alpha=0.3)
            ax.legend()

            if "Signal" not in task_name:
                ax.set_yscale("log")
            else:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            clean_name = task_name.replace(" ", "_")
            plt.savefig(os.path.join(self.save_dir, f"Convergence_{clean_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    def plot_boxplot(self):
        print("[Visualizer] Đang vẽ Boxplot...")
        for task_name, algos in self.results.items():
            data = []
            labels = []

            for algo_name, metrics in algos.items():
                scores = np.array(metrics["best_scores"])
                if "Signal" not in task_name:
                    scores = np.maximum(scores, 1e-300)
                data.append(scores)
                labels.append(algo_name)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data, labels=labels, patch_artist=True)

            ax.set_title(f"Stability Analysis: {task_name}", fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=15)

            if "Signal" not in task_name:
                ax.set_yscale("log")
            else:
                y_formatter = ticker.ScalarFormatter(useOffset=False)
                ax.yaxis.set_major_formatter(y_formatter)

            clean_name = task_name.replace(" ", "_")
            plt.savefig(os.path.join(self.save_dir, f"Boxplot_{clean_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()