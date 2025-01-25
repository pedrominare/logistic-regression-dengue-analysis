import tracemalloc
import pandas as pd
from pandas import DataFrame


class MemoryMonitor:
    def __init__(self):
        tracemalloc.start()
        self.memory_usage = []
        self.checkpoints = []
        self.df_memory_usage = DataFrame

    # Função para adicionar checkpoints
    def add_checkpoint(self, name: str):
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage.append(
            {
                "Checkpoint": name,
                "Current (MB)": current / 10**6,
                "Peak (MB)": peak / 10**6,
            }
        )

    def stop_checkpoint(self):
        tracemalloc.stop()
        self.df_memory_usage = pd.DataFrame(self.memory_usage)
        print(self.df_memory_usage)
