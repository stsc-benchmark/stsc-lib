from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class ResultsInfoPlotOptions:
    directory_path: str
    clear_directory: bool = False
    show_titles: bool = True
    plot_all_test: bool = False


@dataclass
class TestDataset:
    name: str
    data: Union[np.ndarray,List[np.ndarray]]
    start_indices: Optional[List[int]] = None