import os

def normalize_array_between(data, new_low, new_high):
    old_range = data.max() - data.min()
    new_range = new_high - new_low
    normalize_array = (((data - data.min()) * new_range) / old_range) + new_low
    return normalize_array


def save_figure(grid, name):
    base = "paper_images"
    path = os.path.join(base, name)
    grid.savefig(path)

