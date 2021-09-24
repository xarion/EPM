import os

def normalize_array_between(data, new_low, new_high):
    old_range = data.max() - data.min()
    new_range = new_high - new_low
    normalize_array = (((data - data.min()) * new_range) / old_range) + new_low
    return normalize_array


def save_figure(grid, name, base):

    assert(base in ["results", "ood"]), "base can be 'results' or 'ood'"
    pre = ""

    if base == "results":
        pre = "paper_images"
        path = os.path.join(pre, name)
        grid.savefig(path)

    elif base == "ood":
        pre = "ood_images"
        path = os.path.join(pre, name)
        grid.save(path)