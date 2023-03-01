import numpy as np


def histogram(angles, magnitudes):
    h = np.zeros(10, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            angles[i, j] = 160
            index_1 = int(angles[i, j] // 20)
            index_2 = int(angles[i, j] // 20 + 1)

            proportion = (index_2 * 20 - angles[i, j]) / 20

            value_1 = proportion * magnitudes[i, j]
            value_2 = (1 - proportion) * magnitudes[i, j]

            h[index_1] += value_1
            h[index_2] += value_2

    h[0] += h[-1]
    return h[0:9]


def make_cells(angles, magnitudes, cell_size):
    cells = []
    for i in range(0, np.shape(angles)[0], cell_size):
        row = []
        for j in range(0, np.shape(angles)[1], cell_size):
            row.append(np.array(
                histogram(angles[i:i + cell_size, j:j + cell_size], magnitudes[i:i + cell_size, j:j + cell_size]),
                dtype=np.float32))
        cells.append(row)

    return np.array(cells, dtype=np.float32)


def make_blocks(block_size, cells):
    before = int(block_size / 2)
    after = int(block_size / 2)

    if block_size % 2 != 0:
        after = after + 1

    first_stop = before
    second_stop = before

    if np.shape(cells)[0] % block_size == 0:
        first_stop = first_stop - 1

    if np.shape(cells)[1] % block_size == 0:
        second_stop = second_stop - 1

    blocks = []
    for i in range(int(block_size / 2.0), np.shape(cells)[0] - first_stop):
        for j in range(int(block_size / 2.0), np.shape(cells)[1] - second_stop):
            blocks.append(np.array(cells[i - before:i + after, j - before:j + after].flatten()))

    return blocks


def normalize_L1(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L1_sqrt(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return np.sqrt(block / norm)
    else:
        return block


def normalize_L2(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L2_Hys(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)

    if norm != 0:
        block_aux = block / norm
        block_aux[block_aux > 0.2] = 0.2
        norm = np.sqrt(np.sum(block_aux * block_aux) + threshold * threshold)
        if norm != 0:
            return block_aux / norm
        else:
            return block_aux
    else:
        return block


def normalize(block, type_norm, threshold=0):
    if type_norm == 0:
        return normalize_L2(block, threshold)
    elif type_norm == 1:
        return normalize_L2_Hys(block, threshold)
    elif type_norm == 2:
        return normalize_L1(block, threshold)
    elif type_norm == 3:
        return normalize_L1_sqrt(block, threshold)
