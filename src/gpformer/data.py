
def batchify_data(data, batch_size=16):
    batches = []

    for i in range(0, len(data), batch_size):
        if i + batch_size < len(data):
            batches.append(data[i : i + batch_size])

    return batches
