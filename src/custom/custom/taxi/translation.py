def encode_taxi1P(taxi_row, taxi_col, pass_loc, dest_idx):
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i


def decode_taxi1P(state):
    out = []
    out.append(state % 4)
    state = state // 4
    out.append(state % 5)
    state = state // 5
    out.append(state % 5)
    state = state // 5
    out.append(state)
    assert 0 <= state < 5
    return reversed(out)


def encode_taxi2P(taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2):
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc1
    i *= 5
    i += pass_loc2
    i *= 4
    i += dest_idx1
    i *= 4
    i += dest_idx2
    return i


def decode_taxi2P(state):
    out = []
    out.append(state % 4)
    state = state // 4
    out.append(state % 4)
    state = state // 4
    out.append(state % 5)
    state = state // 5
    out.append(state % 5)
    state = state // 5
    out.append(state % 5)
    state = state // 5
    out.append(state)
    assert 0 <= state < 5
    return reversed(out)

def translate(state, person):
    taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2 = decode_taxi2P(state)
    if person == 1:
        pass_loc = pass_loc1
        dest_idx = dest_idx1
    else:
        pass_loc = pass_loc2
        dest_idx = dest_idx2
    return encode_taxi1P(taxi_row, taxi_col, pass_loc, dest_idx)



