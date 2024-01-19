def encode_taxi1P(taxi_row, taxi_col, pass_loc, dest_idx):
    """
    Taxi1PEnv.encode() returns a state from a tuple of (taxi_row, taxi_col, pass_loc, dest_idx)

    :param taxi_row: taxi row
    :param taxi_col: taxi column
    :param pass_loc: passenger location
    :param dest_idx: destination location

    :return: state
    """
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i


def decode_taxi1P(state):
    """
    Taxi1PEnv.decode() returns a tuple of (taxi_row, taxi_col, pass_loc, dest_idx) from a state

    :param state: state to decode
    :return: taxi_row, taxi_col, pass_loc, dest_idx
    """
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
    """
    Taxi2PEnv.encode() returns a state from a tuple of (taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2)

    :param taxi_row: taxi row
    :param taxi_col: taxi column
    :param pass_loc1: passenger 1 location
    :param pass_loc2: passenger 2 location
    :param dest_idx1: destination 1 location
    :param dest_idx2: destination 2 location
    :return: state
    """
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
    """
    Taxi2PEnv.decode() returns a tuple of (taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2) from a state

    :param state: state to decode
    :return: taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2
    """
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
    """
    Translates a state from Taxi2PEnv to Taxi V3. It does so by ignoring the other person's location and destination.


    :param state: state to translate
    :param person: person to translate the state for

    :return: translated state
    """
    taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2 = decode_taxi2P(state)
    if person == 1:
        pass_loc = pass_loc1
        dest_idx = dest_idx1
    else:
        pass_loc = pass_loc2
        dest_idx = dest_idx2
    return encode_taxi1P(taxi_row, taxi_col, pass_loc, dest_idx)



