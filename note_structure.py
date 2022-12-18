def get_relative_minor(hz): 
    n = [0, 2, 3, 5, 7, 8, 10]
    minor = get_notes(hz, n)
    return minor 
 
 
def get_relative_major(hz): 
    n = [0, 2, 4, 5, 7, 9, 11]
    return get_notes(hz, n)
 

def get_phrygian_dominant(hz):
    n = [0, 1, 4, 5, 7, 8, 11]
    return get_notes(hz, n)


def get_minor_triad(hz):
    n = [0, 3, 7]
    return get_notes(hz, n)


def get_major_triad(hz):
    return get_notes(hz, [0, 4, 7])


def get_sus_second(hz):
    return get_notes(hz, [0, 2, 7])


def get_power(hz):
    return get_notes(hz, [0, 7])


def get_chromatic(hz):
    return get_notes(hz, [i for i in range(12)])


def get_frequencies(key, hz):
    if key == "major":
        frequencies = get_major_triad(hz)
    elif key == "minor":
        frequencies = get_minor_triad(hz)
    elif key == "sus":
        frequencies = get_sus_second(hz)
    elif key == "power":
        frequencies = get_power(hz)
    elif key == "chromatic":
        frequencies = get_chromatic(hz)
    else: # Default to power chord
        frequencies = get_power(hz)
    return frequencies

def get_notes(hz, n):
    notes = []
    for i in range(-2, 2, 1):
        for j in n:
            notes.append(2 ** i * 2 ** (j / 12) * hz)
    return notes

