"""
|___     ___     ___  |
|| |     | |     | |  |
|| | --> | | --> | |  |
|| |     | |     | |  |
|^^^     ^^^     ^^^  |
|                     |
"""

from functools import partial
from math import *
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
import numpy as np
import soundfile as sf
import sys

sample_rate = 44100

block_count = 200
seconds = 2
initial_force = 10
initial_force_duration = .1
width = .00001
initial_separation = width 
energy_loss = 1e-11
block_centers = [width / 2 + width * (i - 1) + initial_separation * (i - 1) for i in range(1, block_count + 1)]
max_dist = block_centers[-1] + width / 2 + initial_separation
velocities = [0 for i in range(block_count)]
masses = [.01 for i in range(block_count)]

#movement_recording = [[] for i in range(block_count)]
movement_recording = []
velocity_recording = [[] for i in range(block_count)]

time_delta = 1 / sample_rate

plt.ion()
fig, ax = plt.subplots(figsize=(16, 9))

def plot_func(frame, *fargs):
    lines = []
    for center in block_centers:
        x = [center - width / 2, center - width / 2, center + width / 2, center + width / 2, center - width / 2]
        y = [0, width, width, 0, 0]
        ax.plot(x, y, color="#7777ff") 

def init_func():
    lines = []
    ax.plot([0, 0], [-width, width], linewidth=5, color="#000000")
    ax.plot([max_dist, max_dist], [-width, width], linewidth=5, color="#000000")

    for center in block_centers:
        x = [center - width / 2, center - width / 2, center + width / 2, center + width / 2, center - width / 2]
        y = [-width / 2, width / 2, width / 2, -width / 2, -width / 2]
        lines.append(ax.plot(x, y, color="#7777ff", linewidth=3)[0])

def calculate_velocities(one: int, two: int):
    """
    Given two indices, calculate the new velocities
    """
    v1 = (masses[one] - masses[two]) / (masses[one] + masses[two]) * velocities[one] + 2 * masses[two] / (masses[one] + masses[two]) * velocities[two]
    v2 = 2 * masses[one] / (masses[one] + masses[two]) * velocities[one] + (masses[two] - masses[one]) / (masses[one] + masses[two]) * velocities[two]
    #print("V1: {}\nV2: {}".format(v1, v2))
    return v1, v2


for i in range(int(round(sample_rate * seconds))):
    movement_recording.append(block_centers[0])
    if i / sample_rate <= initial_force_duration:
        # If the elapsed time is LEQ the force duration
        velocities[0] += initial_force / masses[0] * time_delta

    print(f"{i / int(round(sample_rate * seconds)):.2f}    ", end="\r")
    sys.stdout.flush()
    for block_index in range(len(block_centers)):
        #movement_recording[block_index].append(block_centers[block_index])
        """
        For each block:
        1. Reduce abs(velocity) by some fraction
        2. Update position with velocity
        3. Detect any collisions in the direction of travel
           - Calculate new velocities of the two involved blocks
        """
        velocity_recording[block_index].append(velocities[block_index])
        collision_occurred = False
        perform_update = True # Survive the gauntlet of collision tests

        # Check if next step will result in collision; calculate the new position of the block centers
        change = velocities[block_index] * time_delta
        if block_index + 1 < block_count - 1:
            if block_centers[block_index] + change >= block_centers[block_index + 1] - width:
                # If the new position of the block exceeds the current position, set current block 
                block_centers[block_index] = block_centers[block_index + 1] - width
                perform_update = False
                collision_occurred = True
                left = block_index
                right = block_index + 1
                
        if block_index > 0:
            if block_centers[block_index] + change <= block_centers[block_index - 1] + width:
                # If the new position of the block exceeds the boundary of the left adjacent block
                block_centers[block_index] = block_centers[block_index - 1] + width
                perform_update = False
                collision_occurred = True
                left = block_index - 1
                right = block_index

        if block_index == 0:
            if block_centers[block_index] + change - width / 2 <= 0:
                block_centers[block_index] = width / 2
                velocities[block_index] = -velocities[block_index]
                perform_update = False

        if block_index == block_count - 1:
            if block_centers[block_index] + change + width / 2 >= max_dist:
                block_centers[block_index] = max_dist - width / 2
                velocities[block_index] = -velocities[block_index]
                perform_update = False

        if perform_update:
            block_centers[block_index] += change

        perform_update = True

        if collision_occurred:
            #print("Block-on-block Collision occurred")
            #midpoint = (block_centers[left] + block_centers[right]) / 2
            initial_force = 0
            velocities[left], velocities[right] = calculate_velocities(left, right)
            collision_occurred = False
        velocities[block_index] *= (1 - sqrt(energy_loss))
#init_func()

"""ax.plot(position / np.max(position))
ax.plot(vel / np.max(vel))
print(min(vel))"""

position = np.array(movement_recording)
vel = np.array(velocity_recording)

if len(position.shape) > 1:
    shifted = position[0] - np.median(position[0])
else:
    shifted = position - np.median(position)
zd = (shifted - np.mean(shifted)) / np.std(shifted)
norm = zd / np.max(np.abs(zd))
ax.plot(norm)
sf.write("asdf.wav", norm, samplerate=sample_rate)
