from mplsoccer import Pitch
import matplotlib.pyplot as plt

pitch = Pitch(pitch_type='opta')  # example plotting an Opta/ Stats Perform pitch
fig, ax = pitch.draw()

plt.show()