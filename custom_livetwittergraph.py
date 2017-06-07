
"""
Created on Wed Feb  8 19:08:23 2017

@author: Gautham
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use("ggplot")

out = "twitter-feed.txt"
data = open(out, "r").read()
lines = data.split('\n')
query = lines[0]

fig = plt.figure()
fig.canvas.set_window_title('Custom Sentiment Analyzer')
fig.suptitle("Graphing Live Tweets of '{}'" .format(query) + " - Custom Sentiment Analyzer")
ax = fig.add_subplot(1, 1, 1)

def animate(i):
    data = open("twitter-feed.txt", "r").read()
    lines = data.split('\n')

    trend = []
    t = 0
    for line in lines[1:]:
        if "pos" in line:
            t += 1
        elif "neg" in line:
            t -= 1
        trend.append(t)
        
    ax.clear()
    ax.plot(trend)
    pos_val = float(lines.count("pos")) / len(lines) * 100
    neg_val = float(lines.count("neg"))  / len(lines) * 100
    ax.text(0.35, 0.9, "Pos: {:.4}%, Neg: {:.4}%" .format(pos_val, neg_val),
            color='red', transform=ax.transAxes)
    #print (pos_val, neg_val)

ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()
