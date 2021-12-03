import matplotlib.pyplot as plt


def reposition_ticks(ax, x="bottom", y="left"):
    if x == "top":
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    elif x == "bottom":
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
    else:
        raise ValueError(f"{x} is not a valid place for x-ticks")

    if y == "left":
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')
    elif y == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    else:
        raise ValueError(f"{y} is not a valid place for y-ticks")