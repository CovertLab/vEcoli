import matplotlib.pyplot as plt
import seaborn as sns


HIGHLIGHT_BLUE = (0, 0.4, 1)


def prettify_axis(
    ax,
    title_fontsize=12,
    label_fontsize=10,
    ticklabel_fontsize=8,
    tick_format_x="{:.1f}",
    tick_format_y="{:.2f}",
):
    """Prettifies the given axis by doing the following:
    - removes all axis ticks except the first and last
    - de-spines the axis, offsetting the bottom and left axes
    - moves axis labels into the empty space usually occupied by ticks
    - sets all fonts to Arial, font sizes to the given font sizes, bolds the title
    """
    # Restrict ticks to only min and max
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xticks(
        [xmin, xmax],
        labels=[tick_format_x.format(xmin), tick_format_x.format(xmax)],
        fontname="Arial",
    )
    ax.set_yticks(
        [ymin, ymax],
        labels=[tick_format_y.format(ymin), tick_format_y.format(ymax)],
        fontname="Arial",
    )

    ax.tick_params(which="major", labelsize=ticklabel_fontsize)

    # Move axis titles to middle of axis
    ax.set_xticks(
        [(xmin + xmax) / 2], labels=[ax.get_xlabel()], minor=True, fontname="Arial"
    )
    ax.set_yticks(
        [(ymin + ymax) / 2], labels=[ax.get_ylabel()], minor=True, fontname="Arial"
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    ax.tick_params(
        which="minor",
        width=0,
        length=ax.xaxis.get_major_ticks()[0].get_tick_padding(),
        labelsize=label_fontsize,
    )

    # Despine
    sns.despine(ax=ax, offset=3)

    # Need to rotate y label AFTER despining for inexplicable reasons
    ax.yaxis.get_minor_ticks()[0].label.set(rotation=90, va="center")

    # Format title
    ax.set_title(
        ax.get_title(), fontsize=title_fontsize, fontname="Arial", fontweight="bold"
    )

    return ax
