import seaborn as sns


HIGHLIGHT_BLUE = (0, 0.4, 1)
HIGHLIGHT_RED = (0.678, 0, 0.125)
BG_GRAY = "0.4"


def prettify_axis(
    ax,
    xlim=None,
    ylim=None,
    title_fontsize=12,
    label_fontsize=9,
    ticklabel_fontsize=8,
    tick_format_x="{:.1f}",
    tick_format_y="{:.2f}",
    xticks="minimal",
    yticks="minimal",
    xlabel_as_tick=True,
    ylabel_as_tick=True,
):
    """Prettifies the given axis by doing the following:
    - removes all axis ticks except the first and last
    - de-spines the axis, offsetting the bottom and left axes
    - moves axis labels into the empty space usually occupied by ticks
    - sets all fonts to Arial, font sizes to the given font sizes, bolds the title
    """
    # Set axis limits
    xmin, xmax = xlim if xlim is not None else ax.get_xlim()
    ymin, ymax = ylim if ylim is not None else ax.get_ylim()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Set and format axis ticks
    if isinstance(xticks, str):
        if xticks == "minimal":
            # Restrict ticks to only min and max
            xticks = [xmin, xmax]
        elif xticks == "all":
            # take all xticks (after setting axis limits)
            xticks = ax.get_xticks()
        else:
            raise ValueError(
                f"{xticks} is not a valid value for xticks (must be 'minimal', 'all', or a list of ticks.)"
            )

    if isinstance(yticks, str):
        if yticks == "minimal":
            # Restrict ticks to only min and max
            yticks = [ymin, ymax]
        elif yticks == "all":
            # take all xticks (after setting axis limits)
            yticks = ax.get_yticks()
        else:
            raise ValueError(
                f"{yticks} is not a valid value for yticks (must be 'minimal', 'all', or a list of ticks.)"
            )

    ax.set_xticks(
        xticks,
        labels=[tick_format_x.format(tick) for tick in xticks],
        fontname="Arial",
    )

    ax.set_yticks(
        yticks,
        labels=[tick_format_y.format(tick) for tick in yticks],
        fontname="Arial",
    )

    ax.tick_params(which="major", labelsize=ticklabel_fontsize)

    # Move axis titles to middle of axis
    if xlabel_as_tick:
        ax.set_xticks(
            [(xmin + xmax) / 2], labels=[ax.get_xlabel()], minor=True, fontname="Arial"
        )
        ax.set_xlabel(None)

        ax.tick_params(
            which="minor",
            width=0,
            length=ax.xaxis.get_major_ticks()[0].get_tick_padding(),
            labelsize=label_fontsize,
        )

    if ylabel_as_tick:
        ax.set_yticks(
            [(ymin + ymax) / 2], labels=[ax.get_ylabel()], minor=True, fontname="Arial"
        )
        ax.set_ylabel(None)

        ax.tick_params(
            which="minor",
            width=0,
            length=ax.xaxis.get_major_ticks()[0].get_tick_padding(),
            labelsize=label_fontsize,
        )

    # Despine
    sns.despine(ax=ax, offset=3)

    if ylabel_as_tick:
        # Need to rotate y label AFTER despining for inexplicable reasons
        ax.yaxis.get_minor_ticks()[0].label.set(rotation=90, va="center")

    # Format title
    ax.set_title(
        ax.get_title(), fontsize=title_fontsize, fontname="Arial", fontweight="bold"
    )

    return ax
