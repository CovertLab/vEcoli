import os
import shutil
import copy
import concurrent.futures

import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivarium.core.composition import TEST_OUT_DIR
from vivarium.library.units import Quantity, units
from vivarium.plots.agents_multigen import plot_agents_multigen

from ecoli.composites.environment.lattice import test_lattice
from ecoli.plots.snapshots import (
    make_snapshots_figure,
    make_tags_figure,
    format_snapshot_data,
    get_field_range,
    get_agent_colors,
    get_tag_ranges
)

DEFAULT_HIGHLIGHT_COLOR = [0, 1, 1]
PLOT_WIDTH = 7


def make_snapshot_saver(
    multibody_agent_colors,
    multibody_field_range,
    bounds,
    images_dir,
    **kwargs
):
    # Must be global for multiprocessing to work
    global save_snapshot_figure
    def save_snapshot_figure(data_at_time):
        time = data_at_time[0]
        fig_path = os.path.join(images_dir, f"img{time}.jpg")
        multibody_agents, multibody_fields = format_snapshot_data(
            {time: data_at_time[1]})
        fig = make_snapshots_figure(
            agents=multibody_agents,
            fields=multibody_fields,
            bounds=bounds,
            n_snapshots=1,
            time_indices=[time],
            snapshot_times=[time],
            agent_colors=multibody_agent_colors,
            field_range=multibody_field_range,
            default_font_size=12,
            plot_width=PLOT_WIDTH,
            scale_bar_length=0,
            **kwargs)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
        return fig_path
    return save_snapshot_figure


def make_tags_saver(
    agent_colors,
    tag_ranges,
    tag_colors,
    bounds,
    images_dir,
    convert_to_concs=False,
    **kwargs
):
    # Must be global for multiprocessing to work
    global save_tags_figure
    def save_tags_figure(data_at_time):
        time = data_at_time[0]
        fig_path = os.path.join(images_dir, f"img{time}.jpg")
        agents = {time: data_at_time[1].get('agents', {})}
        fig = make_tags_figure(
            time_indices=[time],
            snapshot_times=[time],
            agents=agents,
            agent_colors=agent_colors,
            tag_ranges=tag_ranges,
            tag_colors=tag_colors,
            n_snapshots=1,
            bounds=bounds,
            default_font_size=12,
            plot_width=PLOT_WIDTH,
            scale_bar_length=0,
            convert_to_concs=convert_to_concs,
            **kwargs)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
        return fig_path
    return save_tags_figure


def make_timeseries_saver(
    data,
    show_timeseries,
    highlight_agents,
    highlight_color,
    images_dir
):
    # Must be global for multiprocessing to work
    global save_timeseries_figure
    def save_timeseries_figure(t_index):
        time_vec = list(data.keys())

        plot_settings = {
            'column_width': 6,
            'row_height': 2,
            'stack_column': True,
            'tick_label_size': 10,
            'linewidth': 2,
            'title_size': 10}

        if show_timeseries:
            plot_settings.update({'include_paths': show_timeseries})

        # remove agents not included in highlight_agents
        if highlight_agents:
            for time, state in data.items():
                agents = state['agents']
                for agent_id, agent_state in agents.items():
                    if agent_id not in highlight_agents:
                        del data[time]['agents'][agent_id]
            agent_colors = {agent_id: highlight_color for agent_id in highlight_agents}
            plot_settings.update({'agent_colors': agent_colors})

        fig_path = os.path.join(images_dir, f"timeseries{t_index}.jpg")
        time_indices = np.array(range(0, t_index+1))
        current_data = {
            time_vec[index]: data[time_vec[index]]
            for index in time_indices}
        fig = plot_agents_multigen(current_data, dict(plot_settings))
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
        return fig_path
    return save_timeseries_figure


def video_from_images(img_paths, out_file):
    # make the video
    img_array = []
    size = None
    for img_file in img_paths:
        img = cv2.imread(img_file)
        height, width, layers = img.shape
        if size:
            if width < size[0]:
                size[0] = width
            if height < size[0]:
                size[1] = height
        else:
            size = [width, height]
        img_array.append(img)

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(img_array)):
        # Crop all images to smallest size to avoid frame skips
        img_array[i] = img_array[i][0:size[1], 0:size[0]]
        out.write(img_array[i])
    out.release()


def make_video(
        data,
        bounds,
        plot_type='fields',
        step=1,
        highlight_agents=None,
        show_timeseries=None,
        highlight_color=DEFAULT_HIGHLIGHT_COLOR,
        out_dir='out',
        filename='snapshot_vid',
        cpus=1,
        **kwargs
):
    """Make a video with snapshots across time

    Args:
        plot_type: (str) select either 'fields' or 'tags'. 'fields' is the default
    """
    # Remove last timestep since data may be empty
    data = dict(list(data.items())[:-1])
    highlight_agents = highlight_agents or []
    show_timeseries = show_timeseries or []

    # Strip units from bounds if present.
    if isinstance(bounds[0], Quantity):
        bounds = tuple(bound.to(units.um).magnitude for bound in bounds)

    # make images directory, remove if existing
    out_file = os.path.join(out_dir, f'{filename}.mp4')
    out_file2 = os.path.join(out_dir, f'{filename}_timeseries.mp4')
    images_dir = os.path.join(out_dir, f'_images_{plot_type}')
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    agent_colors = {}
    if highlight_agents:
        agent_colors = {
            agent_id: highlight_color
            for agent_id in highlight_agents}

    # get the single snapshots function
    multibody_agents, multibody_fields = format_snapshot_data(data)
    time_vec = list(multibody_agents.keys())
    if plot_type == 'fields':
        multibody_field_range = get_field_range(multibody_fields, time_vec)
        multibody_agent_colors = get_agent_colors(multibody_agents)
        multibody_agent_colors.update(agent_colors)

        do_plot = make_snapshot_saver(
            multibody_agent_colors, 
            multibody_field_range, 
            bounds,
            images_dir,
            **kwargs)

    elif plot_type == 'tags':
        time_indices = np.array(range(0, len(time_vec)))
        tag_ranges, tag_colors = get_tag_ranges(
            agents=multibody_agents,
            tagged_molecules=kwargs.get('tagged_molecules', None),
            time_indices=time_indices,
            convert_to_concs=kwargs.get('convert_to_concs', False),
            tag_colors=kwargs.get('tag_colors', {}))

        do_plot = make_tags_saver(
            agent_colors, 
            tag_ranges,
            tag_colors, 
            bounds,
            images_dir,
            **kwargs)

    # Only plot data for every `step` timepoints
    if step != 1:
        filtered_data = {}
        time_counter = 0
        for timepoint in time_vec:
            if time_counter % step == 0:
                filtered_data[timepoint] = data[timepoint]
            time_counter += 1
        data = filtered_data

    with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
        img_paths = list(tqdm(executor.map(do_plot, data.items()), total=len(data)))

    img_paths_2 = []
    if show_timeseries:
        timeseries_fun = make_timeseries_saver(
            data,
            show_timeseries=show_timeseries,
            highlight_agents=highlight_agents,
            highlight_color=highlight_color,
            images_dir=images_dir)
        time_indices = list(range(0, len(time_vec)))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            img_paths_2 = list(tqdm(executor.map(
                timeseries_fun, time_indices), total=len(time_indices)))

    # make the video
    video_from_images(img_paths, out_file)
    video_from_images(img_paths_2, out_file2)

    # delete image folder
    shutil.rmtree(images_dir)


def main(total_time=2000, step=60, exchange=False):
    out_dir = os.path.join(TEST_OUT_DIR, 'snapshots_video')
    os.makedirs(out_dir, exist_ok=True)

    # tagged molecules for timeseries
    tagged_molecules = [
        ('boundary', 'width',),
        ('boundary', 'length',),
        ('boundary', 'mass',),
        ('boundary', 'angle',),
    ]
    highlight_agents = ['0', '00', '000']

    # GrowDivide agents
    bounds = [30, 30] * units.um
    n_bins = [20, 20]
    initial_field = np.zeros((n_bins[0], n_bins[1]))
    initial_field[:, -1] = 100
    data = test_lattice(
        exchange=exchange,
        n_agents=3,
        total_time=total_time,
        growth_noise=1e-3,
        bounds=bounds,
        n_bins=n_bins,
        initial_field=initial_field)

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type='fields',
        step=step,
        out_dir=out_dir,
        filename=f"snapshots",
        highlight_agents=highlight_agents,
        show_timeseries=tagged_molecules,
    )

    # make tags video
    make_video(
        data,
        bounds,
        plot_type='tags',
        step=step,
        out_dir=out_dir,
        filename=f"tags",
        highlight_agents=highlight_agents,
        tagged_molecules=tagged_molecules,
        show_timeseries=tagged_molecules,
        background_color='white',
    )


if __name__ == '__main__':
    main(total_time=3000, exchange=False)
