
from utils import CSI_Loader
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox


def plot_interactive(mag, samples_per_bin, sr, label):
    """3-panel interactive plot: magnitude, empty bins, overpopulated bins.

    Controls:
      - Center slider + text box: position the window center (seconds)
      - Window length text box: set visible window width (seconds)
      - Screenshot button: save current view to PNG
    """
    from matplotlib.widgets import Slider, TextBox, Button

    n = mag.shape[0]
    time = np.arange(n) / sr
    max_t = time[-1]

    # Aggregate every 10 bins for bar plots
    agg = 10
    n_bins = len(samples_per_bin)
    n_bars = n_bins // agg
    spb_trimmed = samples_per_bin[:n_bars * agg].reshape(n_bars, agg)

    empty_counts = (spb_trimmed == 0).sum(axis=1).astype(np.float64)
    overpop_counts = np.clip(spb_trimmed - 1, 0, None).sum(axis=1).astype(np.float64)
    bar_time = (np.arange(n_bars) * agg + agg / 2) / sr
    bar_width = agg / sr

    fig, (ax_mag, ax_empty, ax_over) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True)
    plt.subplots_adjust(bottom=0.22)

    # --- Magnitude ---
    for i in range(mag.shape[1]):
        ax_mag.plot(time, mag[:, i], alpha=0.5, linewidth=0.5)
    ax_mag.set_ylabel('Magnitude')
    ax_mag.set_title(f'{label} — CSI Magnitude (all subcarriers)')
    ax_mag.grid(True, alpha=0.3)

    # --- Empty bins (bar: count of empty per 10 bins) ---
    ax_empty.bar(bar_time, empty_counts, width=bar_width, color='red', alpha=0.7, edgecolor='none')
    ax_empty.set_ylabel(f'Empty / {agg} bins')
    ax_empty.set_title(f'{label} — Empty bins (per {agg} samples)')
    ax_empty.set_ylim(0, agg + 0.5)
    ax_empty.grid(True, alpha=0.3)

    # --- Overpopulated bins (bar: total extra samples per 10 bins) ---
    ax_over.bar(bar_time, overpop_counts, width=bar_width, color='blue', alpha=0.7, edgecolor='none')
    ax_over.set_ylabel(f'Extra samples / {agg} bins')
    ax_over.set_xlabel('Time (s)')
    ax_over.set_title(f'{label} — Overpopulated bins (per {agg} samples)')
    ax_over.grid(True, alpha=0.3)

    # State: mutable container so closures can update
    state = {'center': min(max_t / 2, 50.0), 'win_len': min(100.0, max_t)}
    axes = (ax_mag, ax_empty, ax_over)

    def apply_view():
        c = state['center']
        w = state['win_len']
        lo = max(0, c - w / 2)
        hi = min(max_t, c + w / 2)
        for ax in axes:
            ax.set_xlim(lo, hi)
        fig.canvas.draw_idle()

    apply_view()

    # --- Row 1: Center slider + center text box ---
    ax_slider = fig.add_axes([0.15, 0.09, 0.50, 0.03])
    slider = Slider(ax_slider, 'Center (s)', 0, max_t,
                    valinit=state['center'], valstep=0.5)

    ax_center_txt = fig.add_axes([0.72, 0.09, 0.08, 0.03])
    tb_center = TextBox(ax_center_txt, '', initial=f"{state['center']:.1f}")

    # --- Row 2: Window length label + text box + Screenshot button ---
    ax_wl_txt = fig.add_axes([0.15, 0.03, 0.08, 0.03])
    tb_winlen = TextBox(ax_wl_txt, 'Win (s)', initial=f"{state['win_len']:.1f}")

    ax_btn = fig.add_axes([0.72, 0.03, 0.12, 0.03])
    btn_screenshot = Button(ax_btn, 'Screenshot')
    _shot_count = [0]

    def update_from_slider(val):
        state['center'] = float(slider.val)
        tb_center.set_val(f"{state['center']:.1f}")
        apply_view()

    def update_center_text(text):
        try:
            v = max(0, min(float(text), max_t))
        except ValueError:
            return
        state['center'] = v
        slider.set_val(v)
        apply_view()

    def update_winlen_text(text):
        try:
            v = max(0.1, min(float(text), max_t))
        except ValueError:
            return
        state['win_len'] = v
        apply_view()

    def on_screenshot(event):
        _shot_count[0] += 1
        c, w = state['center'], state['win_len']
        fname = f'{label}_c{c:.0f}_w{w:.0f}_{_shot_count[0]}.png'
        fig.savefig(fname, dpi=150)
        print(f'Saved {fname}')

    slider.on_changed(update_from_slider)
    tb_center.on_submit(update_center_text)
    tb_winlen.on_submit(update_winlen_text)
    btn_screenshot.on_clicked(on_screenshot)

    plt.show()


if __name__ == "__main__":
    
    TRAIN_DIR = '../../../wifi_sensing_data/har_data/train'
    TEST_DIR = '../../../wifi_sensing_data/har_data/test'

    labels = ['drink']

    
    loader = CSI_Loader(verbose=True)
    for i, label in enumerate(labels): 
        data_dict = loader.process(os.path.join(TRAIN_DIR, label + '.csv'))    
        mag = data_dict['mag']
        samples_per_bin = data_dict['samples_per_bin']

        plot_interactive(mag, samples_per_bin, 150, label)

