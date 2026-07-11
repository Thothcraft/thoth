(function () {
  function plotConfig() {
    return { responsive: true, displayModeBar: false, displaylogo: false };
  }

  function buildPlaybackLayout(title, xLabel, yLabel, durationMs) {
    const duration = typeof durationMs === 'number' ? durationMs : 120;
    return {
      title: { text: title, font: { size: 15 } },
      margin: { l: 54, r: 20, t: 48, b: 48 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      xaxis: { title: { text: xLabel }, gridcolor: '#e2e8f0', zeroline: false },
      yaxis: { title: { text: yLabel }, gridcolor: '#e2e8f0', zeroline: false },
      hovermode: 'closest',
      updatemenus: [{
        type: 'buttons',
        direction: 'left',
        x: 0,
        y: 1.18,
        xanchor: 'left',
        yanchor: 'top',
        showactive: false,
        buttons: [
          {
            label: 'Play',
            method: 'animate',
            args: [null, {
              fromcurrent: true,
              mode: 'immediate',
              frame: { duration, redraw: true },
              transition: { duration: 0 },
            }],
          },
          {
            label: 'Pause',
            method: 'animate',
            args: [[null], {
              mode: 'immediate',
              frame: { duration: 0, redraw: false },
              transition: { duration: 0 },
            }],
          },
        ],
      }],
    };
  }

  function buildSliderFrames(frames, makeLabel, durationMs) {
    const duration = typeof durationMs === 'number' ? durationMs : 120;
    return [{
      active: 0,
      pad: { t: 28, b: 0 },
      currentvalue: {
        prefix: 'Frame: ',
        font: { size: 12 },
      },
      steps: frames.map((frame, index) => ({
        label: makeLabel(frame, index),
        method: 'animate',
        args: [[frame.name], {
          mode: 'immediate',
          frame: { duration: 0, redraw: true },
          transition: { duration: 0 },
        }],
      })),
    }];
  }

  async function renderAnimatedLine(host, payload, options = {}) {
    if (!host) return;
    const data = payload?.data || {};
    const points = Array.isArray(data.points) ? data.points : [];
    const framesData = Array.isArray(data.frames) ? data.frames : [];
    const title = options.title || data.title || 'Series';
    const xLabel = options.xLabel || data.x_label || 'X';
    const yLabel = options.yLabel || data.y_label || 'Y';
    const intervalMs = options.intervalMs || data.frame_interval_ms || 120;

    if (!points.length) {
      host.innerHTML = '<div class="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-300 text-sm text-slate-500">No data available</div>';
      return;
    }

    const allFrames = framesData.length ? framesData : [{
      index: points.length,
      points,
    }];
    const frames = allFrames.map((frame, index) => {
      const framePoints = Array.isArray(frame.points) ? frame.points : points.slice(0, frame.index || points.length);
      return {
        name: String(index),
        data: [{
          x: framePoints.map((_, pointIndex) => pointIndex + 1),
          y: framePoints,
          mode: 'lines',
          line: { color: '#2563eb', width: 2.5 },
          hovertemplate: 'Packet %{x}<br>Magnitude %{y:.3f}<extra></extra>',
        }],
      };
    });

    const initial = frames[0].data;
    const layout = buildPlaybackLayout(title, xLabel, yLabel, intervalMs);
    layout.uirevision = options.uirevision || 'capture-line';
    layout.sliders = buildSliderFrames(frames, (frame, index) => {
      const label = frame.data?.[0]?.x?.length || index + 1;
      return String(label);
    }, intervalMs);
    layout.yaxis = { title: { text: yLabel }, gridcolor: '#e2e8f0', zeroline: false };

    if (host.data) {
      await Plotly.react(host, initial, layout, plotConfig());
    } else {
      await Plotly.newPlot(host, initial, layout, plotConfig());
    }
    if (frames.length > 1) {
      try {
        await Plotly.deleteFrames(host, frames.map((frame) => frame.name));
      } catch (error) {
        // Frames may not exist yet on the first render.
      }
      await Plotly.addFrames(host, frames);
    }
  }

  async function renderAnimatedHeatmap(host, payload, options = {}) {
    if (!host) return;
    const data = payload?.data || {};
    const framesData = Array.isArray(data.frames) ? data.frames : [];
    const xValues = Array.isArray(data.x) ? data.x : [];
    const yValues = Array.isArray(data.y) ? data.y : [];
    const title = options.title || data.title || 'Heatmap';
    const xLabel = options.xLabel || data.x_label || 'X';
    const yLabel = options.yLabel || data.y_label || 'Y';
    const intervalMs = options.intervalMs || data.frame_interval_ms || 120;
    const isTracking = data.plot === 'xy-tracking';
    const valueLabel = isTracking ? 'Track score' : 'Power';
    const colorbarTitle = isTracking ? 'track history' : 'log power';

    function frameImage(frame) {
      if (Array.isArray(frame?.z)) return frame.z;
      if (!Array.isArray(frame?.z_shape) || !Array.isArray(frame?.z_sparse)) return [];
      const rows = Number(frame.z_shape[0]) || 0;
      const columns = Number(frame.z_shape[1]) || 0;
      const image = Array.from({ length: rows }, () => Array(columns).fill(0));
      frame.z_sparse.forEach((cell) => {
        if (Array.isArray(cell) && image[cell[0]] && cell[1] < columns) image[cell[0]][cell[1]] = cell[2];
      });
      return image;
    }

    function trackingMarker(location, score, detected, snrDb, thresholdDb) {
      const valid = detected === true && Array.isArray(location) && Number.isFinite(location[0]) && Number.isFinite(location[1]);
      return {
        type: 'scatter',
        mode: 'markers',
        x: valid ? [location[0]] : [],
        y: valid ? [location[1]] : [],
        marker: { color: '#ef4444', size: 13, line: { color: '#ffffff', width: 2 } },
        customdata: valid ? [[score, snrDb, thresholdDb]] : [],
        name: 'Tracked target',
        hovertemplate: 'X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Track: %{customdata[0]:.2f}<br>SNR: %{customdata[1]:.1f} dB<br>Threshold: %{customdata[2]:.1f} dB<extra></extra>',
        showlegend: isTracking,
      };
    }

    if (!framesData.length) {
      const z = Array.isArray(data.z) ? data.z : [];
      if (!z.length) {
        host.innerHTML = '<div class="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-300 text-sm text-slate-500">No data available</div>';
        return;
      }
      const staticLayout = buildPlaybackLayout(title, xLabel, yLabel, intervalMs);
      staticLayout.uirevision = options.uirevision || 'capture-heatmap';
      const staticTraces = [{
        type: 'heatmap',
        x: xValues,
        y: yValues,
        z,
        colorscale: 'Viridis',
        zsmooth: false,
        hoverongaps: false,
        hovertemplate: `${yLabel}: %{y}<br>${xLabel}: %{x}<br>${valueLabel} %{z:.2f}<extra></extra>`,
        colorbar: { title: colorbarTitle },
      }];
      if (isTracking) staticTraces.push(trackingMarker(data.location, data.score, data.detected, data.snr_db, data.threshold_db));
      await Plotly.newPlot(host, staticTraces, staticLayout, plotConfig());
      return;
    }

    const frames = framesData.map((frame, index) => {
      const frameTraces = [{
        type: 'heatmap',
        x: Array.isArray(frame.x) && frame.x.length ? frame.x : xValues,
        y: Array.isArray(frame.y) && frame.y.length ? frame.y : yValues,
        z: frameImage(frame),
        colorscale: 'Viridis',
        zsmooth: false,
        hoverongaps: false,
        hovertemplate: `${yLabel}: %{y}<br>${xLabel}: %{x}<br>${valueLabel} %{z:.2f}<extra></extra>`,
        colorbar: { title: colorbarTitle },
      }];
      if (isTracking) frameTraces.push(trackingMarker(frame.location, frame.score, frame.detected, frame.snr_db, frame.threshold_db));
      return { name: String(index), data: frameTraces };
    });

    const staticLayout = buildPlaybackLayout(title, xLabel, yLabel, intervalMs);
    staticLayout.uirevision = options.uirevision || 'capture-heatmap';
    staticLayout.sliders = buildSliderFrames(frames, (frame, index) => String(index + 1), intervalMs);
    if (host.data) {
      await Plotly.react(host, frames[0].data, staticLayout, plotConfig());
    } else {
      await Plotly.newPlot(host, frames[0].data, staticLayout, plotConfig());
    }
    if (frames.length > 1) {
      try {
        await Plotly.deleteFrames(host, frames.map((frame) => frame.name));
      } catch (error) {
        // Frames may not exist yet on the first render.
      }
      await Plotly.addFrames(host, frames);
    }
  }

  window.ThothPlayback = {
    renderAnimatedLine,
    renderAnimatedHeatmap,
  };
})();
