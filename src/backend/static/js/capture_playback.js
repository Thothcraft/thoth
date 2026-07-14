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

  function applyRoomLayout(layout, room) {
    if (!room || !Number(room.width_m) || !Number(room.depth_m)) return;
    const width = Number(room.width_m), depth = Number(room.depth_m);
    layout.xaxis = { ...layout.xaxis, range: [0, width], constrain: 'domain' };
    layout.yaxis = { ...layout.yaxis, range: [0, depth], scaleanchor: 'x', scaleratio: 1 };
    const shapes = [{ type: 'rect', x0: 0, y0: 0, x1: width, y1: depth, line: { color: '#334155', width: 2 }, fillcolor: 'rgba(248,250,252,.18)', layer: 'above' }];
    const cones = Array.isArray(room.radar_cones) && room.radar_cones.length ? room.radar_cones : [{
      wall: room.sensor_wall || 'Back', position_m: room.sensor_position_m || width / 2,
      horizontal_deg: 40, range_m: 15, azimuth_deg: 0, enabled: true,
    }];
    cones.filter((cone) => cone.enabled !== false).forEach((cone) => {
      const wall = cone.wall || cone.sensor_wall || 'Back';
      const position = Number(cone.position_m ?? cone.sensor_position_m ?? 0);
      const origin = wall === 'Back' ? [position, 0] : wall === 'Front' ? [position, depth] : wall === 'Left' ? [0, position] : [width, position];
      const heading = wall === 'Back' ? 90 : wall === 'Front' ? -90 : wall === 'Left' ? 0 : 180;
      const center = (heading + Number(cone.azimuth_deg || 0)) * Math.PI / 180;
      const half = Number(cone.horizontal_deg || 40) * Math.PI / 360;
      const range = Number(cone.range_m || 15);
      const endpoints = [-half, half].map((offset) => [origin[0] + Math.cos(center + offset) * range, origin[1] + Math.sin(center + offset) * range]);
      shapes.push({ type: 'path', path: `M ${origin[0]},${origin[1]} L ${endpoints[0][0]},${endpoints[0][1]} L ${endpoints[1][0]},${endpoints[1][1]} Z`, line: { color: '#0891b2', width: 1.5 }, fillcolor: 'rgba(6,182,212,.10)', layer: 'above' });
    });
    (room.furniture || []).forEach((item) => shapes.push({ type: 'rect', x0: Number(item.x || 0), y0: Number(item.y || 0), x1: Number(item.x || 0) + Number(item.width || .8), y1: Number(item.y || 0) + Number(item.depth || .8), line: { color: '#78716c' }, fillcolor: 'rgba(120,113,108,.20)', layer: 'above' }));
    (room.zones || []).forEach((zone) => {
      const x = Number(zone.x || 0), y = Number(zone.y || 0), zoneWidth = Number(zone.width || 1), zoneDepth = Number(zone.depth || 1);
      shapes.push({ type: 'rect', x0: x, y0: y, x1: x + zoneWidth, y1: y + zoneDepth, line: { color: zone.color || '#22c55e', width: 2 }, fillcolor: `${zone.color || '#22c55e'}18`, layer: 'above' });
    });
    [...(room.doors || []).map((item) => ({ ...item, color: '#f59e0b' })), ...(room.windows || []).map((item) => ({ ...item, color: '#38bdf8' }))].forEach((item) => {
      const wall = item.wall || 'Back', offset = Number(item.offset_m || 0), span = Number(item.width_m || 1);
      const points = wall === 'Back' ? [offset, 0, offset + span, 0] : wall === 'Front' ? [offset, depth, offset + span, depth] : wall === 'Left' ? [0, offset, 0, offset + span] : [width, offset, width, offset + span];
      shapes.push({ type: 'line', x0: points[0], y0: points[1], x1: points[2], y1: points[3], line: { color: item.color, width: 7 }, layer: 'above' });
    });
    layout.shapes = shapes;
  }

  async function renderAnimatedLine(host, payload, options = {}) {
    if (!host) return;
    const data = payload?.data || {};
    const points = Array.isArray(data.points) ? data.points : [];
    const framesData = Array.isArray(data.frames) ? data.frames : [];
    const title = options.title || data.title || 'Series';
    const xLabel = options.xLabel || data.x_label || 'X';
    const yLabel = options.yLabel || data.y_label || 'Y';
    const intervalMs = options.intervalMs || Math.min(750, data.frame_interval_ms || 120);

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
    const intervalMs = options.intervalMs || Math.min(750, data.frame_interval_ms || 120);
    const isTracking = data.plot === 'xy-tracking';
    const valueLabel = isTracking ? 'Track score' : 'Power';
    const colorbarTitle = isTracking ? 'track history' : 'log power';

    function renderOccupancySummary() {
      if (!isTracking || !data.occupancy) return;
      const detected = Number(data.occupancy.detected_frames) || 0;
      const total = Number(data.occupancy.evaluated_frames) || 0;
      const percent = total ? Math.round(detected * 1000 / total) / 10 : 0;
      const threshold = Number(data.occupancy.threshold_percent ?? 50);
      let summary = host.parentElement?.querySelector('[data-radar-occupancy]');
      if (!summary && host.parentElement) {
        summary = document.createElement('div');
        summary.setAttribute('data-radar-occupancy', 'true');
        summary.className = 'mt-2 text-xs font-semibold text-slate-600';
        host.parentElement.appendChild(summary);
      }
      if (summary) summary.textContent = `${data.occupancy.label}: ${detected} / ${total} frames detected (${percent}%); occupied at ≥ ${threshold}%`;
    }

    function frameImage(frame) {
      if (Array.isArray(frame?.z)) return frame.z;
      if (!Array.isArray(frame?.z_shape) || !Array.isArray(frame?.z_sparse)) return Array.isArray(data.z) ? data.z : [];
      const rows = Number(frame.z_shape[0]) || 0;
      const columns = Number(frame.z_shape[1]) || 0;
      const image = Array.from({ length: rows }, () => Array(columns).fill(0));
      frame.z_sparse.forEach((cell) => {
        if (Array.isArray(cell) && image[cell[0]] && cell[1] < columns) image[cell[0]][cell[1]] = cell[2];
      });
      return image;
    }

    function trackingMarker(targets, location, score, detected, snrDb, thresholdDb) {
      const validTargets = Array.isArray(targets) ? targets.filter((target) => Array.isArray(target?.position) && Number.isFinite(Number(target.position[0])) && Number.isFinite(Number(target.position[1]))) : [];
      const fallback = detected === true && Array.isArray(location) && Number.isFinite(location[0]) && Number.isFinite(location[1]) ? [{ id: '?', position: location, confidence: score, snr_db: snrDb, position_error_m: 0 }] : [];
      const plotted = validTargets.length ? validTargets : fallback;
      return {
        type: 'scatter',
        mode: 'markers+text',
        x: plotted.map((target) => Number(target.position[0])),
        y: plotted.map((target) => Number(target.position[1])),
        text: plotted.map((target) => `T${target.id}`),
        textposition: 'top center',
        marker: { color: '#ef4444', size: 13, line: { color: '#ffffff', width: 2 } },
        error_x: { type: 'data', array: plotted.map((target) => Number(target.position_error_m || 0)), visible: true, color: '#ef4444' },
        error_y: { type: 'data', array: plotted.map((target) => Number(target.position_error_m || 0)), visible: true, color: '#ef4444' },
        customdata: plotted.map((target) => [target.id, Number(target.position_error_m || 0), Number(target.confidence ?? score ?? 0), Number(target.snr_db ?? snrDb ?? 0), thresholdDb]),
        name: 'Tracked targets',
        hovertemplate: 'Target %{customdata[0]}<br>X: %{x:.2f} ± %{customdata[1]:.2f} m<br>Y: %{y:.2f} ± %{customdata[1]:.2f} m<br>Confidence: %{customdata[2]:.2f}<br>SNR: %{customdata[3]:.1f} dB<br>Threshold: %{customdata[4]:.1f} dB<extra></extra>',
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
      if (isTracking) staticTraces.push(trackingMarker(data.targets, data.location, data.score, data.detected, data.snr_db, data.threshold_db));
      if (isTracking) applyRoomLayout(staticLayout, data.room);
      await Plotly.newPlot(host, staticTraces, staticLayout, plotConfig());
      renderOccupancySummary();
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
      if (isTracking) frameTraces.push(trackingMarker(frame.targets, frame.location, frame.score, frame.detected, frame.snr_db, frame.threshold_db));
      return { name: String(index), data: frameTraces };
    });

    const staticLayout = buildPlaybackLayout(title, xLabel, yLabel, intervalMs);
    staticLayout.uirevision = options.uirevision || 'capture-heatmap';
    if (isTracking) applyRoomLayout(staticLayout, data.room);
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
    if (typeof host.removeAllListeners === 'function') host.removeAllListeners('plotly_buttonclicked');
    if (typeof host.on === 'function') host.on('plotly_buttonclicked', (event) => {
      host.__thothPlaying = event?.button?.label === 'Play';
    });
    renderOccupancySummary();
  }

  window.ThothPlayback = {
    renderAnimatedLine,
    renderAnimatedHeatmap,
  };
})();
