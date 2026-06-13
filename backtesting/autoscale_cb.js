// ============================================================
// autoscale_cb.js — Bokeh 图表 Y 轴自动缩放回调
// ============================================================
// 上下文层：当用户在 Bokeh HTML 图表中平移/缩放 X 轴时，
//           本脚本自动调整 OHLC 价格轴 (y_range) 和成交量轴 (volume_range)
//           的范围，使得可见区域的 K 线充满 Y 轴。
// 功能层：1. 监听 x_range 变化事件
//          2. 根据可视区内的最高/最低价调整 Y 轴
//          3. 如果存在成交量图，同步缩放成交量轴
// 设计层：使用 setTimeout 防抖（50ms），避免频繁重绘。
//          通过 window 全局变量在多次调用间共享状态。
// ============================================================

// 功能层：工具函数——设置 range 的 start/end，带可选的 3% 内边距
if (!window._bt_scale_range) {
    window._bt_scale_range = function (range, min, max, pad) {
        "use strict";
        if (min !== Infinity && max !== -Infinity) {
            // 功能层：pad=true 时在上下各留 3% 的边距，避免 K 线贴边
            pad = pad ? (max - min) * .03 : 0;
            range.start = min - pad;
            range.end = max + pad;
        } else console.error('backtesting: scale range error:', min, max, range);
    };
}

// 功能层：清除上一次的定时器，实现防抖（debounce）
//         避免用户快速拖拽时触发太多次重绘
clearTimeout(window._bt_autoscale_timeout);

// 功能层：50ms 后执行自动缩放
window._bt_autoscale_timeout = setTimeout(function () {
    /**
     * @variable cb_obj `fig_ohlc.x_range`.        // Bokeh 传入的当前 X 轴范围对象
     * @variable source `ColumnDataSource`          // 包含全部 OHLC 数据的 Bokeh 数据源
     * @variable ohlc_range `fig_ohlc.y_range`.     // OHLC 图表的 Y 轴范围对象
     * @variable volume_range `fig_volume.y_range`. // 成交量图表的 Y 轴范围对象（可选）
     */
    "use strict";

    // 功能层：计算可视区域的起止索引（从 X 轴范围推算）
    let i = Math.max(Math.floor(cb_obj.start), 0),
        j = Math.min(Math.ceil(cb_obj.end), source.data['ohlc_high'].length);

    // 功能层：取可视区域内最高价和最低价
    let max = Math.max.apply(null, source.data['ohlc_high'].slice(i, j)),
        min = Math.min.apply(null, source.data['ohlc_low'].slice(i, j));
    // 功能层：应用 OHLC Y 轴缩放（带 3% 内边距）
    _bt_scale_range(ohlc_range, min, max, true);

    // 功能层：如果存在成交量子图，同步缩放其 Y 轴（从 0 到可视区最大成交量 +3%）
    if (volume_range) {
        max = Math.max.apply(null, source.data['Volume'].slice(i, j));
        _bt_scale_range(volume_range, 0, max * 1.03, false);
    }

}, 50);
