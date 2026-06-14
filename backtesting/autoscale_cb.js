// ============================================================
// autoscale_cb.js -- Bokeh 图表 Y 轴自动缩放
// ============================================================
// 上下文层: 用户平移/缩放 X 轴时, 自动调整 OHLC 价格轴范围, 
//           使可见区域 K 线充满 Y 轴.通过 CustomJS 嵌入 HTML.
// 功能层: 监听 x_range 变化 → 取可视区内最高/最低价 → 设 Y 轴.
// 设计层: setTimeout 50ms 防抖, _bt_scale_range 工具函数复用于 OHLC 和 Volume.

if (!window._bt_scale_range) {
    // 工具: 设置 range.start/end, 可选 3% padding
    window._bt_scale_range = function (range, min, max, pad) {
        "use strict";
        if (min !== Infinity && max !== -Infinity) {
            pad = pad ? (max - min) * .03 : 0;
            range.start = min - pad;
            range.end = max + pad;
        } else console.error('backtesting: scale range error:', min, max, range);
    };
}

// 清除上次定时器----防抖
clearTimeout(window._bt_autoscale_timeout);

window._bt_autoscale_timeout = setTimeout(function () {
    /**
     * @variable cb_obj        -- Bokeh 传入的 x_range 对象
     * @variable source         -- ColumnDataSource(含全部 OHLC + 指标数据)
     * @variable ohlc_range     -- OHLC 图 Y 轴范围
     * @variable volume_range   -- 成交量图 Y 轴范围(可选)
     */
    "use strict";

    // 可视区起止索引
    let i = Math.max(Math.floor(cb_obj.start), 0),
        j = Math.min(Math.ceil(cb_obj.end), source.data['ohlc_high'].length);

    // 取可视区内最高/最低价, 设 OHLC Y 轴
    let max = Math.max.apply(null, source.data['ohlc_high'].slice(i, j)),
        min = Math.min.apply(null, source.data['ohlc_low'].slice(i, j));
    _bt_scale_range(ohlc_range, min, max, true);

    // 如果有成交量图, 同步缩放
    if (volume_range) {
        max = Math.max.apply(null, source.data['Volume'].slice(i, j));
        _bt_scale_range(volume_range, 0, max * 1.03, false);
    }

}, 50);
