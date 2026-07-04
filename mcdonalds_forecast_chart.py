"""
Runs the forecast and opens an interactive HTML chart with explainability.
"""
import sys, os, json, subprocess
from datetime import datetime
import concurrent.futures
sys.path.insert(0, os.path.dirname(__file__))

from mcdonalds_forecast import (
    trends_agent, weather_agent, events_agent,
    build_features, forecast_agent
)

print("Running forecast pipeline...", flush=True)

with concurrent.futures.ThreadPoolExecutor() as pool:
    t = pool.submit(trends_agent)
    w = pool.submit(weather_agent)
    e = pool.submit(events_agent)
    trends_df, weather_df, events_df = t.result(), w.result(), e.result()

features_df = build_features(trends_df, weather_df, events_df)
fc = forecast_agent(features_df)

events_dict = {}
if not events_df.empty:
    for _, row in events_df.iterrows():
        events_dict[str(row["ds"].date())] = row.get("event", "")

dates   = [str(d) for d in fc["date"]]
demand  = fc["demand_index"].tolist()
lower   = fc["lower"].tolist()
upper   = fc["upper"].tolist()
colors       = ["#E74C3C" if v >= 130 else "#F39C12" if v >= 110 else "#3498DB" for v in demand]
event_labels = [events_dict.get(d, "") for d in dates]
border_colors = ["#F1C40F" if event_labels[i] else "transparent" for i in range(len(dates))]
border_widths = [3 if event_labels[i] else 0 for i in range(len(dates))]
event_scatter = [demand[i] + 4 if event_labels[i] else None for i in range(len(dates))]
event_labels = [events_dict.get(d, "") for d in dates]

# Component arrays
c_trend   = fc["c_trend"].tolist()
c_weekly  = fc["c_weekly"].tolist()
c_yearly  = fc["c_yearly"].tolist()
c_temp    = fc["c_temp"].tolist() if "c_temp" in fc else [0]*len(dates)
c_rain    = fc["c_rain"].tolist() if "c_rain" in fc else [0]*len(dates)
c_weekend = fc["c_weekend"].tolist() if "c_weekend" in fc else [0]*len(dates)
c_friday  = fc["c_friday"].tolist() if "c_friday" in fc else [0]*len(dates)
c_event   = fc["c_event"].tolist() if "c_event" in fc else [0]*len(dates)

event_labels  = [events_dict.get(d, "") for d in dates]
events_rows = "".join(
    f"""<div class="event-row">
      <span class="ev-date">{d}</span>
      <span class="ev-name">{e}</span>
      <span class="ev-idx" style="color:#E74C3C">{demand[dates.index(d)]}</span>
    </div>"""
    for d, e in events_dict.items() if d in dates
)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>McDonald's Toronto Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#F4F6F8;padding:24px;color:#111}}
  h1{{font-size:18px;font-weight:700;margin-bottom:2px}}
  .sub{{font-size:12px;color:#888;margin-bottom:18px}}
  .stats{{display:flex;gap:10px;margin-bottom:16px}}
  .stat{{background:#fff;border-radius:10px;border:1px solid #eee;padding:12px 16px;flex:1}}
  .stat-v{{font-size:22px;font-weight:700}}
  .stat-l{{font-size:11px;color:#888;margin-top:2px}}
  .card{{background:#fff;border-radius:12px;border:1px solid #eee;padding:20px;margin-bottom:14px}}
  .card h2{{font-size:13px;font-weight:700;margin-bottom:14px}}
  .legend{{display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;font-size:11px;color:#555}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}}
  .explainer{{display:none;background:#F8F9FA;border-radius:8px;padding:14px;margin-top:12px;font-size:12px}}
  .explainer.show{{display:block}}
  .bar-row{{display:flex;align-items:center;gap:8px;margin:5px 0}}
  .bar-label{{min-width:90px;font-size:11px;color:#555}}
  .bar-track{{flex:1;height:14px;background:#F0F0F0;border-radius:4px;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:4px;transition:width .3s}}
  .bar-val{{font-size:11px;color:#333;min-width:38px;text-align:right;font-weight:600}}
  .ev-row{{display:flex;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid #f0f0f0;font-size:12px}}
  .ev-date{{color:#888;min-width:90px}}
  .ev-idx{{margin-left:auto;font-weight:700}}
  select{{font-size:12px;padding:4px 8px;border-radius:6px;border:1px solid #ddd;background:#fff;cursor:pointer}}
</style>
</head>
<body>
<h1>🍔 McDonald's Toronto — 30-Day Demand Forecast</h1>
<div class="sub">Generated {datetime.today().strftime('%B %d, %Y')} · Demand index: 100 = average day · Click any bar to explain it</div>

<div class="stats">
  <div class="stat"><div class="stat-v" style="color:#E74C3C">{max(demand)}</div><div class="stat-l">Peak demand index</div></div>
  <div class="stat"><div class="stat-v" style="color:#3498DB">{min(demand)}</div><div class="stat-l">Low demand index</div></div>
  <div class="stat"><div class="stat-v">{round(sum(demand)/len(demand))}</div><div class="stat-l">30-day average</div></div>
  <div class="stat"><div class="stat-v" style="color:#E74C3C">{sum(1 for v in demand if v >= 130)}</div><div class="stat-l">High-demand days (≥130)</div></div>
</div>

<!-- Main forecast chart -->
<div class="card">
  <h2>📊 Daily Demand Index <span style="font-weight:400;color:#888;font-size:12px">— click a bar to see why</span></h2>
  <canvas id="mainChart" height="260"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#E74C3C"></span>High (≥130)</span>
    <span><span class="dot" style="background:#F39C12"></span>Elevated (110–129)</span>
    <span><span class="dot" style="background:#3498DB"></span>Normal (&lt;110)</span>
    <span><span class="dot" style="background:#BDC3C7"></span>Confidence range</span>
    <span><span style="display:inline-block;width:10px;height:10px;border:2px solid #F1C40F;border-radius:2px;margin-right:4px"></span>Event day</span>
    <span>⭐ = event marker</span>
  </div>
  <!-- Explainer panel (shown on bar click) -->
  <div class="explainer" id="explainer">
    <strong id="exp-title"></strong>
    <div style="color:#888;font-size:11px;margin-bottom:10px" id="exp-sub"></div>
    <div id="exp-bars"></div>
  </div>
</div>

<!-- Component breakdown chart -->
<div class="card">
  <h2>🔍 What's driving demand each day?</h2>
  <div style="margin-bottom:10px;font-size:12px;color:#555">Each segment shows how much that factor adds or subtracts from the baseline of 100.</div>
  <canvas id="compChart" height="260"></canvas>
  <div class="legend" style="margin-top:10px">
    <span><span class="dot" style="background:#3498DB"></span>Trend</span>
    <span><span class="dot" style="background:#2ECC71"></span>Day of week</span>
    <span><span class="dot" style="background:#9B59B6"></span>Seasonality</span>
    <span><span class="dot" style="background:#E67E22"></span>Weather</span>
    <span><span class="dot" style="background:#E74C3C"></span>Events</span>
  </div>
</div>

<!-- Events table -->
<div class="card">
  <h2>📅 Key Toronto Events This Period</h2>
  {events_rows if events_rows else '<div style="color:#888;font-size:12px">No events found for this period.</div>'}
</div>

<script>
const dates = {json.dumps(dates)};
const demand = {json.dumps(demand)};
const lower  = {json.dumps(lower)};
const upper  = {json.dumps(upper)};
const colors = {json.dumps(colors)};
const evLabels = {json.dumps(event_labels)};
const cTrend   = {json.dumps(c_trend)};
const cWeekly  = {json.dumps(c_weekly)};
const cYearly  = {json.dumps(c_yearly)};
const cTemp    = {json.dumps(c_temp)};
const cRain    = {json.dumps(c_rain)};
const cWeekend = {json.dumps(c_weekend)};
const cFriday  = {json.dumps(c_friday)};
const cEvent   = {json.dumps(c_event)};

// ── Main chart ────────────────────────────────────────────────────────────────
const mainCtx = document.getElementById('mainChart').getContext('2d');
const borderColors = {json.dumps(border_colors)};
const borderWidths = {json.dumps(border_widths)};
const eventScatter = {json.dumps(event_scatter)};

const mainChart = new Chart(mainCtx, {{
  data: {{
    labels: dates,
    datasets: [
      {{ type:'line', label:'Upper', data:upper, borderColor:'rgba(189,195,199,0.4)', backgroundColor:'rgba(189,195,199,0.12)', borderWidth:1, pointRadius:0, fill:'+1', tension:0.3, order:0 }},
      {{ type:'line', label:'Lower', data:lower, borderColor:'rgba(189,195,199,0.4)', fill:false, borderWidth:1, pointRadius:0, tension:0.3, order:0 }},
      {{ type:'bar',  label:'Demand Index', data:demand, backgroundColor:colors, borderColor:borderColors, borderWidth:borderWidths, borderRadius:4, borderSkipped:false, order:1 }},
      {{ type:'line', label:'Baseline (100)', data:Array(dates.length).fill(100), borderColor:'#95A5A6', borderWidth:1.5, borderDash:[5,4], pointRadius:0, fill:false, order:0 }},
      {{ type:'scatter', label:'Event', data:eventScatter.map((v,i)=>v!=null?{{x:dates[i],y:v}}:null).filter(Boolean),
         pointStyle:'star', pointRadius:8, pointBackgroundColor:'#F1C40F', pointBorderColor:'#E67E22', pointBorderWidth:1, order:0 }},
    ]
  }},
  options: {{
    responsive:true,
    interaction:{{mode:'index',intersect:false}},
    onClick(evt, elems) {{
      if (!elems.length) return;
      const i = elems.find(e => e.datasetIndex === 2)?.index;
      if (i == null) return;
      showExplainer(i);
    }},
    plugins:{{
      legend:{{display:false}},
      tooltip:{{callbacks:{{afterBody:(items)=>{{const i=items[0].dataIndex;return evLabels[i]?['','📅 '+evLabels[i]]:[];}}}}}}
    }},
    scales:{{
      x:{{ticks:{{maxRotation:45,font:{{size:10}}}},grid:{{display:false}}}},
      y:{{min:80,max:200,title:{{display:true,text:'Demand Index',font:{{size:11}}}},grid:{{color:'#F5F5F5'}}}}
    }}
  }}
}});

// ── Component chart ───────────────────────────────────────────────────────────
const compCtx = document.getElementById('compChart').getContext('2d');
new Chart(compCtx, {{
  type:'bar',
  data:{{
    labels:dates,
    datasets:[
      {{ label:'Trend',      data:cTrend,   backgroundColor:'#3498DB', stack:'s' }},
      {{ label:'Day of week',data:cWeekly,  backgroundColor:'#2ECC71', stack:'s' }},
      {{ label:'Seasonality',data:cYearly,  backgroundColor:'#9B59B6', stack:'s' }},
      {{ label:'Temperature',data:cTemp,    backgroundColor:'#E67E22', stack:'s' }},
      {{ label:'Rain',       data:cRain,    backgroundColor:'#7F8C8D', stack:'s' }},
      {{ label:'Events',     data:cEvent,   backgroundColor:'#E74C3C', stack:'s' }},
    ]
  }},
  options:{{
    responsive:true,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{maxRotation:45,font:{{size:10}}}},grid:{{display:false}},stacked:true}},
      y:{{stacked:true,title:{{display:true,text:'Index contribution',font:{{size:11}}}},grid:{{color:'#F5F5F5'}}}}
    }}
  }}
}});

// ── Explainer panel ───────────────────────────────────────────────────────────
function showExplainer(i) {{
  const panel = document.getElementById('explainer');
  panel.classList.add('show');

  const d = dates[i];
  const v = demand[i];
  document.getElementById('exp-title').textContent = d + '  —  Demand Index: ' + v;
  const ev = evLabels[i] ? '📅 ' + evLabels[i] : '';
  document.getElementById('exp-sub').textContent = ev || (v >= 130 ? 'High demand day' : v >= 110 ? 'Elevated demand' : 'Normal demand day');

  const factors = [
    {{ name:'Trend',       val:cTrend[i],   color:'#3498DB' }},
    {{ name:'Day of week', val:cWeekly[i],  color:'#2ECC71' }},
    {{ name:'Seasonality', val:cYearly[i],  color:'#9B59B6' }},
    {{ name:'Temperature', val:cTemp[i],    color:'#E67E22' }},
    {{ name:'Rain',        val:cRain[i],    color:'#7F8C8D' }},
    {{ name:'Events',      val:cEvent[i],   color:'#E74C3C' }},
  ].filter(f => Math.abs(f.val) > 0.5)
   .sort((a,b) => Math.abs(b.val)-Math.abs(a.val));

  const total = factors.reduce((s,f) => s + Math.abs(f.val), 0);
  document.getElementById('exp-bars').innerHTML = factors.map(f => {{
    const pct = total > 0 ? Math.abs(f.val)/total*100 : 0;
    const sign = f.val >= 0 ? '+' : '';
    return `<div class="bar-row">
      <div class="bar-label">${{f.name}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct.toFixed(0)}}%;background:${{f.color}}"></div></div>
      <div class="bar-val">${{sign}}${{f.val.toFixed(1)}}</div>
    </div>`;
  }}).join('');
}}
</script>
</body>
</html>"""

out = "/Users/saujanyashrivastava/Git/forecast_chart.html"
with open(out, "w") as f:
    f.write(html)

print(f"\nOpening chart with explainability...")
subprocess.run(["open", out])
print(f"File: {out}")
