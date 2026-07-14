import { useEffect, useMemo, useRef, useState } from "react";

const API = "/api";
const ROWS = 6;

type Kind = "empty" | "fuel" | "vent" | "exchanger" | "coolant" | "condensator" | "plating" | "reflector";
type ComponentSpec = {
  id: string; name: string; short_name: string; kind: Kind; texture?: string;
  max_heat: number; max_damage: number; rod_count: number;
};
type TickRow = {
  game_tick: number; seconds: number; reactor_tick: number; hull_heat: number; max_hull_heat: number;
  eu_per_tick: number; total_eu: number; generated_heat: number; vented_heat: number;
};
type Summary = {
  stop_reason: string; game_ticks: number; hull_heat: number; max_hull_heat: number; peak_hull_heat: number;
  current_eu_per_tick: number; average_eu_per_tick: number; total_eu: number; first_intervention_tick?: number;
  meltdown_tick?: number; mark?: string; stable: boolean; events: Array<{game_tick: number; message: string; type: string}>;
};
type Candidate = { layout: {columns: number; slots: string[]}; mark: string; average_eu_per_tick: number; total_eu: number; safe_game_ticks: number; safety_margin: number; component_count: number };

const FALLBACK: ComponentSpec[] = [
  ["empty", "空格", "空", "empty"], ["uranium_single", "燃料棒（铀）", "单铀", "fuel"],
  ["uranium_dual", "双联燃料棒（铀）", "双铀", "fuel"], ["uranium_quad", "四联燃料棒（铀）", "四铀", "fuel"],
  ["heat_vent", "散热片", "散热", "vent"], ["advanced_heat_vent", "高级散热片", "高散", "vent"],
  ["reactor_heat_vent", "反应堆散热片", "堆散", "vent"], ["component_heat_vent", "元件散热片", "元散", "vent"],
  ["overclocked_heat_vent", "超频散热片", "超散", "vent"], ["coolant_10k", "10k 冷却单元", "10k", "coolant"],
  ["coolant_30k", "30k 冷却单元", "30k", "coolant"], ["coolant_60k", "60k 冷却单元", "60k", "coolant"],
  ["heat_exchanger", "热交换器", "换热", "exchanger"], ["advanced_heat_exchanger", "高级热交换器", "高换", "exchanger"],
  ["reactor_heat_exchanger", "反应堆热交换器", "堆换", "exchanger"], ["component_heat_exchanger", "元件热交换器", "元换", "exchanger"],
  ["reactor_plating", "反应堆隔板", "隔板", "plating"], ["heat_capacity_plating", "高热容反应堆隔板", "热板", "plating"],
  ["containment_plating", "密封反应堆隔板", "密板", "plating"], ["rsh_condensator", "红石冷凝模块", "RSH", "condensator"],
  ["lzh_condensator", "青金石冷凝模块", "LZH", "condensator"], ["neutron_reflector", "中子反射板", "反射", "reflector"],
  ["thick_neutron_reflector", "加厚中子反射板", "厚反", "reflector"], ["iridium_reflector", "铱中子反射板", "铱反", "reflector"],
].map(([id, name, short_name, kind]) => ({id, name, short_name, kind: kind as Kind, max_heat: 0, max_damage: 0, rod_count: 0}));

const KIND_LABELS: Record<Kind, string> = {empty: "工具", fuel: "燃料", vent: "散热片", exchanger: "换热器", coolant: "冷却单元", condensator: "冷凝模块", plating: "隔板", reflector: "反射板"};
const DEFAULT_LIMITS = ["heat_vent", "advanced_heat_vent", "reactor_heat_vent", "component_heat_vent", "overclocked_heat_vent", "heat_exchanger", "advanced_heat_exchanger", "reactor_heat_exchanger", "component_heat_exchanger"];

function fmt(value: number, digits = 0) {
  return new Intl.NumberFormat("zh-CN", {maximumFractionDigits: digits}).format(value || 0);
}

function fmtIntegerString(value: string) {
  return value.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function ComponentIcon({spec, small = false}: {spec?: ComponentSpec; small?: boolean}) {
  const [failed, setFailed] = useState(false);
  if (!spec || spec.id === "empty") return <span className="empty-mark">＋</span>;
  return <>
    {spec.texture && !failed ? <img src={spec.texture} alt="" onError={() => setFailed(true)} /> : null}
    {(!spec.texture || failed) && <span className={small ? "abbr small" : "abbr"}>{spec.short_name}</span>}
  </>;
}

function TraceCanvas({points, currentTick, onSeek}: {points: any[]; currentTick: number; onSeek: (tick: number) => void}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const box = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    canvas.width = box.width * dpr; canvas.height = box.height * dpr;
    const ctx = canvas.getContext("2d"); if (!ctx) return; ctx.scale(dpr, dpr);
    const w = box.width, h = box.height, pad = 32; ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#29323a"; ctx.lineWidth = 1;
    for (let i=0;i<5;i++){ const y=pad+(h-pad*2)*i/4; ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke(); }
    if (!points.length) { ctx.fillStyle="#71808c";ctx.font="13px sans-serif";ctx.fillText("运行模拟后显示温度与发电曲线", pad, h/2);return; }
    const maxTick = points.at(-1).game_tick || 1; const maxEU = Math.max(1, ...points.map(p=>p.eu_per_tick));
    const draw = (color: string, calc: (p:any)=>number) => {ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{const x=pad+(w-pad*2)*p.game_tick/maxTick;const y=h-pad-(h-pad*2)*calc(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()};
    draw("#ff6b35", p=>p.hull_heat/Math.max(1,p.max_hull_heat)); draw("#ffd23f", p=>p.eu_per_tick/maxEU);
    const x=pad+(w-pad*2)*Math.min(currentTick,maxTick)/maxTick;ctx.strokeStyle="#dfe7eb";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,h-pad);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle="#ff6b35";ctx.fillText("堆温 %",pad,16);ctx.fillStyle="#ffd23f";ctx.fillText("EU/t",pad+58,16);
  }, [points, currentTick]);
  return <canvas ref={ref} className="trace-canvas" onClick={e=>{if(!points.length)return;const r=e.currentTarget.getBoundingClientRect();const ratio=Math.max(0,Math.min(1,(e.clientX-r.left-32)/(r.width-64)));onSeek(Math.round(ratio*points.at(-1).game_tick));}} />;
}

export default function Home() {
  const [tab, setTab] = useState<"simulate"|"optimize"|"rules">("simulate");
  const [components, setComponents] = useState<ComponentSpec[]>(FALLBACK);
  const specs = useMemo(()=>Object.fromEntries(components.map(c=>[c.id,c])),[components]);
  const [columns, setColumns] = useState(6);
  const [slots, setSlots] = useState<string[]>(Array(36).fill("empty"));
  const [selected, setSelected] = useState("uranium_single");
  const [painting, setPainting] = useState(false);
  const [initialHeat, setInitialHeat] = useState(0);
  const [maxTicks, setMaxTicks] = useState(400000);
  const [tickRate, setTickRate] = useState(200);
  const [autoRefuel, setAutoRefuel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [simulationId, setSimulationId] = useState("");
  const [summary, setSummary] = useState<Summary|null>(null);
  const [chart, setChart] = useState<any[]>([]);
  const [tickRows, setTickRows] = useState<TickRow[]>([]);
  const [currentTick, setCurrentTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [componentState, setComponentState] = useState<any[]>([]);
  const [fuelMode, setFuelMode] = useState<"separate"|"total_rods">("separate");
  const [fuelLimits, setFuelLimits] = useState({single:1,dual:0,quad:0,total_rods:1});
  const [limits, setLimits] = useState<Record<string, number>>(()=>Object.fromEntries(DEFAULT_LIMITS.map(id=>[id,8])));
  const [marks, setMarks] = useState<string[]>(["I"]);
  const [solver, setSolver] = useState<"heuristic"|"exhaustive">("heuristic");
  const [search, setSearch] = useState({seconds:30,generations:100,population:80,workers:Math.max(1,(navigator.hardwareConcurrency||2)-1),seed:221});
  const [enumerationEstimate, setEnumerationEstimate] = useState("");
  const [job, setJob] = useState<any>(null);
  const eventSource = useRef<EventSource|null>(null);

  useEffect(()=>{fetch(`${API}/components`).then(r=>r.ok?r.json():Promise.reject()).then(data=>setComponents(data.components)).catch(()=>{});},[]);
  useEffect(()=>()=>eventSource.current?.close(),[]);
  useEffect(()=>{if(!playing||!summary)return;const id=setInterval(()=>setCurrentTick(t=>{const next=Math.min(summary.game_ticks,t+Math.max(1,Math.round(tickRate/10)));if(next>=summary.game_ticks)setPlaying(false);return next}),100);return()=>clearInterval(id)},[playing,summary,tickRate]);
  useEffect(()=>{if(!simulationId||!summary)return;const page=Math.floor(currentTick/200)*200;fetch(`${API}/simulations/${simulationId}/ticks?offset=${page}&limit=200`).then(r=>r.json()).then(d=>setTickRows(d.rows));fetch(`${API}/simulations/${simulationId}/components?game_tick=${currentTick}`).then(r=>r.json()).then(d=>setComponentState(d.components||[]));},[simulationId,currentTick,summary]);

  const resize = (next:number) => {setColumns(next);setSlots(old=>{const out=Array(ROWS*next).fill("empty");for(let r=0;r<ROWS;r++)for(let c=0;c<Math.min(columns,next);c++)out[r*next+c]=old[r*columns+c];return out})};
  const paint = (index:number,id=selected) => setSlots(old=>old.map((v,i)=>i===index?id:v));
  const counts = useMemo(()=>slots.reduce<Record<string,number>>((a,id)=>(a[id]=(a[id]||0)+1,a),{}),[slots]);

  const runSimulation = async () => {
    setBusy(true);setError("");setPlaying(false);
    try {const res=await fetch(`${API}/simulations`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({layout:{columns,initial_hull_heat:initialHeat,slots},max_game_ticks:maxTicks,auto_refuel:autoRefuel,record_components:true})});if(!res.ok)throw new Error((await res.json()).detail||"模拟失败");const data=await res.json();setSimulationId(data.id);setSummary(data.summary);setCurrentTick(0);const c=await fetch(`${API}/simulations/${data.id}/chart?points=1200`).then(r=>r.json());setChart(c.points);setPlaying(true)} catch(e:any){setError(e.message||"无法连接计算服务")} finally{setBusy(false)}
  };
  const toggleLimit=(id:string)=>setLimits(v=>id in v?Object.fromEntries(Object.entries(v).filter(([k])=>k!==id)):{...v,[id]:8});
  const toggleMark=(mark:string)=>setMarks(v=>v.includes(mark)?v.filter(x=>x!==mark):[...v,mark]);
  const followJob=(id:string)=>{eventSource.current?.close();const source=new EventSource(`${API}/optimizations/${id}/events`);eventSource.current=source;source.onmessage=e=>{const value=JSON.parse(e.data);setJob(value);if(["completed","cancelled","failed"].includes(value.status))source.close()};source.onerror=()=>source.close()};
  useEffect(()=>{fetch(`${API}/optimizations/latest`).then(r=>r.ok?r.json():Promise.reject()).then(value=>{setJob(value);setEnumerationEstimate(value.estimate||"");if(["queued","running"].includes(value.status))followJob(value.id)}).catch(()=>{})},[]);
  const runOptimization=async()=>{setError("");try{const payload={columns,fuel:{mode:fuelMode,...fuelLimits},component_limits:limits,marks,solver,max_reactor_ticks:40000,cpu_workers:search.workers,...(solver==="heuristic"?{time_budget_seconds:search.seconds,generations:search.generations,population:search.population,seed:search.seed}:{})};if(solver==="exhaustive"){const estimateResponse=await fetch(`${API}/optimizations/estimate`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!estimateResponse.ok)throw new Error((await estimateResponse.json()).detail||"无法计算枚举方案数");const estimate=String((await estimateResponse.json()).estimate);if(!window.confirm(`枚举方案数：${fmtIntegerString(estimate)}\n\n是否开始？`))return;setEnumerationEstimate(estimate)}else setEnumerationEstimate("");setJob(null);eventSource.current?.close();const res=await fetch(`${API}/optimizations`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!res.ok)throw new Error((await res.json()).detail||"无法启动优化");const data=await res.json();followJob(data.id)}catch(e:any){setError(e.message)}};
  const resumeOptimization=async()=>{if(!job)return;setError("");const res=await fetch(`${API}/optimizations/${job.id}/resume`,{method:"POST"});if(!res.ok){setError((await res.json()).detail||"无法继续优化");return}followJob(job.id)};
  const loadCandidate=async(candidate:Candidate,mark:string,rank:number)=>{setColumns(candidate.layout.columns);setSlots(candidate.layout.slots);setComponentState([]);setTab("simulate");setSummary(null);setChart([]);setCurrentTick(0);if(!job)return;setBusy(true);try{const res=await fetch(`${API}/optimizations/${job.id}/candidates/${mark}/${rank}/simulation`,{method:"POST"});if(!res.ok)throw new Error((await res.json()).detail||"无法生成候选轨迹");const data=await res.json();setSimulationId(data.id);setSummary(data.summary);const c=await fetch(`${API}/simulations/${data.id}/chart?points=1200`).then(r=>r.json());setChart(c.points)}catch(e:any){setError(e.message)}finally{setBusy(false)}};

  const groups = useMemo(()=>components.filter(c=>c.id!=="empty").reduce<Record<string,ComponentSpec[]>>((a,c)=>(a[c.kind]??=[],a[c.kind].push(c),a),{}),[components]);
  const displayedState=componentState.length?componentState:slots.map((component_id,slot)=>({component_id,slot,heat:0,damage:0}));

  return <main onMouseUp={()=>setPainting(false)} onMouseLeave={()=>setPainting(false)}>
    <header className="topbar">
      <div className="brand"><span className="reactor-logo">IC²</span><div><h1>核反应堆模拟与优化器</h1><p>Experimental 2.8.221 · EU 模式</p></div></div>
      <nav>{[["simulate","布局模拟"],["optimize","计算最优"],["rules","规则数据"]].map(([id,label])=><button key={id} className={tab===id?"active":""} onClick={()=>setTab(id as any)}>{label}</button>)}</nav>
      <div className="status"><span className="dot"/>本地计算</div>
    </header>
    {error&&<div className="error-banner">{error}<button onClick={()=>setError("")}>×</button></div>}

    {tab==="simulate"&&<div className="workspace">
      <aside className="palette panel"><div className="panel-title"><span>组件仓库</span><b>{slots.filter(x=>x!=="empty").length}/{slots.length}</b></div>
        <button className={`palette-item eraser ${selected==="empty"?"selected":""}`} onClick={()=>setSelected("empty")}><span>⌫</span><label>擦除工具</label></button>
        {Object.entries(groups).map(([kind,items])=><section key={kind}><h3>{KIND_LABELS[kind as Kind]}</h3><div className="palette-grid">{items.map(spec=><button title={spec.name} key={spec.id} className={`palette-item ${selected===spec.id?"selected":""}`} onClick={()=>setSelected(spec.id)}><span className={`item-icon kind-${spec.kind}`}><ComponentIcon spec={spec}/></span><label>{spec.short_name}</label>{counts[spec.id]?<em>{counts[spec.id]}</em>:null}</button>)}</div></section>)}
      </aside>
      <section className="center-column">
        <div className="panel reactor-panel"><div className="panel-title"><span>反应堆布局</span><div className="inline-controls"><label>反应堆仓 <select value={columns-3} onChange={e=>resize(Number(e.target.value)+3)}>{[0,1,2,3,4,5,6].map(n=><option key={n} value={n}>{n}（{6*(n+3)} 格）</option>)}</select></label><button onClick={()=>setSlots(Array(slots.length).fill("empty"))}>清空</button><button onClick={()=>navigator.clipboard?.writeText(JSON.stringify({columns,slots}))}>复制布局</button></div></div>
          <div className="reactor-frame"><div className="reactor-grid" style={{gridTemplateColumns:`repeat(${columns}, 62px)`}}>{displayedState.map((state:any,index:number)=>{const spec=specs[state.component_id];const heat=spec?.max_heat?state.heat/spec.max_heat:0;const dmg=spec?.max_damage?state.damage/spec.max_damage:0;return <button data-slot={index} title={`${index+1}. ${spec?.name||"空格"}`} key={index} className={`reactor-slot kind-${spec?.kind||"empty"} ${state.component_id==="empty"?"empty":""}`} onMouseDown={e=>{e.preventDefault();setPainting(true);paint(index,e.button===2?"empty":selected)}} onMouseEnter={()=>painting&&paint(index)} onContextMenu={e=>{e.preventDefault();paint(index,"empty")}}><ComponentIcon spec={spec}/><span className="slot-no">{index+1}</span>{heat>0&&<span className="heat-bar" style={{height:`${Math.min(1,heat)*100}%`}}/>}{dmg>0&&<span className="damage-bar" style={{width:`${Math.min(1,dmg)*100}%`}}/>}</button>})}</div></div>
          <div className="reactor-summary"><div><span>堆体热量</span><strong>{fmt(summary?.hull_heat||initialHeat)} / {fmt(summary?.max_hull_heat||10000)}</strong></div><div><span>当前输出</span><strong className="yellow">{fmt(summary?.current_eu_per_tick||0,2)} EU/t</strong></div><div><span>累计发电</span><strong>{fmt(summary?.total_eu||0)} EU</strong></div><div><span>分类</span><strong>{summary?.mark||"待模拟"}</strong></div></div>
        </div>
        <div className="panel chart-panel"><div className="panel-title"><span>运行曲线</span><span className="legend">橙：堆温 · 黄：发电量</span></div><TraceCanvas points={chart} currentTick={currentTick} onSeek={setCurrentTick}/><div className="transport"><button onClick={()=>setCurrentTick(0)}>↤</button><button className="play" disabled={!summary} onClick={()=>setPlaying(v=>!v)}>{playing?"暂停":"播放"}</button><button onClick={()=>setCurrentTick(t=>Math.min(summary?.game_ticks||0,t+1))}>+1 tick</button><input type="range" min="0" max={summary?.game_ticks||1} value={currentTick} onChange={e=>setCurrentTick(Number(e.target.value))}/><b>{fmt(currentTick)} tick</b></div></div>
      </section>
      <aside className="right-column">
        <div className="panel settings"><div className="panel-title">模拟设置</div><label>初始堆热<input type="number" min="0" value={initialHeat} onChange={e=>setInitialHeat(Number(e.target.value))}/></label><label>game tick 上限<input type="number" min="20" step="20" value={maxTicks} onChange={e=>setMaxTicks(Number(e.target.value))}/></label><label>播放 tick/s<input type="number" min="1" value={tickRate} onChange={e=>setTickRate(Number(e.target.value))}/></label><label className="check"><input type="checkbox" checked={autoRefuel} onChange={e=>setAutoRefuel(e.target.checked)}/>燃料耗尽后原位续棒</label><button className="primary" disabled={busy||!slots.some(id=>specs[id]?.kind==="fuel")} onClick={runSimulation}>{busy?"正在计算…":"开始模拟"}</button>{summary&&<div className={`stop-card ${summary.stop_reason}`}><b>{summary.stop_reason==="meltdown"?"反应堆已融毁":summary.stable?"检测到周期稳态":"达到 tick 上限"}</b><span>峰值堆温 {fmt(summary.peak_hull_heat)}</span><span>平均 {fmt(summary.average_eu_per_tick,2)} EU/t</span></div>}</div>
        <div className="panel events"><div className="panel-title">事件记录</div><div className="event-list">{summary?.events.length?summary.events.slice(-10).reverse().map((e,i)=><div key={i}><time>{fmt(e.game_tick)}t</time><span>{e.message}</span></div>):<p>临界、组件损坏、燃料耗尽/续棒和融毁事件将显示在这里。</p>}</div></div>
      </aside>
      <section className="panel tick-panel"><div className="panel-title"><span>完整 game tick 表</span>{simulationId&&<div><a href={`${API}/simulations/${simulationId}/export.csv`}>导出摘要 CSV</a><a href={`${API}/simulations/${simulationId}/export.csv?components=true`}>导出组件明细</a></div>}</div><div className="table-wrap"><table><thead><tr><th>Game Tick</th><th>时间/s</th><th>反应堆周期</th><th>堆热</th><th>EU/t</th><th>累计 EU</th><th>产热</th><th>散热</th></tr></thead><tbody>{tickRows.map(row=><tr key={row.game_tick} className={row.game_tick===currentTick?"selected-row":""} onClick={()=>setCurrentTick(row.game_tick)}><td>{fmt(row.game_tick)}</td><td>{fmt(row.seconds,2)}</td><td>{fmt(row.reactor_tick)}</td><td>{fmt(row.hull_heat)} / {fmt(row.max_hull_heat)}</td><td>{fmt(row.eu_per_tick,2)}</td><td>{fmt(row.total_eu)}</td><td>{fmt(row.generated_heat)}</td><td>{fmt(row.vented_heat)}</td></tr>)}</tbody></table>{!tickRows.length&&<div className="empty-table">尚无轨迹数据</div>}</div></section>
    </div>}

    {tab==="optimize"&&<div className="optimizer-layout"><section className="panel optimizer-form"><div className="panel-title">搜索约束</div><div className="form-grid"><label>反应堆仓<select value={columns-3} onChange={e=>resize(Number(e.target.value)+3)}>{[0,1,2,3,4,5,6].map(n=><option key={n} value={n}>{n}（{6*(n+3)} 格）</option>)}</select></label><label>燃料限制方式<select value={fuelMode} onChange={e=>setFuelMode(e.target.value as any)}><option value="separate">单/双/四联分别限制</option><option value="total_rods">实际棒数总上限</option></select></label>{fuelMode==="separate"?<><label>单联上限<input type="number" value={fuelLimits.single} onChange={e=>setFuelLimits(v=>({...v,single:Number(e.target.value)}))}/></label><label>双联上限<input type="number" value={fuelLimits.dual} onChange={e=>setFuelLimits(v=>({...v,dual:Number(e.target.value)}))}/></label><label>四联上限<input type="number" value={fuelLimits.quad} onChange={e=>setFuelLimits(v=>({...v,quad:Number(e.target.value)}))}/></label></>:<label>实际棒数上限<input type="number" value={fuelLimits.total_rods} onChange={e=>setFuelLimits(v=>({...v,total_rods:Number(e.target.value)}))}/></label>}</div>
        <h3>可用组件与库存上限</h3><div className="limit-grid">{components.filter(c=>!["empty","fuel"].includes(c.kind)).map(c=><label key={c.id} className={c.id in limits?"enabled":""}><input type="checkbox" checked={c.id in limits} onChange={()=>toggleLimit(c.id)}/><span className={`mini-icon kind-${c.kind}`}><ComponentIcon spec={c} small/></span><span>{c.name}</span><input type="number" min="0" disabled={!(c.id in limits)} value={limits[c.id]??0} onChange={e=>setLimits(v=>({...v,[c.id]:Number(e.target.value)}))}/></label>)}</div>
        <div className="optimizer-options"><div><h3>目标 Mark（分别计算）</h3><div className="mark-pills">{["I","II","III","IV","V"].map(m=><button className={marks.includes(m)?"on":""} onClick={()=>toggleMark(m)} key={m}>Mark {m}</button>)}</div></div><div><h3>求解模式</h3><select value={solver} onChange={e=>setSolver(e.target.value as any)}><option value="heuristic">限时启发式最优</option><option value="exhaustive">穷举并证明全局最优</option></select></div></div>
        {solver==="heuristic"&&<div className="form-grid search-tuning"><label>时间预算/秒<input type="number" min="1" value={search.seconds} onChange={e=>setSearch(v=>({...v,seconds:Number(e.target.value)}))}/></label><label>最大代数<input type="number" min="1" value={search.generations} onChange={e=>setSearch(v=>({...v,generations:Number(e.target.value)}))}/></label><label>种群规模<input type="number" min="10" value={search.population} onChange={e=>setSearch(v=>({...v,population:Number(e.target.value)}))}/></label><label>CPU 工作进程<input type="number" min="1" max="64" value={search.workers} onChange={e=>setSearch(v=>({...v,workers:Number(e.target.value)}))}/></label><label>随机种子<input type="number" value={search.seed} onChange={e=>setSearch(v=>({...v,seed:Number(e.target.value)}))}/></label></div>}
        {solver==="exhaustive"&&<div className="form-grid search-tuning"><label>全局枚举工作进程<input type="number" min="1" max="64" value={search.workers} onChange={e=>setSearch(v=>({...v,workers:Number(e.target.value)}))}/></label></div>}
        <button className="primary large" disabled={!marks.length||job?.status==="running"} onClick={runOptimization}>{solver==="exhaustive"?"计算方案数并开始穷举":"开始计算最优布局"}</button>
      </section><section className="panel optimization-results"><div className="panel-title"><span>搜索结果</span><div>{job?.status==="running"&&<button onClick={()=>fetch(`${API}/optimizations/${job.id}/cancel`,{method:"POST"})}>取消</button>}{job&&solver==="heuristic"&&["completed","cancelled"].includes(job.status)&&<button onClick={resumeOptimization}>继续改进</button>}</div></div>{job?<><div className="progress-card"><div><strong>{job.message}</strong><span>{solver==="exhaustive"&&enumerationEstimate?`已检查 ${fmt(job.checked||0)} / ${fmtIntegerString(enumerationEstimate)} 个有标签方案`:`已评估 ${fmt(job.evaluated)} 个布局`} · {fmt(job.elapsed_seconds,1)} 秒</span></div><b>{Math.round(job.progress*100)}%</b><div className="progress"><span style={{width:`${job.progress*100}%`}}/></div>{job.proven_global&&<em>全部有标签布局均已模拟或由功率上界证明，结果为全局最优</em>}</div>{marks.map(mark=><div className="leaderboard" key={mark}><h2>Mark {mark} <span>{job.leaderboards?.[mark]?.length||0} 个候选</span></h2>{job.leaderboards?.[mark]?.map((candidate:Candidate,i:number)=><button key={i} onClick={()=>loadCandidate(candidate,mark,i)}><b>#{i+1}</b><span><strong>{fmt(candidate.average_eu_per_tick,2)} EU/t</strong><small>{candidate.mark} · 安全 {fmt(candidate.safe_game_ticks)} tick · {candidate.component_count} 个组件</small></span><em>布局与曲线 →</em></button>)}{!job.leaderboards?.[mark]?.length&&<p>尚未找到精确匹配该 Mark 的可行布局。</p>}</div>)}</>:<div className="empty-results"><span>⚛</span><h2>等待搜索</h2><p>优化器会分别检查镜像方向，并为每个 Mark 的每组镜像布局保留最佳方向。排行榜只按平均 EU/t 排序。</p></div>}</section></div>}

    {tab==="rules"&&<div className="rules-page"><section className="panel rule-intro"><span className="eyebrow">规则集锁定</span><h2>IC2 Experimental 2.8.221</h2><p>20 game ticks 构成一次反应堆结算。先执行产热、散热与换热，再执行发电与耐久消耗。85% 堆热为 Mark 临界，100% 为融毁。</p><div className="formula"><code>EU/t = {"{5,10,20}"} × p</code><code>热量 = {"{2,4,8}"} × p(p+1)</code></div></section><section className="panel data-table"><div className="panel-title">组件数据注册表</div><table><thead><tr><th>组件</th><th>类别</th><th>热容量</th><th>耐久</th><th>棒数</th><th>规则说明</th></tr></thead><tbody>{components.filter(c=>c.id!=="empty").map(c=><tr key={c.id}><td><span className={`mini-icon kind-${c.kind}`}><ComponentIcon spec={c} small/></span>{c.name}</td><td>{KIND_LABELS[c.kind]}</td><td>{c.max_heat?fmt(c.max_heat):"—"}</td><td>{c.max_damage?fmt(c.max_damage):c.id==="iridium_reflector"?"无限":"—"}</td><td>{c.rod_count||"—"}</td><td>{c.kind==="fuel"?"寿命 20,000 个反应堆周期":c.kind==="reflector"?"向相邻燃料棒反射中子脉冲":c.kind==="plating"?"提高堆体最大热量":"按注册参数处理热量"}</td></tr>)}</tbody></table></section></div>}
    <footer><span>IC2 Reactor Optimizer · 本地计算，不上传布局</span><span>规则与贴图目标：IndustrialCraft² 2.8.221</span></footer>
  </main>;
}
