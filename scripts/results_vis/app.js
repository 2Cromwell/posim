/* ========== POSIM Visualization System ========== */

const EXPERIMENTS = {
  xibeiyuzhicai: {
    label: '西贝预制菜事件',
    dataDir: '../xibeiyuzhicai/data',
  },
  wudatushuguan: {
    label: '武大图书馆事件',
    dataDir: '../wudatushuguan/data',
  },
  tianjiaerhuan: {
    label: '添加二环事件',
    dataDir: '../tianjiaerhuan/data',
  },
};

let macroData = null;
let microData = null;
let eventsData = null;
let configData = null;
let charts = {};
let currentPostPage = 0;
const POSTS_PER_PAGE = 30;
let filteredPosts = [];
let processedData = null; // cached aggregations
let activeTab = 'overview';
let networkNeedsRender = true;

const ACTION_LABELS = {
  short_post: '短博文', long_post: '长博文', short_comment: '短评论',
  long_comment: '长评论', repost: '转发', repost_comment: '转发评论', like: '点赞'
};
const AGENT_LABELS = { citizen: '普通用户', kol: '大V', media: '媒体', government: '政府' };
const AGENT_ICONS = { citizen: '👤', kol: '⭐', media: '📺', government: '🏛️' };
const EMOTION_LABELS = {
  anger: '愤怒', disgust: '厌恶', sadness: '悲伤', fear: '恐惧',
  excitement: '兴奋', joy: '喜悦', neutral: '中性', surprise: '惊讶',
  anxiety: '焦虑', contempt: '鄙视', hope: '希望', disappointment: '失望'
};
const EMOTION_COLORS = {
  anger: '#ff6b6b', disgust: '#a78bfa', sadness: '#5b8def', fear: '#4ecdc4',
  excitement: '#ffd166', joy: '#06d6a0', neutral: '#9ea3b5', surprise: '#ff8c42',
  anxiety: '#e879a8', contempt: '#8b5cf6', hope: '#36d399', disappointment: '#6b7085'
};

// ========== Initialization ==========
document.addEventListener('DOMContentLoaded', () => {
  initEventSelector();
  initTabNavigation();
  initThemeToggle();
  loadDefaultExperiment();
});

function initEventSelector() {
  const eventSelect = document.getElementById('eventSelect');
  for (const [key, val] of Object.entries(EXPERIMENTS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = val.label;
    eventSelect.appendChild(opt);
  }
  eventSelect.value = 'xibeiyuzhicai';
  eventSelect.addEventListener('change', () => scanExperiments(eventSelect.value));
  scanExperiments('xibeiyuzhicai');
  document.getElementById('loadBtn').addEventListener('click', loadSelectedExperiment);
}

async function scanExperiments(eventKey) {
  const expSelect = document.getElementById('experimentSelect');
  expSelect.innerHTML = '<option value="">扫描中...</option>';
  const defaultMap = {
    xibeiyuzhicai: 'xibeiyuzhicai_baseline_20260223_145442_14B效果不错',
    wudatushuguan: 'wudatushuguan_baseline_20260221_021403_14B_行为分布好',
    tianjiaerhuan: 'tianjiaerhuan_baseline_20260221_152957_14B效果好',
  };
  try {
    const basePath = `../${eventKey}/output/`;
    const resp = await fetch(basePath);
    if (!resp.ok) throw new Error('Cannot list directory');
    const text = await resp.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, 'text/html');
    const links = doc.querySelectorAll('a');
    expSelect.innerHTML = '';
    const names = [];
    links.forEach(a => {
      const href = a.getAttribute('href') || '';
      const name = decodeURIComponent(href.replace(/\/$/, '').trim());
      if (name && name !== '..' && name !== '.' && !name.startsWith('.')) {
        names.push(name);
      }
    });
    if (names.length === 0) throw new Error('No experiments found');
    names.sort().reverse();
    names.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      expSelect.appendChild(opt);
    });
    const preferred = defaultMap[eventKey];
    if (preferred && names.includes(preferred)) {
      expSelect.value = preferred;
    }
  } catch (e) {
    console.warn('Directory listing not available, using hardcoded list');
    addHardcodedExperiments(eventKey, expSelect);
  }
}

function addHardcodedExperiments(eventKey, select) {
  const hardcoded = {
    xibeiyuzhicai: [
      'xibeiyuzhicai_baseline_20260223_145442_14B效果不错',
      'xibeiyuzhicai_baseline_20260223_210314',
      'xibeiyuzhicai_baseline_20260223_144652_ABM方法',
      'xibeiyuzhicai_baseline_20260223_144149',
      'xibeiyuzhicai_baseline_20260223_143436',
      'xibeiyuzhicai_baseline_20260223_142553',
      'xibeiyuzhicai_baseline_20260223_134658',
      'xibeiyuzhicai_baseline_20260223_133424',
      'xibeiyuzhicai_baseline_20260223_011758',
      'xibeiyuzhicai_baseline_20260223_005227',
      'xibeiyuzhicai_baseline_20260223_002913',
      'xibeiyuzhicai_baseline_20260222_235811',
      'xibeiyuzhicai_baseline_20260222_232639',
      'xibeiyuzhicai_baseline_20260222_182123',
      'xibeiyuzhicai_baseline_20260222_181959',
      'xibeiyuzhicai_baseline_20260222_172827',
      'xibeiyuzhicai_baseline_20260222_151046',
      'xibeiyuzhicai_baseline_20260222_150839',
    ],
    wudatushuguan: [
      'wudatushuguan_baseline_20260224_200017',
      'wudatushuguan_baseline_20260221_021403_14B_行为分布好',
      'wudatushuguan_baseline_20260220_120139_完整模拟_行为分布不好',
    ],
    tianjiaerhuan: [
      'tianjiaerhuan_baseline_20260221_152957_14B效果好',
      'tianjiaerhuan_baseline_20260214_220824_三天_热度吻合好',
      'tianjiaerhuan_baseline_20260219_213300_备选',
      'tianjiaerhuan_baseline_20260221_114905',
      'tianjiaerhuan_baseline_20260221_003522',
    ],
  };
  select.innerHTML = '';
  const list = hardcoded[eventKey] || [];
  if (list.length === 0) {
    select.innerHTML = '<option value="">无可用实验</option>';
    return;
  }
  list.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });
}

function loadDefaultExperiment() {
  const expSelect = document.getElementById('experimentSelect');
  setTimeout(() => {
    if (expSelect.value) loadSelectedExperiment();
  }, 300);
}

async function loadSelectedExperiment() {
  const eventKey = document.getElementById('eventSelect').value;
  const expName = document.getElementById('experimentSelect').value;
  if (!expName) return;

  const encodedExp = expName.split('/').map(s => encodeURIComponent(s)).join('/');
  const basePath = `../${eventKey}/output/${encodedExp}/simulation_results`;
  const dataDir = EXPERIMENTS[eventKey]?.dataDir || `../${eventKey}/data`;
  const cacheKey = `${eventKey}/${expName}`;

  showLoading(true);
  try {
    // Try cache first
    let cached = null;
    try {
      const cacheResp = await fetch(`/results_vis/api/cache/${encodeURIComponent(cacheKey)}`);
      if (cacheResp.ok) cached = await cacheResp.json();
    } catch {}

    if (cached) {
      macroData = cached.macro;
      microData = cached.micro;
      eventsData = cached.events || null;
      configData = cached.config || null;
      processedData = cached.processed;
      console.log(`从缓存加载: ${cacheKey}`);
    } else {
      const macroResp = await fetch(`${basePath}/macro_results.json`);
      if (!macroResp.ok) throw new Error(`无法加载 macro_results.json (${macroResp.status})，该实验可能没有仿真结果`);
      macroData = await macroResp.json();

      const microResp = await fetch(`${basePath}/micro_results.json`);
      if (!microResp.ok) throw new Error(`无法加载 micro_results.json (${microResp.status})`);
      microData = await microResp.json();

      try {
        const configResp = await fetch(`${basePath}/config.json`);
        configData = configResp.ok ? await configResp.json() : null;
      } catch { configData = null; }

      try {
        const eventsResp = await fetch(`${dataDir}/events.json`);
        eventsData = eventsResp.ok ? await eventsResp.json() : null;
      } catch { eventsData = null; }

      processedData = buildProcessedData();

      // Save to cache in background
      saveCache(cacheKey, { macro: macroData, micro: microData, events: eventsData, config: configData, processed: processedData });
    }

    const shortName = expName.length > 40 ? '...' + expName.slice(-37) : expName;
    document.getElementById('statusBadge').textContent = `已加载: ${shortName}`;
    document.getElementById('statusBadge').className = 'status-badge loaded';

    networkNeedsRender = true;
    renderAll();
  } catch (err) {
    alert('加载失败: ' + err.message + '\n\n请尝试选择其他实验');
    console.error(err);
  } finally {
    showLoading(false);
  }
}

function buildProcessedData() {
  const pd = {};

  // Per-post interaction counts
  pd.postInteractions = {};
  microData.forEach(a => {
    if (!a.target_post_id) return;
    if (!pd.postInteractions[a.target_post_id]) pd.postInteractions[a.target_post_id] = { reposts: 0, comments: 0, likes: 0, total: 0 };
    const pi = pd.postInteractions[a.target_post_id];
    if (a.action_type === 'repost' || a.action_type === 'repost_comment') pi.reposts++;
    else if (a.action_type.includes('comment')) pi.comments++;
    else if (a.action_type === 'like') pi.likes++;
    pi.total++;
  });

  // Assign each action a synthetic post_id for original posts
  pd.postIdMap = {};
  let postCounter = 0;
  microData.forEach((a, i) => {
    if (!a.target_post_id && (a.action_type === 'short_post' || a.action_type === 'long_post')) {
      const syntheticId = `synth_${a.user_id}_${postCounter++}`;
      pd.postIdMap[i] = syntheticId;
    }
  });

  // Topic frequencies from micro data (for top 10)
  pd.topicCounts = {};
  microData.forEach(a => {
    (a.topics || []).forEach(tp => {
      const clean = tp.replace(/#/g, '').trim();
      if (clean) pd.topicCounts[clean] = (pd.topicCounts[clean] || 0) + 1;
    });
  });

  // User stats
  pd.userStats = {};
  microData.forEach(a => {
    const u = a.username;
    if (!pd.userStats[u]) pd.userStats[u] = { total: 0, reposts: 0, comments: 0, posts: 0, likes: 0, agent_type: a.agent_type, user_id: a.user_id };
    const s = pd.userStats[u];
    s.total++;
    if (a.action_type === 'repost' || a.action_type === 'repost_comment') s.reposts++;
    if (a.action_type.includes('comment')) s.comments++;
    if (a.action_type === 'short_post' || a.action_type === 'long_post') s.posts++;
    if (a.action_type === 'like') s.likes++;
  });

  return pd;
}

async function saveCache(key, data) {
  try {
    await fetch('/results_vis/api/cache', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key, data }),
    });
    console.log(`缓存已保存: ${key}`);
  } catch (e) {
    console.warn('缓存保存失败:', e);
  }
}

function showLoading(show) {
  document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

// ========== Theme ==========
function initThemeToggle() {
  const toggle = document.getElementById('themeToggle');
  toggle.addEventListener('click', () => {
    const body = document.body;
    const isDark = body.classList.contains('dark-theme');
    body.classList.toggle('dark-theme', !isDark);
    body.classList.toggle('light-theme', isDark);
    document.getElementById('themeIcon').innerHTML = isDark
      ? '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
      : '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    Object.values(charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
  });
}

// ========== Tab Navigation ==========
function initTabNavigation() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeTab = btn.dataset.tab;
      document.getElementById('panel-' + activeTab).classList.add('active');
      setTimeout(() => {
        Object.values(charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
        if (activeTab === 'network' && networkNeedsRender && microData) {
          networkNeedsRender = false;
          buildNetworkGraph();
        }
      }, 150);
    });
  });
}

// ========== ECharts Helpers ==========
function getThemeColors() {
  const isDark = document.body.classList.contains('dark-theme');
  return {
    bg: isDark ? '#1e2235' : '#ffffff',
    text: isDark ? '#e8eaf0' : '#1a1d2e',
    subtext: isDark ? '#9ea3b5' : '#5a5f72',
    axis: isDark ? '#3a3f5a' : '#e0e3ea',
    tooltip: isDark ? '#252940' : '#ffffff',
    series: ['#5b8def','#ff6b6b','#4ecdc4','#ffd166','#a78bfa','#ff8c42','#06d6a0','#e879a8','#36d399','#8b5cf6'],
  };
}

function initChart(id) {
  const dom = document.getElementById(id);
  if (!dom) return null;
  if (charts[id]) { charts[id].dispose(); }
  const c = echarts.init(dom, null, { renderer: 'canvas' });
  charts[id] = c;
  return c;
}

function baseOption(title) {
  const t = getThemeColors();
  return {
    backgroundColor: 'transparent',
    textStyle: { color: t.text, fontFamily: "'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif" },
    title: title ? { text: title, textStyle: { color: t.text, fontSize: 14 }, left: 'center', top: 0 } : undefined,
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text, fontSize: 12 } },
    grid: { left: 60, right: 30, top: 40, bottom: 40, containLabel: false },
    color: t.series,
  };
}

// ========== Render All ==========
function renderAll() {
  if (!processedData && microData) processedData = buildProcessedData();
  renderOverview();
  renderPosts();
  renderHotList();
  renderNetwork();
  renderTopLists();
  renderCurves();
  renderAnalysis();
}

// ========== 1. Overview ==========
function renderOverview() {
  if (!macroData || !microData) return;
  const t = getThemeColors();

  const agentCounts = {};
  const uniqueUsers = new Set();
  microData.forEach(a => { agentCounts[a.agent_type] = (agentCounts[a.agent_type] || 0) + 1; uniqueUsers.add(a.user_id); });

  const statsHTML = [
    { icon: '📊', label: '总行为数', value: macroData.stats.total_actions.toLocaleString() },
    { icon: '👥', label: '参与用户', value: uniqueUsers.size },
    { icon: '🕐', label: '仿真步数', value: macroData.steps },
    { icon: '📝', label: '原创帖子', value: (macroData.stats.actions_by_type.short_post||0) + (macroData.stats.actions_by_type.long_post||0) },
    { icon: '💬', label: '评论总数', value: (macroData.stats.actions_by_type.short_comment||0) + (macroData.stats.actions_by_type.long_comment||0) },
    { icon: '🔄', label: '转发总数', value: (macroData.stats.actions_by_type.repost||0) + (macroData.stats.actions_by_type.repost_comment||0) },
    { icon: '❤️', label: '点赞总数', value: macroData.stats.actions_by_type.like || 0 },
    { icon: '🔥', label: '热搜话题', value: macroData.final_hot_search?.length || 0 },
  ].map(s => `<div class="stat-card"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`).join('');
  document.getElementById('statsCards').innerHTML = statsHTML;

  // Actions per step
  const stepsData = macroData.stats.actions_per_step || [];
  const timeEngine = macroData.time_engine;
  const timeLabels = generateTimeLabels(timeEngine, stepsData.length);
  
  const chartAPS = initChart('chart-actionsPerStep');
  chartAPS.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    xAxis: { type: 'category', data: timeLabels, axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
    series: [{ name: '行为数', type: 'line', data: stepsData, smooth: true, areaStyle: { opacity: 0.15 }, lineStyle: { width: 2 }, itemStyle: { color: '#5b8def' }, symbol: 'none' }],
  });

  // Active agents per step
  const activeData = macroData.stats.active_agents_per_step || [];
  const chartAA = initChart('chart-activeAgents');
  chartAA.setOption({
    ...baseOption(),
    xAxis: { type: 'category', data: timeLabels.slice(0, activeData.length), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
    series: [{ name: '活跃用户数', type: 'bar', data: activeData, itemStyle: { color: '#4ecdc4', borderRadius: [2,2,0,0] }, barMaxWidth: 6 }],
  });

  // Action types pie
  const chartAT = initChart('chart-actionTypes');
  chartAT.setOption({
    ...baseOption(),
    tooltip: { trigger: 'item', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    legend: { bottom: 0, textStyle: { color: t.subtext, fontSize: 11 } },
    series: [{
      type: 'pie', radius: ['35%','65%'], center: ['50%','45%'],
      label: { color: t.text, fontSize: 11, formatter: '{b}: {d}%' },
      data: Object.entries(macroData.stats.actions_by_type).map(([k,v]) => ({ name: ACTION_LABELS[k]||k, value: v })),
      emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.3)' } },
    }],
  });

  // Agent types pie
  const chartAG = initChart('chart-agentTypes');
  chartAG.setOption({
    ...baseOption(),
    tooltip: { trigger: 'item', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    legend: { bottom: 0, textStyle: { color: t.subtext, fontSize: 11 } },
    series: [{
      type: 'pie', radius: ['35%','65%'], center: ['50%','45%'], roseType: 'radius',
      label: { color: t.text, fontSize: 11, formatter: '{b}: {c}' },
      data: Object.entries(agentCounts).map(([k,v]) => ({ name: AGENT_LABELS[k]||k, value: v })),
    }],
  });

  // Intensity curve
  const intensityData = macroData.stats.intensity_history || [];
  const chartInt = initChart('chart-intensity');
  chartInt.setOption({
    ...baseOption(),
    xAxis: { type: 'category', data: timeLabels.slice(0, intensityData.length), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
    series: [{ name: 'Hawkes强度', type: 'line', data: intensityData, smooth: true, areaStyle: { opacity: 0.1, color: '#ff6b6b' }, lineStyle: { width: 2, color: '#ff6b6b' }, symbol: 'none' }],
  });

  // Emotion overview
  const emotionCounts = {};
  microData.forEach(a => { const em = a.emotion || 'neutral'; emotionCounts[em] = (emotionCounts[em] || 0) + 1; });
  const chartEO = initChart('chart-emotionOverview');
  chartEO.setOption({
    ...baseOption(),
    tooltip: { trigger: 'item', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    legend: { bottom: 0, textStyle: { color: t.subtext, fontSize: 11 } },
    series: [{
      type: 'pie', radius: ['30%','60%'], center: ['50%','45%'],
      label: { color: t.text, fontSize: 11, formatter: '{b}: {d}%' },
      data: Object.entries(emotionCounts).sort((a,b) => b[1]-a[1]).map(([k,v]) => ({
        name: EMOTION_LABELS[k] || k, value: v, itemStyle: { color: EMOTION_COLORS[k] || '#9ea3b5' }
      })),
    }],
  });
}

function generateTimeLabels(timeEngine, count) {
  if (!timeEngine) return Array.from({length: count}, (_,i) => `Step ${i}`);
  const start = new Date(timeEngine.start_time);
  const gran = (timeEngine.granularity || 10) * 60 * 1000;
  return Array.from({length: count}, (_, i) => {
    const d = new Date(start.getTime() + i * gran);
    return `${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  });
}

// ========== 2. Posts ==========
let postsInitialized = false;
function renderPosts() {
  if (!microData) return;
  currentPostPage = 0;

  if (!postsInitialized) {
    postsInitialized = true;
    document.getElementById('postSearch').addEventListener('input', debounce(() => { currentPostPage = 0; applyPostFilters(); }, 300));
    document.getElementById('postSortBy').addEventListener('change', () => { currentPostPage = 0; applyPostFilters(); });
    document.getElementById('postActionFilter').addEventListener('change', () => { currentPostPage = 0; applyPostFilters(); });

    document.querySelectorAll('#agentFilterChips .chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#agentFilterChips .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentPostPage = 0;
        applyPostFilters();
      });
    });

    document.getElementById('loadMorePosts').addEventListener('click', () => {
      currentPostPage++;
      renderPostCards(true);
    });
  }

  applyPostFilters();
}

function applyPostFilters() {
  const search = document.getElementById('postSearch').value.toLowerCase();
  const agentFilter = document.querySelector('#agentFilterChips .chip.active')?.dataset.filter || 'all';
  const actionFilter = document.getElementById('postActionFilter').value;
  const sortBy = document.getElementById('postSortBy').value;

  filteredPosts = microData.filter(a => {
    if (agentFilter !== 'all' && a.agent_type !== agentFilter) return false;
    if (actionFilter !== 'all' && a.action_type !== actionFilter) return false;
    if (search) {
      const txt = `${a.content||''} ${a.username||''} ${(a.topics||[]).join(' ')} ${a.target_author||''}`.toLowerCase();
      if (!txt.includes(search)) return false;
    }
    return true;
  });

  // Compute influence scores
  const interactionCount = {};
  microData.forEach(a => {
    if (a.target_post_id) interactionCount[a.target_post_id] = (interactionCount[a.target_post_id] || 0) + 1;
  });
  const postIds = {};
  filteredPosts.forEach((a, i) => {
    if (!a.target_post_id) postIds[`gen_${a.user_id}_${i}`] = i;
  });

  if (sortBy === 'influence') {
    filteredPosts.sort((a, b) => {
      const aScore = (interactionCount[a.target_post_id] || 0) + (a.emotion_intensity || 0);
      const bScore = (interactionCount[b.target_post_id] || 0) + (b.emotion_intensity || 0);
      return bScore - aScore;
    });
  } else if (sortBy === 'time_desc') {
    filteredPosts.sort((a, b) => new Date(b.time) - new Date(a.time));
  } else if (sortBy === 'time_asc') {
    filteredPosts.sort((a, b) => new Date(a.time) - new Date(b.time));
  } else if (sortBy === 'emotion') {
    filteredPosts.sort((a, b) => (b.emotion_intensity || 0) - (a.emotion_intensity || 0));
  }

  document.getElementById('postsCount').textContent = `共 ${filteredPosts.length} 条结果`;
  renderPostCards(false);
}

function renderPostCards(append) {
  const feed = document.getElementById('postsFeed');
  if (!append) feed.innerHTML = '';

  const start = append ? currentPostPage * POSTS_PER_PAGE : 0;
  const end = Math.min((append ? currentPostPage + 1 : 1) * POSTS_PER_PAGE, filteredPosts.length);
  const slice = filteredPosts.slice(start, end);

  const pi = processedData?.postInteractions || {};

  slice.forEach(post => {
    const card = document.createElement('div');
    card.className = 'post-card';

    const actionCat = getActionCategory(post.action_type);
    const contentHTML = highlightContent(post.content || post.text || '');
    const emotionClass = post.emotion || 'neutral';

    let targetHTML = '';
    if (post.target_post_id && post.target_author) {
      targetHTML = `<div class="post-target">↩️ 回复/转发 @${escapeHtml(post.target_author)} 的帖子</div>`;
    }

    // Interaction stats: for original posts, count interactions targeting them
    // For comments/reposts, show what post they interact with
    const postId = post.target_post_id || post._syntheticId;
    const interactions = pi[postId] || { reposts: 0, comments: 0, likes: 0, total: 0 };
    const isOriginal = !post.target_post_id;

    card.innerHTML = `
      <div class="post-header">
        <div class="post-avatar ${post.agent_type}">${AGENT_ICONS[post.agent_type] || '👤'}</div>
        <div class="post-user-info">
          <div class="post-username">${escapeHtml(post.username)} <span class="post-type-badge ${actionCat}">${ACTION_LABELS[post.action_type] || post.action_type}</span></div>
          <div class="post-meta">
            <span>${AGENT_LABELS[post.agent_type] || post.agent_type}</span>
            <span>·</span>
            <span>${formatTime(post.time)}</span>
            ${post.topics?.length ? `<span>· ${post.topics.map(t=>`<span class="hashtag" style="color:var(--accent);font-size:11px">${t}</span>`).join(' ')}</span>` : ''}
          </div>
        </div>
      </div>
      ${targetHTML}
      <div class="post-content">${contentHTML}</div>
      <div class="post-footer">
        <span class="post-emotion ${emotionClass}">${EMOTION_LABELS[post.emotion] || post.emotion || '未知'} ${post.emotion_intensity != null ? (post.emotion_intensity * 100).toFixed(0) + '%' : ''}</span>
        <span class="post-stance">🎯 ${post.stance || '未知'} ${post.stance_intensity != null ? (post.stance_intensity * 100).toFixed(0) + '%' : ''}</span>
        ${post.style ? `<span style="font-size:12px;color:var(--text-muted)">📝 ${post.style}</span>` : ''}
        ${post.narrative ? `<span style="font-size:12px;color:var(--text-muted)">📖 ${post.narrative}</span>` : ''}
      </div>
      <div class="post-interactions">
        <span class="interact-item"><span class="interact-icon">🔄</span> <span class="interact-count">${interactions.reposts}</span> 转发</span>
        <span class="interact-item"><span class="interact-icon">💬</span> <span class="interact-count">${interactions.comments}</span> 评论</span>
        <span class="interact-item"><span class="interact-icon">❤️</span> <span class="interact-count">${interactions.likes}</span> 点赞</span>
        <span class="interact-item" style="margin-left:auto;"><span class="interact-icon">📊</span> <span class="interact-count">${interactions.total}</span> 互动</span>
      </div>
    `;
    feed.appendChild(card);
  });

  const loadMore = document.getElementById('loadMorePosts');
  loadMore.style.display = end >= filteredPosts.length ? 'none' : 'block';
}

function getActionCategory(type) {
  if (type.includes('repost')) return 'repost';
  if (type.includes('comment')) return 'comment';
  return 'post';
}

function highlightContent(text) {
  let html = escapeHtml(text);
  html = html.replace(/#[^#]+#/g, '<span class="hashtag">$&</span>');
  html = html.replace(/@[\w\u4e00-\u9fff-]+/g, '<span class="mention">$&</span>');
  return html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatTime(t) {
  if (!t) return '';
  const d = new Date(t);
  return `${d.getMonth()+1}月${d.getDate()}日 ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

function debounce(fn, delay) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

// ========== 3. Hot List / Event Timeline ==========
function renderHotList() {
  if (!macroData) return;
  const t = getThemeColors();

  // Hot search list - merge from macro final_hot_search + micro topic counts to get top 10
  const topicCounts = processedData?.topicCounts || {};
  const existingTopics = new Set();
  const mergedHotList = [];

  (macroData.final_hot_search || []).forEach(item => {
    const topic = item.topic.replace(/#/g, '').trim();
    existingTopics.add(topic);
    mergedHotList.push({
      topic: item.topic,
      score: item.heat_score || item.score || 0,
      mentions: item.mentions || topicCounts[topic] || 0,
    });
  });

  Object.entries(topicCounts)
    .filter(([t]) => !existingTopics.has(t))
    .sort((a, b) => b[1] - a[1])
    .forEach(([topic, count]) => {
      mergedHotList.push({ topic: `#${topic}#`, score: count, mentions: count });
    });

  mergedHotList.sort((a, b) => b.score - a.score);
  const hotList = mergedHotList.slice(0, 10);
  const hotHTML = hotList.map((item, i) => {
    const rankClass = i === 0 ? 'top1' : i === 1 ? 'top2' : i === 2 ? 'top3' : '';
    return `<div class="hot-item">
      <div class="hot-rank ${rankClass}">${i + 1}</div>
      <div class="hot-topic">${escapeHtml(item.topic)}</div>
      <div class="hot-score">🔥 ${item.score?.toFixed?.(1) ?? item.score}</div>
      <div class="hot-mentions">${item.mentions} 提及</div>
    </div>`;
  }).join('');
  document.getElementById('hotSearchList').innerHTML = hotHTML || '<p style="color:var(--text-muted);padding:20px">暂无热搜数据</p>';

  // Event timeline
  if (eventsData && eventsData.length > 0) {
    const timelineHTML = eventsData.map(ev => `
      <div class="event-item">
        <div class="event-time">${formatTime(ev.time)} <span class="event-type-badge">${ev.type === 'global_broadcast' ? '全局事件' : '节点帖子'}</span></div>
        <div class="event-title">${escapeHtml(ev.topic)}</div>
        <div class="event-desc">${escapeHtml(ev.content?.substring(0, 200) || '')}</div>
      </div>
    `).join('');
    document.getElementById('eventTimeline').innerHTML = timelineHTML;
  } else {
    document.getElementById('eventTimeline').innerHTML = '<p style="color:var(--text-muted);padding:20px">暂无事件数据</p>';
  }

  // Hot search evolution chart
  const history = macroData.hot_search_history || [];
  if (history.length > 0) {
    const allTopics = new Set();
    history.forEach(h => h.topics?.forEach(tp => allTopics.add(tp.topic)));
    const topicArr = [...allTopics];
    const times = history.map(h => h.time?.replace('T', ' ') || '');

    const series = topicArr.slice(0, 8).map(topic => ({
      name: topic,
      type: 'line',
      smooth: true,
      symbol: 'none',
      stack: 'total',
      areaStyle: { opacity: 0.25 },
      lineStyle: { width: 2 },
      data: history.map(h => {
        const found = h.topics?.find(tp => tp.topic === topic);
        return found ? (found.score || found.mentions || 0) : 0;
      }),
    }));

    const chartHSE = initChart('chart-hotSearchEvolution');
    chartHSE.setOption({
      ...baseOption(),
      legend: { top: 0, textStyle: { color: t.subtext, fontSize: 11 }, type: 'scroll' },
      tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
      xAxis: { type: 'category', data: times, axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
      yAxis: { type: 'value', name: '热度', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
      grid: { left: 60, right: 30, top: 50, bottom: 60 },
      series,
    });
  }
}

// ========== 4. Network ==========
let networkInitialized = false;
function renderNetwork() {
  if (!microData) return;
  if (!networkInitialized) {
    networkInitialized = true;
    document.getElementById('renderNetworkBtn').addEventListener('click', () => { buildNetworkGraph(); });
  }
  // Only render if tab is visible; otherwise flag for deferred render
  if (activeTab === 'network') {
    networkNeedsRender = false;
    setTimeout(() => buildNetworkGraph(), 200);
  } else {
    networkNeedsRender = true;
  }
}

function buildNetworkGraph() {
  const t = getThemeColors();
  const networkType = document.getElementById('networkType').value;
  const nodeSize = parseInt(document.getElementById('nodeSize').value);
  const edgeWidth = parseFloat(document.getElementById('edgeWidth').value);
  const maxNodes = parseInt(document.getElementById('maxNodes').value);

  // Aggregate edges with counts
  const edgeMap = {};
  const nodeDegree = {};
  const nodeInDeg = {};
  const nodeOutDeg = {};

  microData.forEach(a => {
    if (!a.target_author || !a.username || a.username === a.target_author) return;
    const isRepost = a.action_type === 'repost' || a.action_type === 'repost_comment';
    const isComment = a.action_type.includes('comment');

    if (networkType === 'repost' && !isRepost) return;
    if (networkType === 'comment' && !isComment) return;

    const eKey = `${a.username}→${a.target_author}`;
    edgeMap[eKey] = (edgeMap[eKey] || 0) + 1;

    nodeDegree[a.username] = (nodeDegree[a.username] || 0) + 1;
    nodeDegree[a.target_author] = (nodeDegree[a.target_author] || 0) + 1;
    nodeOutDeg[a.username] = (nodeOutDeg[a.username] || 0) + 1;
    nodeInDeg[a.target_author] = (nodeInDeg[a.target_author] || 0) + 1;
  });

  const sortedNodes = Object.entries(nodeDegree).sort((a, b) => b[1] - a[1]).slice(0, maxNodes);
  const nodeSet = new Set(sortedNodes.map(n => n[0]));

  const agentTypeMap = {};
  microData.forEach(a => { if (a.username) agentTypeMap[a.username] = a.agent_type; });

  const agentColors = { citizen: '#5b8def', kol: '#ffd166', media: '#a78bfa', government: '#4ecdc4' };
  const catIndexMap = { citizen: 0, kol: 1, media: 2, government: 3 };

  const graphNodes = sortedNodes.map(([name, degree]) => {
    const agType = agentTypeMap[name] || 'citizen';
    return {
      name,
      symbolSize: Math.max(nodeSize * 0.5, Math.min(nodeSize + Math.log2(degree + 1) * 5, 55)),
      itemStyle: { color: agentColors[agType] || '#5b8def', borderColor: 'rgba(255,255,255,0.3)', borderWidth: 1 },
      category: catIndexMap[agType] ?? 0,
      value: degree,
      _agentType: agType,
      _inDeg: nodeInDeg[name] || 0,
      _outDeg: nodeOutDeg[name] || 0,
    };
  });

  const graphEdges = [];
  Object.entries(edgeMap).forEach(([key, count]) => {
    const [src, tgt] = key.split('→');
    if (nodeSet.has(src) && nodeSet.has(tgt)) {
      graphEdges.push({
        source: src,
        target: tgt,
        value: count,
        lineStyle: { width: Math.min(edgeWidth + Math.log2(count) * 0.5, 4), opacity: Math.min(0.2 + count * 0.05, 0.7), curveness: 0.3 },
      });
    }
  });

  const categories = [
    { name: AGENT_LABELS.citizen || '普通用户' },
    { name: AGENT_LABELS.kol || '大V' },
    { name: AGENT_LABELS.media || '媒体' },
    { name: AGENT_LABELS.government || '政府' },
  ];

  console.log(`网络渲染: ${graphNodes.length} 节点, ${graphEdges.length} 边`);

  const chart = initChart('chart-network');
  chart.setOption({
    backgroundColor: 'transparent',
    textStyle: { fontFamily: "'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif" },
    tooltip: {
      trigger: 'item',
      backgroundColor: t.tooltip,
      borderColor: t.axis,
      textStyle: { color: t.text, fontSize: 12 },
      formatter: p => {
        if (p.dataType === 'node') {
          return `<b>${p.name}</b><br/>类型: ${AGENT_LABELS[p.data._agentType]||p.data._agentType}<br/>总交互: ${p.value}<br/>入度: ${p.data._inDeg} / 出度: ${p.data._outDeg}`;
        }
        return `${p.data.source} → ${p.data.target}<br/>交互 ${p.data.value} 次`;
      },
    },
    legend: { data: categories.map(c => c.name), top: 5, textStyle: { color: t.subtext, fontSize: 12 } },
    animationDuration: 500,
    animationDurationUpdate: 300,
    series: [{
      type: 'graph',
      layout: 'force',
      data: graphNodes,
      links: graphEdges,
      categories,
      roam: true,
      draggable: true,
      force: {
        repulsion: Math.max(80, 300 - graphNodes.length),
        gravity: 0.08,
        edgeLength: [60, 250],
        friction: 0.6,
        layoutAnimation: true,
      },
      label: {
        show: graphNodes.length < 60,
        fontSize: 10,
        color: t.text,
        position: 'right',
        formatter: '{b}',
      },
      emphasis: {
        focus: 'adjacency',
        label: { show: true, fontSize: 13, fontWeight: 'bold' },
        lineStyle: { width: 4, opacity: 0.9 },
      },
      lineStyle: { color: 'source', opacity: 0.25, curveness: 0.3 },
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: [0, 6],
    }],
  });

  // Click handler for node info panel
  chart.off('click');
  chart.on('click', { dataType: 'node' }, params => {
    showNodeInfo(params.data);
  });
}

function showNodeInfo(nodeData) {
  const panel = document.getElementById('nodeInfoContent');
  const name = nodeData.name;
  const stats = processedData?.userStats?.[name] || {};
  const agType = nodeData._agentType || stats.agent_type || 'unknown';

  // Gather recent actions for this user
  const userActions = microData.filter(a => a.username === name).slice(0, 8);
  const emotions = {};
  const stances = {};
  microData.forEach(a => {
    if (a.username !== name) return;
    const em = a.emotion || 'neutral';
    emotions[em] = (emotions[em] || 0) + 1;
    if (a.stance) stances[a.stance] = (stances[a.stance] || 0) + 1;
  });
  const topEmotion = Object.entries(emotions).sort((a, b) => b[1] - a[1])[0];
  const topStance = Object.entries(stances).sort((a, b) => b[1] - a[1])[0];

  panel.innerHTML = `
    <div class="info-row"><span class="info-label">用户名</span><span class="info-value">${escapeHtml(name)}</span></div>
    <div class="info-row"><span class="info-label">类型</span><span class="info-value">${AGENT_ICONS[agType]||''} ${AGENT_LABELS[agType]||agType}</span></div>
    <div class="info-row"><span class="info-label">总交互度</span><span class="info-value">${nodeData.value}</span></div>
    <div class="info-row"><span class="info-label">入度 / 出度</span><span class="info-value">${nodeData._inDeg} / ${nodeData._outDeg}</span></div>
    <div class="info-row"><span class="info-label">总行为数</span><span class="info-value">${stats.total || 0}</span></div>
    <div class="info-row"><span class="info-label">发帖</span><span class="info-value">${stats.posts || 0}</span></div>
    <div class="info-row"><span class="info-label">评论</span><span class="info-value">${stats.comments || 0}</span></div>
    <div class="info-row"><span class="info-label">转发</span><span class="info-value">${stats.reposts || 0}</span></div>
    <div class="info-row"><span class="info-label">点赞</span><span class="info-value">${stats.likes || 0}</span></div>
    <div class="info-row"><span class="info-label">主要情绪</span><span class="info-value">${topEmotion ? `${EMOTION_LABELS[topEmotion[0]]||topEmotion[0]} (${topEmotion[1]}次)` : '-'}</span></div>
    <div class="info-row"><span class="info-label">主要立场</span><span class="info-value">${topStance ? `${topStance[0]} (${topStance[1]}次)` : '-'}</span></div>

    <div class="info-section">
      <div class="info-section-title">📝 最近行为</div>
      ${userActions.map(a => `
        <div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;">
          <span style="color:var(--accent)">${ACTION_LABELS[a.action_type]||a.action_type}</span>
          <span style="color:var(--text-muted)"> · ${formatTime(a.time)}</span>
          <div style="margin-top:3px;color:var(--text-secondary);line-height:1.5">${escapeHtml((a.content||'').substring(0, 80))}${(a.content||'').length > 80 ? '...' : ''}</div>
        </div>
      `).join('')}
    </div>
  `;
}

// ========== 5. Top Lists ==========
function renderTopLists() {
  if (!microData) return;
  const t = getThemeColors();

  const repostsByUser = {}, repostedPosts = {}, commentsByUser = {}, commentedPosts = {}, postsByUser = {}, allActionsByUser = {};

  microData.forEach(a => {
    const u = a.username;
    allActionsByUser[u] = (allActionsByUser[u] || 0) + 1;

    if (a.action_type === 'repost' || a.action_type === 'repost_comment') {
      repostsByUser[u] = (repostsByUser[u] || 0) + 1;
      if (a.target_post_id) repostedPosts[a.target_author || a.target_post_id] = (repostedPosts[a.target_author || a.target_post_id] || 0) + 1;
    }
    if (a.action_type.includes('comment')) {
      commentsByUser[u] = (commentsByUser[u] || 0) + 1;
      if (a.target_post_id) commentedPosts[a.target_author || a.target_post_id] = (commentedPosts[a.target_author || a.target_post_id] || 0) + 1;
    }
    if (a.action_type === 'short_post' || a.action_type === 'long_post') {
      postsByUser[u] = (postsByUser[u] || 0) + 1;
    }
  });

  renderTopList('topReposters', repostsByUser, '次转发');
  renderTopList('topReposted', repostedPosts, '次被转发');
  renderTopList('topCommenters', commentsByUser, '次评论');
  renderTopList('topCommented', commentedPosts, '次被评论');
  renderTopList('topPosters', postsByUser, '次发帖');
  renderTopList('topActive', allActionsByUser, '次行为');

  // Top users bar chart
  const topUsers = Object.entries(allActionsByUser).sort((a,b) => b[1]-a[1]).slice(0, 20);
  const chartTU = initChart('chart-topUsersBar');
  chartTU.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    grid: { left: 120, right: 30, top: 10, bottom: 30 },
    xAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'category', data: topUsers.map(u=>u[0]).reverse(), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 11, width: 100, overflow: 'truncate' } },
    series: [{ type: 'bar', data: topUsers.map(u=>u[1]).reverse(), itemStyle: { color: new echarts.graphic.LinearGradient(0,0,1,0, [{offset:0,color:'#5b8def'},{offset:1,color:'#4ecdc4'}]), borderRadius: [0,4,4,0] } }],
  });

  // Top interacted posts
  const topInteracted = Object.entries(repostedPosts).sort((a,b) => b[1]-a[1]).slice(0, 20);
  const chartTP = initChart('chart-topPostsBar');
  chartTP.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    grid: { left: 120, right: 30, top: 10, bottom: 30 },
    xAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'category', data: topInteracted.map(u=>u[0]).reverse(), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 11, width: 100, overflow: 'truncate' } },
    series: [{ type: 'bar', data: topInteracted.map(u=>u[1]).reverse(), itemStyle: { color: new echarts.graphic.LinearGradient(0,0,1,0, [{offset:0,color:'#ff6b6b'},{offset:1,color:'#ffd166'}]), borderRadius: [0,4,4,0] } }],
  });
}

function renderTopList(containerId, data, suffix) {
  const sorted = Object.entries(data).sort((a,b) => b[1]-a[1]).slice(0, 15);
  const max = sorted.length > 0 ? sorted[0][1] : 1;
  const html = sorted.map(([name, val], i) => `
    <div class="top-item">
      <div class="top-rank">${i + 1}</div>
      <div class="top-name" title="${escapeHtml(name)}">
        ${escapeHtml(name)}
        <div class="top-bar-fill" style="width:${(val/max*100).toFixed(1)}%"></div>
      </div>
      <div class="top-value">${val} ${suffix}</div>
    </div>
  `).join('');
  document.getElementById(containerId).innerHTML = html || '<p style="color:var(--text-muted);padding:10px">暂无数据</p>';
}

// ========== 6. Curves ==========
let curvesInitialized = false;
function renderCurves() {
  if (!microData) return;
  const t = getThemeColors();

  if (!curvesInitialized) {
    curvesInitialized = true;
    document.getElementById('curveMetric').addEventListener('change', renderMainCurve);
    document.getElementById('curveGroupBy').addEventListener('change', renderMainCurve);
  }
  renderMainCurve();
  renderEmotionIntensityTime();
  renderStanceIntensityTime();
  renderEmotionByAgent();
  renderStanceByAgent();
}

function renderMainCurve() {
  const metric = document.getElementById('curveMetric').value;
  const groupBy = document.getElementById('curveGroupBy').value;
  const t = getThemeColors();

  const titles = { emotion: '情绪变化趋势', stance: '立场变化趋势', style: '表达风格变化', narrative: '叙事策略变化', emotion_intensity: '情绪强度变化', stance_intensity: '立场强度变化' };
  document.getElementById('curveTitle').textContent = titles[metric] || metric;

  const timeSlots = {};
  microData.forEach(a => {
    const time = a.time?.substring(0, 16) || 'unknown';
    const group = groupBy === 'agent_type' ? (a.agent_type || 'unknown') : 'all';
    const key = `${time}|${group}`;
    if (!timeSlots[key]) timeSlots[key] = { values: {}, count: 0, intensitySum: 0 };
    
    if (metric === 'emotion_intensity') {
      timeSlots[key].intensitySum += (a.emotion_intensity || 0);
      timeSlots[key].count++;
    } else if (metric === 'stance_intensity') {
      timeSlots[key].intensitySum += (a.stance_intensity || 0);
      timeSlots[key].count++;
    } else {
      const val = a[metric] || 'unknown';
      timeSlots[key].values[val] = (timeSlots[key].values[val] || 0) + 1;
      timeSlots[key].count++;
    }
  });

  const allTimes = [...new Set(Object.keys(timeSlots).map(k => k.split('|')[0]))].sort();
  const allGroups = [...new Set(Object.keys(timeSlots).map(k => k.split('|')[1]))];

  const chart = initChart('chart-curveTrend');

  if (metric === 'emotion_intensity' || metric === 'stance_intensity') {
    const series = allGroups.map((group, gi) => ({
      name: groupBy === 'agent_type' ? (AGENT_LABELS[group] || group) : '整体',
      type: 'line',
      smooth: true,
      symbol: 'none',
      data: allTimes.map(time => {
        const slot = timeSlots[`${time}|${group}`];
        return slot ? +(slot.intensitySum / slot.count).toFixed(3) : null;
      }),
    }));
    chart.setOption({
      ...baseOption(),
      legend: { top: 0, textStyle: { color: t.subtext } },
      xAxis: { type: 'category', data: allTimes.map(t => t.replace('T',' ')), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
      yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
      grid: { left: 60, right: 30, top: 40, bottom: 60 },
      series,
    });
  } else {
    const allValues = new Set();
    Object.values(timeSlots).forEach(s => Object.keys(s.values).forEach(v => allValues.add(v)));
    const valueArr = [...allValues];
    
    if (groupBy === 'all') {
      const series = valueArr.slice(0, 10).map(val => ({
        name: (metric === 'emotion' ? EMOTION_LABELS[val] : val) || val,
        type: 'line',
        smooth: true,
        stack: 'total',
        areaStyle: { opacity: 0.3 },
        symbol: 'none',
        data: allTimes.map(time => {
          const slot = timeSlots[`${time}|all`];
          return slot?.values[val] || 0;
        }),
        ...(metric === 'emotion' && EMOTION_COLORS[val] ? { itemStyle: { color: EMOTION_COLORS[val] }, lineStyle: { color: EMOTION_COLORS[val] } } : {}),
      }));
      chart.setOption({
        ...baseOption(),
        legend: { top: 0, textStyle: { color: t.subtext, fontSize: 11 }, type: 'scroll' },
        xAxis: { type: 'category', data: allTimes.map(t => t.replace('T',' ')), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
        yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
        dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
        grid: { left: 60, right: 30, top: 50, bottom: 60 },
        series,
      });
    } else {
      const topValue = valueArr.sort((a,b) => {
        let ac = 0, bc = 0;
        Object.values(timeSlots).forEach(s => { ac += s.values[a]||0; bc += s.values[b]||0; });
        return bc - ac;
      }).slice(0, 6);

      const series = [];
      allGroups.forEach(group => {
        topValue.forEach(val => {
          series.push({
            name: `${AGENT_LABELS[group]||group}-${(metric==='emotion'?EMOTION_LABELS[val]:val)||val}`,
            type: 'line', smooth: true, symbol: 'none',
            data: allTimes.map(time => {
              const slot = timeSlots[`${time}|${group}`];
              return slot?.values[val] || 0;
            }),
          });
        });
      });
      chart.setOption({
        ...baseOption(),
        legend: { top: 0, textStyle: { color: t.subtext, fontSize: 10 }, type: 'scroll' },
        xAxis: { type: 'category', data: allTimes.map(t => t.replace('T',' ')), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
        yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
        dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
        grid: { left: 60, right: 30, top: 50, bottom: 60 },
        series,
      });
    }
  }
}

function renderEmotionIntensityTime() {
  const t = getThemeColors();
  const timeSlots = {};
  microData.forEach(a => {
    const time = a.time?.substring(0, 16) || 'unknown';
    if (!timeSlots[time]) timeSlots[time] = { sum: 0, count: 0 };
    timeSlots[time].sum += (a.emotion_intensity || 0);
    timeSlots[time].count++;
  });
  const times = Object.keys(timeSlots).sort();
  const chart = initChart('chart-emotionIntensityTime');
  chart.setOption({
    ...baseOption(),
    xAxis: { type: 'category', data: times.map(t => t.replace('T',' ')), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', min: 0, max: 1, axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    dataZoom: [{ type: 'inside' }],
    series: [{ name: '平均情绪强度', type: 'line', smooth: true, symbol: 'none', data: times.map(time => +(timeSlots[time].sum / timeSlots[time].count).toFixed(3)),
      areaStyle: { opacity: 0.15, color: '#ff6b6b' }, lineStyle: { color: '#ff6b6b', width: 2 } }],
  });
}

function renderStanceIntensityTime() {
  const t = getThemeColors();
  const timeSlots = {};
  microData.forEach(a => {
    const time = a.time?.substring(0, 16) || 'unknown';
    if (!timeSlots[time]) timeSlots[time] = { sum: 0, count: 0 };
    timeSlots[time].sum += (a.stance_intensity || 0);
    timeSlots[time].count++;
  });
  const times = Object.keys(timeSlots).sort();
  const chart = initChart('chart-stanceIntensityTime');
  chart.setOption({
    ...baseOption(),
    xAxis: { type: 'category', data: times.map(t => t.replace('T',' ')), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', min: 0, max: 1, axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    dataZoom: [{ type: 'inside' }],
    series: [{ name: '平均立场强度', type: 'line', smooth: true, symbol: 'none', data: times.map(time => +(timeSlots[time].sum / timeSlots[time].count).toFixed(3)),
      areaStyle: { opacity: 0.15, color: '#4ecdc4' }, lineStyle: { color: '#4ecdc4', width: 2 } }],
  });
}

function renderEmotionByAgent() {
  const t = getThemeColors();
  const agentEmotions = {};
  microData.forEach(a => {
    const agent = a.agent_type || 'unknown';
    const emotion = a.emotion || 'neutral';
    if (!agentEmotions[agent]) agentEmotions[agent] = {};
    agentEmotions[agent][emotion] = (agentEmotions[agent][emotion] || 0) + 1;
  });

  const agents = Object.keys(agentEmotions);
  const emotions = [...new Set(microData.map(a => a.emotion || 'neutral'))];

  const chart = initChart('chart-emotionByAgent');
  chart.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    legend: { top: 0, textStyle: { color: t.subtext, fontSize: 11 }, type: 'scroll' },
    grid: { left: 80, right: 30, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: agents.map(a => AGENT_LABELS[a]||a), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    series: emotions.map(em => ({
      name: EMOTION_LABELS[em] || em,
      type: 'bar',
      stack: 'total',
      data: agents.map(ag => agentEmotions[ag]?.[em] || 0),
      itemStyle: { color: EMOTION_COLORS[em] || undefined },
    })),
  });
}

function renderStanceByAgent() {
  const t = getThemeColors();
  const agentStances = {};
  microData.forEach(a => {
    const agent = a.agent_type || 'unknown';
    const stance = a.stance || '未知';
    if (!agentStances[agent]) agentStances[agent] = {};
    agentStances[agent][stance] = (agentStances[agent][stance] || 0) + 1;
  });

  const agents = Object.keys(agentStances);
  const stances = [...new Set(microData.map(a => a.stance || '未知'))];

  const chart = initChart('chart-stanceByAgent');
  chart.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    legend: { top: 0, textStyle: { color: t.subtext, fontSize: 11 }, type: 'scroll' },
    grid: { left: 80, right: 30, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: agents.map(a => AGENT_LABELS[a]||a), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    series: stances.slice(0, 10).map(st => ({
      name: st,
      type: 'bar',
      stack: 'total',
      data: agents.map(ag => agentStances[ag]?.[st] || 0),
    })),
  });
}

// ========== 7. Analysis ==========
function renderAnalysis() {
  if (!microData) return;
  const t = getThemeColors();

  renderWordCloud();
  renderTopicFreq();
  renderHourlyDist();
  renderDailyDist();
  renderEmotionStanceScatter();
  renderActionAgentHeatmap();
  renderSankey();
  renderEmotionRadar();
  renderCumulativeActions();
}

function renderWordCloud() {
  const wordFreq = {};
  microData.forEach(a => {
    const text = a.content || a.text || '';
    const words = text.replace(/#[^#]+#/g, '').replace(/@[\w\u4e00-\u9fff-]+/g, '').replace(/[，。！？、：；""''【】（）\s]+/g, ' ').split(' ');
    words.forEach(w => {
      w = w.trim();
      if (w.length >= 2 && w.length <= 10) wordFreq[w] = (wordFreq[w] || 0) + 1;
    });
    (a.topics || []).forEach(tp => { const t = tp.replace(/#/g,''); if(t) wordFreq[t] = (wordFreq[t] || 0) + 3; });
  });

  const sorted = Object.entries(wordFreq).sort((a,b) => b[1]-a[1]).slice(0, 150);

  const chart = initChart('chart-wordcloud');
  chart.setOption({
    series: [{
      type: 'wordCloud',
      sizeRange: [12, 60],
      rotationRange: [-30, 30],
      rotationStep: 15,
      gridSize: 8,
      shape: 'circle',
      width: '90%',
      height: '90%',
      textStyle: {
        fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif',
        color: () => {
          const colors = ['#5b8def','#ff6b6b','#4ecdc4','#ffd166','#a78bfa','#ff8c42','#06d6a0','#e879a8'];
          return colors[Math.floor(Math.random() * colors.length)];
        },
      },
      data: sorted.map(([name, value]) => ({ name, value })),
    }],
  });
}

function renderTopicFreq() {
  const t = getThemeColors();
  const topicCounts = {};
  microData.forEach(a => {
    (a.topics || []).forEach(tp => { topicCounts[tp] = (topicCounts[tp] || 0) + 1; });
  });

  const sorted = Object.entries(topicCounts).sort((a,b) => b[1]-a[1]).slice(0, 15);
  const chart = initChart('chart-topicFreq');
  chart.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    grid: { left: 160, right: 30, top: 10, bottom: 30 },
    xAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'category', data: sorted.map(s=>s[0]).reverse(), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 11, width: 140, overflow: 'truncate' } },
    series: [{ type: 'bar', data: sorted.map(s=>s[1]).reverse(), itemStyle: { color: new echarts.graphic.LinearGradient(0,0,1,0,[{offset:0,color:'#a78bfa'},{offset:1,color:'#5b8def'}]), borderRadius: [0,4,4,0] } }],
  });
}

function renderHourlyDist() {
  const t = getThemeColors();
  const hourCounts = Array(24).fill(0);
  microData.forEach(a => {
    const h = new Date(a.time).getHours();
    if (!isNaN(h)) hourCounts[h]++;
  });

  const chart = initChart('chart-hourlyDist');
  chart.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    xAxis: { type: 'category', data: Array.from({length:24}, (_,i) => `${String(i).padStart(2,'0')}:00`), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    series: [{
      type: 'bar', data: hourCounts,
      itemStyle: {
        color: p => {
          const colors = ['#2a3050','#2a3050','#2a3050','#2a3050','#2a3050','#2a3050',
            '#4ecdc4','#4ecdc4','#5b8def','#5b8def','#5b8def','#5b8def',
            '#ffd166','#ffd166','#5b8def','#5b8def','#5b8def','#5b8def',
            '#ff8c42','#ff8c42','#ff6b6b','#ff6b6b','#a78bfa','#a78bfa'];
          return colors[p.dataIndex] || '#5b8def';
        },
        borderRadius: [4,4,0,0],
      },
    }],
  });
}

function renderDailyDist() {
  const t = getThemeColors();
  const dailyCounts = {};
  microData.forEach(a => {
    const day = a.time?.substring(0, 10) || 'unknown';
    dailyCounts[day] = (dailyCounts[day] || 0) + 1;
  });

  const days = Object.keys(dailyCounts).sort();
  const chart = initChart('chart-dailyDist');
  chart.setOption({
    ...baseOption(),
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    xAxis: { type: 'category', data: days, axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    series: [{ type: 'bar', data: days.map(d => dailyCounts[d]), itemStyle: { color: '#5b8def', borderRadius: [4,4,0,0] }, barMaxWidth: 50 }],
  });
}

function renderEmotionStanceScatter() {
  const t = getThemeColors();
  const scatterData = {};
  microData.forEach(a => {
    const em = a.emotion || 'neutral';
    if (!scatterData[em]) scatterData[em] = [];
    if (scatterData[em].length < 500) {
      scatterData[em].push([a.emotion_intensity || 0, a.stance_intensity || 0, a.username]);
    }
  });

  const chart = initChart('chart-emotionStanceScatter');
  chart.setOption({
    ...baseOption(),
    tooltip: {
      trigger: 'item',
      backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text },
      formatter: p => `${p.seriesName}<br/>情绪强度: ${p.value[0]}<br/>立场强度: ${p.value[1]}<br/>用户: ${p.value[2]}`
    },
    legend: { top: 0, textStyle: { color: t.subtext, fontSize: 11 }, type: 'scroll' },
    grid: { left: 60, right: 30, top: 40, bottom: 40 },
    xAxis: { type: 'value', name: '情绪强度', min: 0, max: 1, axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    yAxis: { type: 'value', name: '立场强度', min: 0, max: 1, axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    series: Object.entries(scatterData).map(([em, data]) => ({
      name: EMOTION_LABELS[em] || em,
      type: 'scatter',
      data,
      symbolSize: 6,
      itemStyle: { color: EMOTION_COLORS[em] || '#5b8def', opacity: 0.6 },
    })),
  });
}

function renderActionAgentHeatmap() {
  const t = getThemeColors();
  const actionTypes = Object.keys(ACTION_LABELS);
  const agentTypes = ['citizen', 'kol', 'media', 'government'];
  
  const heatData = [];
  const matrix = {};
  microData.forEach(a => {
    const key = `${a.action_type}|${a.agent_type}`;
    matrix[key] = (matrix[key] || 0) + 1;
  });

  actionTypes.forEach((at, xi) => {
    agentTypes.forEach((ag, yi) => {
      heatData.push([xi, yi, matrix[`${at}|${ag}`] || 0]);
    });
  });

  const maxVal = Math.max(...heatData.map(d => d[2]), 1);

  const chart = initChart('chart-actionAgentHeatmap');
  chart.setOption({
    ...baseOption(),
    tooltip: { position: 'top', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text },
      formatter: p => `${ACTION_LABELS[actionTypes[p.value[0]]]}<br/>${AGENT_LABELS[agentTypes[p.value[1]]]}<br/>数量: ${p.value[2]}` },
    grid: { left: 80, right: 30, top: 10, bottom: 60 },
    xAxis: { type: 'category', data: actionTypes.map(a => ACTION_LABELS[a]), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 11, rotate: 30 }, splitArea: { show: true, areaStyle: { color: ['transparent', 'rgba(255,255,255,0.02)'] } } },
    yAxis: { type: 'category', data: agentTypes.map(a => AGENT_LABELS[a]), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext } },
    visualMap: { min: 0, max: maxVal, calculable: true, orient: 'horizontal', left: 'center', bottom: 0, textStyle: { color: t.subtext },
      inRange: { color: ['#1e2235', '#5b8def', '#4ecdc4', '#ffd166', '#ff6b6b'] } },
    series: [{ type: 'heatmap', data: heatData, label: { show: true, color: t.text, fontSize: 12 }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }],
  });
}

function renderSankey() {
  const t = getThemeColors();
  
  const agentActionFlow = {};
  microData.forEach(a => {
    const src = AGENT_LABELS[a.agent_type] || a.agent_type;
    const tgt = ACTION_LABELS[a.action_type] || a.action_type;
    const key = `${src}|${tgt}`;
    agentActionFlow[key] = (agentActionFlow[key] || 0) + 1;
  });

  const actionEmotionFlow = {};
  microData.forEach(a => {
    const src = ACTION_LABELS[a.action_type] || a.action_type;
    const tgt = (EMOTION_LABELS[a.emotion] || a.emotion || '未知') + '情绪';
    const key = `${src}|${tgt}`;
    actionEmotionFlow[key] = (actionEmotionFlow[key] || 0) + 1;
  });

  const nodeNames = new Set();
  const links = [];

  Object.entries(agentActionFlow).forEach(([key, value]) => {
    const [src, tgt] = key.split('|');
    nodeNames.add(src);
    nodeNames.add(tgt);
    links.push({ source: src, target: tgt, value });
  });

  Object.entries(actionEmotionFlow).forEach(([key, value]) => {
    const [src, tgt] = key.split('|');
    nodeNames.add(src);
    nodeNames.add(tgt);
    if (value > 50) links.push({ source: src, target: tgt, value });
  });

  const chart = initChart('chart-sankey');
  chart.setOption({
    ...baseOption(),
    tooltip: { trigger: 'item', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    series: [{
      type: 'sankey',
      layout: 'none',
      emphasis: { focus: 'adjacency' },
      nodeAlign: 'left',
      data: [...nodeNames].map(name => ({ name })),
      links,
      lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.3 },
      label: { color: t.text, fontSize: 11 },
      itemStyle: { borderWidth: 1, borderColor: t.axis },
    }],
  });
}

function renderEmotionRadar() {
  const t = getThemeColors();
  const agentEmotionAvg = {};

  microData.forEach(a => {
    const agent = a.agent_type || 'unknown';
    const emotion = a.emotion || 'neutral';
    if (!agentEmotionAvg[agent]) agentEmotionAvg[agent] = {};
    if (!agentEmotionAvg[agent][emotion]) agentEmotionAvg[agent][emotion] = { sum: 0, count: 0 };
    agentEmotionAvg[agent][emotion].sum += (a.emotion_intensity || 0);
    agentEmotionAvg[agent][emotion].count++;
  });

  const emotions = [...new Set(microData.map(a => a.emotion || 'neutral'))].slice(0, 8);
  const agents = Object.keys(agentEmotionAvg);

  const chart = initChart('chart-emotionRadar');
  chart.setOption({
    ...baseOption(),
    legend: { top: 5, textStyle: { color: t.subtext } },
    radar: {
      indicator: emotions.map(em => ({ name: EMOTION_LABELS[em] || em, max: 1 })),
      shape: 'polygon',
      axisName: { color: t.subtext },
      splitLine: { lineStyle: { color: t.axis, opacity: 0.5 } },
      splitArea: { areaStyle: { color: ['transparent', 'rgba(91,141,239,0.03)'] } },
      axisLine: { lineStyle: { color: t.axis } },
    },
    series: [{
      type: 'radar',
      data: agents.map((agent, i) => ({
        name: AGENT_LABELS[agent] || agent,
        value: emotions.map(em => {
          const d = agentEmotionAvg[agent]?.[em];
          return d ? +(d.sum / d.count).toFixed(3) : 0;
        }),
        areaStyle: { opacity: 0.15 },
        lineStyle: { width: 2 },
      })),
    }],
  });
}

function renderCumulativeActions() {
  const t = getThemeColors();
  const timeSlots = {};
  microData.forEach(a => {
    const time = a.time?.substring(0, 16) || 'unknown';
    const actionType = a.action_type;
    if (!timeSlots[time]) timeSlots[time] = {};
    timeSlots[time][actionType] = (timeSlots[time][actionType] || 0) + 1;
  });

  const times = Object.keys(timeSlots).sort();
  const actionTypes = Object.keys(ACTION_LABELS);

  const cumulative = {};
  actionTypes.forEach(at => { cumulative[at] = []; let sum = 0; times.forEach(time => { sum += timeSlots[time]?.[at] || 0; cumulative[at].push(sum); }); });

  const chart = initChart('chart-cumulativeActions');
  chart.setOption({
    ...baseOption(),
    legend: { top: 0, textStyle: { color: t.subtext, fontSize: 11 } },
    tooltip: { trigger: 'axis', backgroundColor: t.tooltip, borderColor: t.axis, textStyle: { color: t.text } },
    grid: { left: 70, right: 30, top: 40, bottom: 60 },
    xAxis: { type: 'category', data: times.map(t => t.replace('T',' ')), axisLine: { lineStyle: { color: t.axis } }, axisLabel: { color: t.subtext, fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: t.axis } }, splitLine: { lineStyle: { color: t.axis, opacity: 0.3 } }, axisLabel: { color: t.subtext } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 5 }],
    series: actionTypes.map(at => ({
      name: ACTION_LABELS[at],
      type: 'line',
      smooth: true,
      symbol: 'none',
      stack: 'cumulative',
      areaStyle: { opacity: 0.3 },
      lineStyle: { width: 1.5 },
      data: cumulative[at],
    })),
  });
}

// ========== Window resize ==========
window.addEventListener('resize', () => {
  Object.values(charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
});
