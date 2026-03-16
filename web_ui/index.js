// ==============================================================================
// 📦 БЛОК 1: ГЛОБАЛЬНИЙ СТАН (STATE)
// Цей блок відповідає за зберігання змінних, налаштувань UI та даних у пам'яті.
// ==============================================================================
let levelData = { 'l3': null, 'l2': null, 'l1': null };
let hierarchy = {};
let kmeansData = null; // Тут зберігаються кластери АБО похибки

// Змінні для спец. територій (Крим, Чорнобиль)
let crimeaGeojson = null;
let stripeData = [];
let chornobylGeojson = null;
let chornobylStripes = [];

const clusterColors = [
    [0, 255, 136, 230], [0, 191, 255, 230], [255, 23, 68, 230], [255, 235, 59, 230], [255, 64, 129, 230],
    [178, 255, 89, 230], [213, 0, 249, 230], [255, 145, 0, 230], [0, 229, 255, 230], [200, 200, 200, 230]
];

// Відновлення стану з попередньої сесії
let savedUI = JSON.parse(sessionStorage.getItem('uiState') || '{}');
let cy = savedUI.cy || '2024';
let cl = savedUI.cl || 'l3';
let cMain = 'BAL'; 
let cDetail = savedUI.cDetail || '4';
let cSub = savedUI.cSub || '';
let viewMode = savedUI.viewMode || 'data';
let cIndex = savedUI.cIndex || 'income';

// Синхронізація базових контролів з відновленим станом
document.getElementById('year-slider').value = cy;
document.getElementById('year-display').innerText = cy;
document.getElementById('main-metric').value = 'BAL';
document.getElementById('level-switcher').value = cl;
document.getElementById('view-mode').value = viewMode;

let selectedFeature = null;
let chartInstances = [];
let deckgl = null;
let compareBuffer = JSON.parse(localStorage.getItem('compareBuffer')) || [];


// ==============================================================================
// 📡 БЛОК 2: РОБОТА З API ТА ДАНИМИ (FETCH)
// Тут зібрані всі функції, які роблять запити до бекенду.
// ==============================================================================

// 2.1 Завантаження конфігурації фільтрів з бекенду + Створення селекта компонент
async function loadFiltersConfig() {
    try {
        const response = await fetch('/api/filters_config');
        if (!response.ok) throw new Error('Failed to load filters config');
        const config = await response.json();

        const populateSelect = (selectId, optionsData) => {
            const select = document.getElementById(selectId);
            if (!select) return;
            select.innerHTML = ''; 
            optionsData.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                if (opt.selected) option.selected = true;
                select.appendChild(option);
            });
        };

        populateSelect('view-mode', config.view_modes);
        populateSelect('level-switcher', config.levels);
        populateSelect('main-metric', config.categories);
        populateSelect('detail-lvl', config.detail_levels);

        const indexSelect = document.getElementById('index-metric');
        if (indexSelect) {
            indexSelect.innerHTML = '';
            
            const groupClusters = document.createElement('optgroup');
            groupClusters.label = "PCA K-Means Clusters";
            config.analytical_indexes.forEach(opt => {
                const option = document.createElement('option'); option.value = opt.value; option.textContent = opt.label;
                groupClusters.appendChild(option);
            });
            indexSelect.appendChild(groupClusters);

            const groupErrors = document.createElement('optgroup');
            groupErrors.label = "Model Accuracy (Errors)";
            config.pca_errors.forEach(opt => {
                const option = document.createElement('option'); option.value = opt.value; option.textContent = opt.label;
                groupErrors.appendChild(option);
            });
            indexSelect.appendChild(groupErrors);
            
            // Відновлюємо вибране значення індексу після заповнення
            indexSelect.value = cIndex; 
        }

        // Динамічне створення фільтра 3/10 компонент
        const indexControls = document.getElementById('index-controls');
        if (indexControls && !document.getElementById('comp-switcher')) {
            const compLabel = document.createElement('label');
            compLabel.textContent = "PCA Dimensions";
            compLabel.style.marginTop = "15px";
            
            const compSelect = document.createElement('select');
            compSelect.id = 'comp-switcher';
            config.components.forEach(opt => {
                const option = document.createElement('option'); option.value = opt.value; option.textContent = opt.label;
                compSelect.appendChild(option);
            });
            
            indexControls.appendChild(compLabel);
            indexControls.appendChild(compSelect);
        }

    } catch (error) { console.error("❌ Error loading filters:", error); }
}

// 2.2 УНІВЕРСАЛЬНА функція завантаження індексів (РОЗУМНИЙ РОУТИНГ)
async function fetchIndexData() {
    try {
        const indexSelect = document.getElementById('index-metric');
        const compSelect = document.getElementById('comp-switcher');
        
        const selectedIndex = indexSelect ? indexSelect.value : 'income'; // 'econ' або 'econ_error'
        const nComp = compSelect ? compSelect.value : 3; // '3' або '10'

        let endpointUrl = '';

        // Роутинг: Куди йдемо? За кластерами чи за похибками?
        if (selectedIndex.endsWith('_error')) {
            const category = selectedIndex.replace('_error', '');
            endpointUrl = `/api/pca_errors/${category}?n_comp=${nComp}`;
        } else {
            endpointUrl = `/api/kmeans_data/${selectedIndex}?n_comp=${nComp}`;
        }

        console.log(`📡 Завантажую аналітику: ${endpointUrl}`);
        let res = await fetch(endpointUrl);
        if (res.ok) {
            kmeansData = await res.json();
        } else {
            kmeansData = null;
        }
    } catch(e) {
        console.error("Помилка завантаження індексу:", e);
        kmeansData = null;
    }
}

// 2.3 Головний ініціалізатор (завантажує все базове при старті)
async function loadDataFromServer() {
    try {
        document.getElementById('loading-details').innerText = 'Підключення...';

        const [resConfig, resL3, resL2, resL1, resCrimea, resChern] = await Promise.all([
            fetchWithBytes('/api/config', 'config'),
            fetchWithBytes('/api/geo/l3', 'l3'),
            fetchWithBytes('/api/geo/l2', 'l2'),
            fetchWithBytes('/api/geo/l1', 'l1'),
            fetchWithBytes('/api/geo/crimea', 'crimea'),
            fetchWithBytes('/api/geo/chornobyl', 'chern') 
        ]);
        
        document.getElementById('loading-details').innerText = 'Декодування геометрії...';

        hierarchy = resConfig;
        levelData['l3'] = resL3; levelData['l2'] = resL2; levelData['l1'] = resL1;
        
        if (resCrimea) { crimeaGeojson = resCrimea; stripeData = generateStripes(crimeaGeojson, 0.04); }
        if (resChern) { chornobylGeojson = resChern; chornobylStripes = generateStripes(chornobylGeojson, 0.04); }
        
        document.getElementById('loading-details').innerText = 'Підготовка аналітики...';
        await fetchIndexData(); // Завантажуємо стартовий індекс
        
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-text').innerText = '100%';

        setTimeout(() => {
            document.getElementById('loading-screen').style.display = 'none';
            document.getElementById('control-panel').style.display = 'block';
            document.getElementById('compare-widget').style.display = 'block';
            document.getElementById('btn-old-dash').style.display = 'block';

            initMap(); 
            updateUI(true); 
            updateCompareWidget(); 
            
            if (cl === 'l3') loadSecondaryData('l3'); 
        }, 400);

    } catch (e) {
        console.error(e);
        document.getElementById('loading-screen').innerHTML = `<h2 style="color:red;">Помилка підключення</h2><p>${e.message}</p>`;
    }
}

// 2.4 Фонове підвантаження важких даних (чанки)
async function loadSecondaryData(level) {
    const categories = [
        { id: 'INC', name: 'Власні Доходи', endpoint: 'income' },
        { id: 'PROG', name: 'Видатки Програмні (КПКВК)', endpoint: 'prog' },
        { id: 'ECON', name: 'Видатки Економічні (КЕКВ)', endpoint: 'econ' },
        { id: 'FUNC', name: 'Видатки Функціональні (ФКВК)', endpoint: 'func' }
    ];

    const mainMetricSelect = document.getElementById('main-metric');
    Array.from(mainMetricSelect.options).forEach(opt => {
        if (opt.value !== 'BAL') {
            opt.disabled = true;
            if (!opt.dataset.originalText) opt.dataset.originalText = opt.text;
            opt.text = `⏳ ${opt.dataset.originalText} (завантаження...)`;
        }
    });

    for (const cat of categories) {
        try {
            let res = await fetch(`/api/chunk_data/${level}/${cat.endpoint}`);
            if (!res.ok) continue; 
            let chunkData = await res.json();
            
            if (levelData[level] && levelData[level].features) {
                levelData[level].features.forEach(feature => {
                    let code = feature.properties.BUDGET_CODE;
                    if (chunkData[code]) Object.assign(feature.properties, chunkData[code]);
                });
            }

            let option = Array.from(mainMetricSelect.options).find(o => o.value === cat.id);
            if (option) { option.disabled = false; option.text = option.dataset.originalText; }
        } catch (e) {
            console.error(`Помилка завантаження ${cat.name}:`, e);
            let option = Array.from(mainMetricSelect.options).find(o => o.value === cat.id);
            if (option) option.text = `❌ ${option.dataset.originalText} (помилка)`;
        }
    } 
}


// ==============================================================================
// 🎨 БЛОК 3: УТИЛІТИ ДЛЯ ГЕОМЕТРІЇ ТА КОЛЬОРІВ
// ==============================================================================
function generateStripes(geojson, spacing = 0.04) {
  if (!geojson) return [];
  const lines = []; const polyCoords = geojson.features[0].geometry.coordinates[0];
  const minX = 32.0, maxX = 37.0, minY = 44.0, maxY = 46.5;
  
  function isInside(pt) {
    let x = pt[0], y = pt[1], inside = false;
    for (let i = 0, j = polyCoords.length - 1; i < polyCoords.length; j = i++) {
      let xi = polyCoords[i][0], yi = polyCoords[i][1]; let xj = polyCoords[j][0], yj = polyCoords[j][1];
      let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  for (let c = minY + minX; c <= maxY + maxX; c += spacing) {
      let currentSeg = null;
      for (let x = minX; x <= maxX; x += 0.01) {
          let y = -x + c; if (y < minY || y > maxY) continue;
          if (isInside([x, y])) { if (!currentSeg) currentSeg = [[x, y]]; else currentSeg[1] = [x, y]; } 
          else { if (currentSeg && currentSeg.length === 2) lines.push({path: [currentSeg[0], currentSeg[1]]}); currentSeg = null; }
      }
      if (currentSeg && currentSeg.length === 2) lines.push({path: [currentSeg[0], currentSeg[1]]});
  }
  return lines;
}

function interpolateColor(c1, c2, factor) { return [ Math.round(c1[0] + (c2[0] - c1[0]) * factor), Math.round(c1[1] + (c2[1] - c1[1]) * factor), Math.round(c1[2] + (c2[2] - c1[2]) * factor), 230 ]; }
function getHeatColor(val, maxVal) {
    if (val === 0 || maxVal === 0) return [50, 50, 50, 150];
    let t = Math.abs(val) / maxVal; t = Math.pow(Math.max(0, Math.min(1, t)), 0.6); 
    if (t < 0.25) return interpolateColor([30, 10, 60], [120, 20, 100], t / 0.25);
    if (t < 0.5) return interpolateColor([120, 20, 100], [220, 60, 50], (t - 0.25) / 0.25);
    if (t < 0.75) return interpolateColor([220, 60, 50], [255, 140, 0], (t - 0.5) / 0.25);
    return interpolateColor([255, 140, 0], [255, 255, 100], (t - 0.75) / 0.25);
}


// ==============================================================================
// 🗺️ БЛОК 4: ВІЗУАЛІЗАЦІЯ КАРТИ (DECK.GL)
// Відповідає за рендер полігонів, 3D стовпців та тултипів на карті.
// ==============================================================================
function initMap() {
    let defaultView = { latitude: 48.4, longitude: 31.2, zoom: 5.3, pitch: 50, bearing: 0 };
    let savedView = JSON.parse(sessionStorage.getItem('deckViewState'));

    deckgl = new deck.DeckGL({
        container: 'container',
        mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',        initialViewState: savedView || defaultView,
        controller: true,
        getTooltip: ({object, layer}) => {
            if (!object) return null;

            // --- 1. ПЕРЕВІРКА НА СПЕЦТЕРИТОРІЇ ---
            if (layer && layer.id.startsWith('crimea')) {
                return {
                    html: `<div style="padding: 12px; min-width: 200px;">
                               <div style="font-weight:bold; font-size:16px; margin-bottom:5px;">АР Крим та м. Севастополь</div>
                               <div style="color:#ffcc00; font-size:13px;">Тимчасово окупована територія.<br>Бюджетні дані відсутні.</div>
                           </div>`,
                    style: { backgroundColor: 'rgba(15,15,20,0.95)', color: '#fff', border: '1px solid #ffcc00', borderRadius: '8px' }
                };
            }
            if (layer && layer.id.startsWith('chornobyl')) {
                return {
                    html: `<div style="padding: 12px; min-width: 200px;">
                               <div style="font-weight:bold; font-size:16px; margin-bottom:5px;">Зона відчуження ЧАЕС</div>
                               <div style="color:#00ff88; font-size:13px;">Спеціальний статус.<br>Бюджетні дані відсутні.</div>
                           </div>`,
                    style: { backgroundColor: 'rgba(15,15,20,0.95)', color: '#fff', border: '1px solid #00ff88', borderRadius: '8px' }
                };
            }

            // --- 2. СТАНДАРТНІ ГРОМАДИ ---
            const p = object.properties;
            if (!p || !p.BUDGET_CODE) return null; // Запобіжник від порожніх полігонів
            const code = p.BUDGET_CODE;
            const displayName = p.display_name || "Невідома громада"; // Запобіжник від undefined

            // РЕЖИМ АНАЛІТИЧНИХ ІНДЕКСІВ
            if (viewMode === 'index' && kmeansData) {
                const isErrorMode = cIndex.endsWith('_error');
                
                if (isErrorMode) {
                    const item = kmeansData[cl]?.[code];
                    if (!item || !item.dates) return { html: `<div style="padding:12px;">${displayName}<br>Немає даних за цей рік</div>` };
                    
                    const yearly = item.error.filter((v, i) => String(item.dates[i]).includes(String(cy)));
                    if (yearly.length === 0) return { html: `<div style="padding:12px;">${displayName}<br>Немає даних за цей рік</div>` };
                    
                    const avg = yearly.reduce((a, b) => a + b, 0) / yearly.length;
                    const errValue = (avg * 100).toFixed(2);
                    
                    return {
                        html: `
                        <div style="padding: 12px; min-width: 220px;">
                            <div style="font-weight:bold; font-size:18px; margin-bottom:8px;">${displayName}</div>
                            <div style="color:#ff1744; font-size:16px; font-weight:bold;">
                                ⚠️ Аномальність: ${errValue}%
                            </div>
                        </div>`,
                        style: { backgroundColor: 'rgba(15,15,20,0.95)', color: '#fff', border: '1px solid #ff1744', borderRadius: '8px' }
                    };
                } else {
                    const item = kmeansData[cl]?.[cy]?.[code];
                    if (!item) return { html: `<div style="padding: 12px;"><b>${displayName}</b><br>Немає даних</div>`, style: {backgroundColor: 'rgba(15,15,20,0.95)', color:'#fff'} };
                    return {
                        html: `<div style="padding: 12px;"><div style="font-weight:bold; font-size:18px;">${displayName}</div><div style="color:#00bfff; margin-top:5px;"><b>Кластер:</b> ${item.cluster}</div></div>`,
                        style: { backgroundColor: 'rgba(15,15,20,0.95)', color: '#fff', border: '1px solid #333', borderRadius: '8px' }
                    };
                }
            }

            // РЕЖИМ ЗВИЧАЙНИХ ДАНИХ (Баланс/Доходи/Видатки)
            let key = cMain === 'BAL' ? 'BAL_' + cy : cMain + '_' + cSub + '_' + cy;
            let val = parseFloat(p[key]) || 0;
            let metricName = cMain === 'BAL' ? 'Загальний Баланс' : (document.getElementById('sub-metric').options[document.getElementById('sub-metric').selectedIndex]?.text || '');
            
            return {
                html: `
                <div style="padding: 12px; min-width: 250px;">
                    <div style="font-weight:bold; font-size:18px; margin-bottom:5px;">${displayName}</div>
                    <div style="color:#00ff88; font-size:15px; font-weight:bold;">${metricName}:<br><span style="font-size:22px;">${(val/1e6).toFixed(2)} млн ₴</span></div>
                    <hr style="border-color:#444; margin:10px 0;">
                    <div style="font-size:12px; color:#aaa;">Рік: ${cy}</div>
                </div>`,
                style: { backgroundColor: 'rgba(15,15,20,0.95)', color: '#fff', border: '1px solid #333', borderRadius: '8px' }
            };
        }
    });

    updateMap();
}

function updateMap() {
    if (!deckgl || !levelData[cl]) return;

    const isIndex = viewMode === 'index';
    const isErrorMode = isIndex && cIndex.endsWith('_error');
    const key = cMain === 'BAL' ? 'BAL_' + cy : cMain + '_' + cSub + '_' + cy;

    // --- 1. ШУКАЄМО МАКСИМУМ (Потрібен для розрахунку інтенсивності кольору) ---
    let maxVal = 0.0001; 
    
    if (isIndex && kmeansData && kmeansData[cl]) {
        if (isErrorMode) {
            Object.values(kmeansData[cl]).forEach(d => {
                if (d.error && d.dates) {
                    const yearly = d.error.filter((v, i) => d.dates[i] && String(d.dates[i]).includes(String(cy)));
                    if (yearly.length > 0) {
                        const avg = yearly.reduce((a, b) => a + b, 0) / yearly.length;
                        if (avg > maxVal) maxVal = avg;
                    }
                }
            });
        } else if (kmeansData[cl][cy]) {
            Object.values(kmeansData[cl][cy]).forEach(d => {
                if (d.distance > maxVal) maxVal = d.distance;
            });
        }
    } else if (!isIndex) {
        levelData[cl].features.forEach(f => {
            let val = Math.abs(parseFloat(f.properties[key]) || 0);
            if (val > maxVal) maxVal = val;
        });
    }

    // --- 2. ВИСОТА = 0 (ПЛАСКА КАРТА ДЛЯ ВСЬОГО) ---
    const getElev = f => 0;

    // --- 3. ЛОГІКА ІНТЕНСИВНОСТІ КОЛЬОРУ ---
    const getFillCol = f => {
        const code = f.properties.BUDGET_CODE;
        
        // А. РЕЖИМ ПОХИБОК (Errors)
        if (isIndex && isErrorMode && kmeansData?.[cl]) {
            const item = kmeansData[cl][code];
            if (!item || !item.dates) return [200, 200, 200, 40]; // Немає даних - ледь помітний сірий
            
            const yearly = item.error.filter((v, i) => item.dates[i] && String(item.dates[i]).includes(String(cy)));
            if (yearly.length === 0) return [200, 200, 200, 40];
            
            const avg = yearly.reduce((a, b) => a + b, 0) / yearly.length;
            
            // Розрахунок інтенсивності (0.0 - 1.0)
            let factor = Math.min(avg / maxVal, 1.0);
            
            // Чим більша похибка, тим червоніший колір і менш прозорий
            const r = 255;
            const g = Math.round(255 * (1 - factor));
            const b = Math.round(255 * (1 - factor));
            const alpha = Math.round(40 + 215 * factor); // Від 40 (прозорий) до 255 (суцільний)
            
            return [r, g, b, alpha];
        }
        
        // Б. РЕЖИМ КЛАСТЕРІВ
        if (isIndex && !isErrorMode && kmeansData?.[cl]) {
            const item = kmeansData[cl][cy] ? kmeansData[cl][cy][code] : null;
            if (item) {
                let col = clusterColors[(item.cluster - 1) % 10];
                return [col[0], col[1], col[2], 200]; // Кластери малюємо рівномірно
            }
            return [200, 200, 200, 40];
        }
        
        // В. РЕЖИМ БАЛАНСУ ТА ДОХОДІВ
        let val = parseFloat(f.properties[key]) || 0;
        let factor = Math.min(Math.abs(val) / maxVal, 1.0);
        const alpha = Math.round(40 + 215 * factor); // Інтенсивність залежить від суми грошей
        
        if (cMain === 'BAL') {
            if (val >= 0) return [0, 180, 80, alpha]; // Від блідо-зеленого до насиченого
            return [255, 50, 50, alpha];              // Від блідо-червоного до насиченого
        }
        
        // Для інших метрик (Доходи/Видатки) використовуємо твій Heatmap
        return getHeatColor(val, maxVal); 
    };

    // --- 4. РЕНДЕР (ОПТИМІЗОВАНО ПІД 2D) ---
    const layers = [
        // Головний шар-заливка (без 3D)
        new deck.GeoJsonLayer({
            id: 'poly-fill', 
            data: levelData[cl], 
            extruded: false, // ❌ ВИМКНЕНО 3D
            stroked: false,  // ❌ Вимикаємо контури тут, щоб не було "бруду" на кордонах
            filled: true,
            pickable: true, 
            autoHighlight: true, 
            highlightColor: [0, 0, 0, 40], // Темне затінення при наведенні мишкою
            getFillColor: getFillCol,
            updateTriggers: { 
                getFillColor: [cy, cl, key, viewMode, cIndex, kmeansData, maxVal] 
            },
            transitions: { getFillColor: 600 },
            onClick: (info) => { if (info.object) { selectedFeature = info.object; openPanel(); } }
        }),
        
        // Шар тонких кордонів (накладаємо зверху)
        new deck.GeoJsonLayer({ 
            id: 'borders-base', 
            data: levelData[cl], 
            filled: false, 
            stroked: true, 
            getLineColor: [50, 50, 50, 80], // Темно-сірі тонкі контури
            lineWidthMinPixels: 1, 
            getElevation: 0 
        })
    ];

    // Спеціальні території (Крим та Чорнобиль) — теж плоскі
if (crimeaGeojson) {
        layers.push(
            // Напівпрозорий фон Криму
            new deck.GeoJsonLayer({
                id: 'crimea-fill-solid', data: crimeaGeojson, pickable: true, stroked: true, filled: true, extruded: false,
                getFillColor: [255, 204, 0, 30], 
                getLineColor: [255, 204, 0, 150], lineWidthMinPixels: 1
            }),
            // Жовті смужки
            new deck.PathLayer({
                id: 'crimea-stripes', data: stripeData, getPath: d => d.path, 
                getColor: [255, 204, 0, 180], getWidth: 3, widthMinPixels: 2, pickable: false, extruded: false
            }),
            // Текстовий підпис (опціонально, але гарно виглядає на скріншотах)
            new deck.TextLayer({
                id: 'crimea-label-text', data: [{position: [34.3, 45.1], text: "Тимчасово окупована\nтериторія АР Крим"}],
                getPosition: d => d.position, getText: d => d.text, getSize: 14, getColor: [150, 100, 0, 255], 
                outlineWidth: 3, outlineColor: [255, 255, 255, 255], fontFamily: 'Arial, sans-serif', fontWeight: 'bold'
            })
        );
    }

    if (chornobylGeojson) {
        layers.push(
            // Напівпрозорий фон Чорнобиля
            new deck.GeoJsonLayer({
                id: 'chornobyl-fill-solid', data: chornobylGeojson, pickable: true, stroked: true, filled: true, extruded: false,
                getFillColor: [0, 255, 136, 20], 
                getLineColor: [0, 255, 136, 120], lineWidthMinPixels: 1
            }),
            // Зелені смужки
            new deck.PathLayer({
                id: 'chornobyl-stripes', data: chornobylStripes, getPath: d => d.path,
                getColor: [0, 255, 136, 150], getWidth: 2, widthMinPixels: 1, pickable: false, extruded: false
            })
        );
    }

    deckgl.setProps({layers});
}


// ==============================================================================
// 🖥️ БЛОК 5: ЛОГІКА ІНТЕРФЕЙСУ ТА САЙД-ПАНЕЛІ
// ==============================================================================
function saveUIState() { sessionStorage.setItem('uiState', JSON.stringify({cy, cl, cMain, cDetail, cSub, viewMode, cIndex})); }
function openAuthorModal() { document.getElementById('author-modal').style.display = 'flex'; }
function closeAuthorModal() { document.getElementById('author-modal').style.display = 'none'; }
function openPanel() { document.getElementById('side-panel').classList.add('open'); document.getElementById('sp-title').innerText = selectedFeature.properties.display_name; refreshCompareButtonState(); drawSideCharts(); }
function closePanel() { document.getElementById('side-panel').classList.remove('open'); selectedFeature = null; }
function getCurrentSelectionID() { if (!selectedFeature) return null; return `${selectedFeature.properties.BUDGET_CODE}_${cMain}_${cDetail}`; }

function createSingleChart(container, title, dataArray, borderColor, backgroundColor, isFill) {
    const wrapper = document.createElement('div'); wrapper.className = 'chart-wrapper';
    const canvas = document.createElement('canvas'); wrapper.appendChild(canvas); container.appendChild(wrapper);
    const chart = new Chart(canvas.getContext('2d'), { type: 'line', data: { labels: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], datasets: [{ label: title, data: dataArray, borderColor: borderColor, backgroundColor: backgroundColor, fill: isFill, borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#222', tension: 0.3 }] }, options: { responsive: true, maintainAspectRatio: false, color: '#ccc', interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'top', labels: { color: '#fff', font: {size: 11, weight: 'bold'} } } }, scales: { y: { grid: { color: '#333' }, ticks: { color: '#aaa' } }, x: { grid: { color: '#333' }, ticks: { color: '#aaa' } } } } });
    chartInstances.push(chart);
}

function drawSideCharts() {
    if (!selectedFeature) return;
    chartInstances.forEach(c => c.destroy()); chartInstances = [];
    const container = document.getElementById('charts-container'); container.innerHTML = '';
    const p = selectedFeature.properties; const years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

    if (viewMode === 'index' && !cIndex.endsWith('_error')) {
        document.getElementById('sp-subtitle').innerText = "Динаміка аномальності (Відстань до центроїда)";
        let distData = years.map(y => kmeansData?.[cl]?.[y]?.[p.BUDGET_CODE]?.distance || 0);
        createSingleChart(container, 'Відстань до центру кластера', distData, '#ff1744', 'rgba(255,23,68,0.1)', true);
        return;
    }

    if (cMain === 'BAL') {
        document.getElementById('sp-subtitle').innerText = "Основні фінансові показники";
        createSingleChart(container, 'Баланс (Млн ₴)', years.map(y => (parseFloat(p['BAL_' + y]) || 0) / 1e6), '#00ff88', 'rgba(0,255,136,0.1)', true);
        createSingleChart(container, 'Власні Доходи (Млн ₴)', years.map(y => (parseFloat(p['TOT_INC_' + y]) || 0) / 1e6), '#00bfff', 'transparent', false);
        createSingleChart(container, 'Загальні Видатки (Млн ₴)', years.map(y => (parseFloat(p['TOT_EXP_' + y]) || 0) / 1e6), '#ff1744', 'transparent', false);
    } else {
        document.getElementById('sp-subtitle').innerText = `Деталізація: ${cDetail} знаки`;
        let codesObj = {}; years.forEach(y => { if (hierarchy[y]?.[cl]?.[cMain]?.[cDetail]) Object.assign(codesObj, hierarchy[y][cl][cMain][cDetail]); });
        const codes = Object.keys(codesObj); let drawnCount = 0;
        codes.forEach((code, i) => {
            let rawData = years.map(y => (parseFloat(p[cMain + '_' + code + '_' + y]) || 0) / 1e6);
            if (rawData.some(v => v > 0)) {
                let labelText = codesObj[code].length > 70 ? codesObj[code].substring(0, 70) + '...' : codesObj[code];
                createSingleChart(container, labelText, rawData, `hsl(${(i * 360) / Math.max(codes.length, 1)}, 80%, 65%)`, 'transparent', false);
                drawnCount++;
            }
        });
        if (drawnCount === 0) container.innerHTML = '<div style="color:#888; text-align:center; padding:30px;">На жаль, по цій категорії немає даних.</div>';
    }
}

function updateUI(redrawSide = true) {
    if (!hierarchy[cy]) return;
    const isIndex = viewMode === 'index';
    document.getElementById('data-controls').style.display = isIndex ? 'none' : 'block';
    document.getElementById('index-controls').style.display = isIndex ? 'block' : 'none';

    if (isIndex) {
        document.getElementById('legend-bal').style.display = 'none'; document.getElementById('legend-heat').style.display = 'none'; document.getElementById('legend-cluster').style.display = 'block';
        const clusterCount = cl === 'l1' ? 3 : (cl === 'l2' ? 5 : 10); let legendHtml = '';
        for(let i=0; i<clusterCount; i++) { let rgb = clusterColors[i].slice(0,3).join(','); legendHtml += `<div class="cluster-item"><span class="dot" style="background:rgb(${rgb})"></span>Кластер ${i+1}</div>`; }
        document.getElementById('legend-clusters').innerHTML = legendHtml;
    } else {
        document.getElementById('legend-cluster').style.display = 'none';
        const sub = document.getElementById('sub-metric'), det = document.getElementById('detail-lvl'), lblDet = document.getElementById('lbl-det'), lblSub = document.getElementById('lbl-sub');
        if (cMain === 'BAL') { sub.style.display = 'none'; det.style.display = 'none'; lblDet.style.display = 'none'; lblSub.style.display = 'none'; document.getElementById('legend-bal').style.display = 'block'; document.getElementById('legend-heat').style.display = 'none'; } 
        else {
            sub.style.display = 'block'; det.style.display = 'block'; lblDet.style.display = 'block'; lblSub.style.display = 'block'; document.getElementById('legend-bal').style.display = 'none'; document.getElementById('legend-heat').style.display = 'block';
            const availableLevels = hierarchy[cy]?.[cl]?.[cMain] || {};
            Array.from(det.options).forEach(opt => opt.disabled = !availableLevels[opt.value] || Object.keys(availableLevels[opt.value]).length === 0);
            if (det.options[det.selectedIndex].disabled) det.value = Array.from(det.options).find(o => !o.disabled)?.value || "4";
            cDetail = det.value; let previousSelection = cSub; sub.innerHTML = ''; const codes = availableLevels[cDetail] || {};
            if(Object.keys(codes).length === 0) { sub.add(new Option("Немає даних", "")); cSub = ""; } 
            else { Object.keys(codes).forEach(code => sub.add(new Option(codes[code], code))); if (Object.keys(codes).includes(previousSelection)) { sub.value = previousSelection; cSub = previousSelection; } else { cSub = sub.options[0].value; } }
        }
    }
    saveUIState(); if (deckgl) updateMap(); if (selectedFeature && redrawSide) { drawSideCharts(); refreshCompareButtonState(); }
}


// ==============================================================================
// 📋 БЛОК 6: БУФЕР ПОРІВНЯННЯ ТА IFRAME (ВІДЖЕТИ)
// ==============================================================================
function toggleCompareItem() {
    let selId = getCurrentSelectionID(); if (!selId) return;
    let idx = compareBuffer.findIndex(item => item.id === selId);
    if (idx > -1) { compareBuffer.splice(idx, 1); } else {
        if (compareBuffer.length >= 4) { alert('Максимум 4 панелі для порівняння!'); return; }
        let metricName = document.getElementById('main-metric').options[document.getElementById('main-metric').selectedIndex].text;
        compareBuffer.push({ id: selId, code: selectedFeature.properties.BUDGET_CODE, level: cl, main: cMain, detail: cDetail, regionName: selectedFeature.properties.display_name, metricDesc: cMain === 'BAL' ? 'Основні показники' : `Деталізація: ${cDetail} знаки (${metricName})`, pcaCategory: cIndex });
    }
    localStorage.setItem('compareBuffer', JSON.stringify(compareBuffer)); refreshCompareButtonState(); updateCompareWidget();
}

function removeFromCompare(id) { compareBuffer = compareBuffer.filter(item => item.id !== id); localStorage.setItem('compareBuffer', JSON.stringify(compareBuffer)); refreshCompareButtonState(); updateCompareWidget(); }

function refreshCompareButtonState() {
    let selId = getCurrentSelectionID(); if (!selId) return;
    let btn = document.getElementById('btn-select-compare');
    if (compareBuffer.some(item => item.id === selId)) { btn.innerText = "✖ Видалити панель з буфера"; btn.style.background = "#ff1744"; } 
    else { btn.innerText = "+ Зберегти цю панель в буфер"; btn.style.background = "#2b82d9"; }
}

function updateCompareWidget() {
    document.getElementById('compare-count').innerText = compareBuffer.length; let listDiv = document.getElementById('compare-list');
    if (compareBuffer.length === 0) { listDiv.innerHTML = "Немає збережених панелей"; } 
    else { listDiv.innerHTML = compareBuffer.map(item => `<div class="compare-li"><div class="compare-li-header"><span>${item.regionName} <span style="color:#666; font-size:10px;">(${item.level})</span></span><span class="compare-li-remove" onclick="removeFromCompare('${item.id}')">✕</span></div><div class="compare-li-desc">${item.metricDesc}</div></div>`).join(''); }
    const isDisabled = compareBuffer.length < 1; document.getElementById('btn-go-compare').disabled = isDisabled; document.getElementById('btn-go-pca').disabled = isDisabled;
}

function openScreen(url) { const iframe = document.getElementById('app-iframe'); iframe.src = url; iframe.style.display = 'block'; }
function openOldDashboard() { openScreen('/old_dashboard/index.html'); document.getElementById('btn-close-iframe').style.display = 'block'; }
window.closeCompareScreen = function() { const iframe = document.getElementById('app-iframe'); iframe.style.display = 'none'; iframe.src = ''; document.getElementById('btn-close-iframe').style.display = 'none'; };


// ==============================================================================
// 🖱️ БЛОК 7: ОБРОБНИКИ ПОДІЙ (EVENT LISTENERS)
// ==============================================================================
const searchInput = document.getElementById('search-input'), searchResults = document.getElementById('search-results');
searchInput.addEventListener('input', function(e) { const query = e.target.value.toLowerCase(); searchResults.innerHTML = ''; if (query.length < 2) { searchResults.style.display = 'none'; return; } const matches = levelData[cl].features.filter(f => f.properties.display_name && f.properties.display_name.toLowerCase().includes(query)); if (matches.length > 0) { matches.forEach(f => { const div = document.createElement('div'); div.className = 'search-item'; div.innerText = f.properties.display_name; div.onclick = () => { deckgl.setProps({ initialViewState: { longitude: parseFloat(f.properties.lon), latitude: parseFloat(f.properties.lat), zoom: cl === 'l1' ? 7 : (cl === 'l2' ? 8.5 : 10), pitch: 45, bearing: 0, transitionDuration: 1500, transitionInterpolator: new deck.FlyToInterpolator() } }); selectedFeature = f; openPanel(); searchInput.value = f.properties.display_name; searchResults.style.display = 'none'; }; searchResults.appendChild(div); }); searchResults.style.display = 'block'; } else { searchResults.style.display = 'none'; } });
document.addEventListener('click', function(e) { if (!document.getElementById('search-container').contains(e.target)) { searchResults.style.display = 'none'; } });

document.getElementById('year-slider').oninput = e => { cy = e.target.value; document.getElementById('year-display').innerText = cy; updateUI(false); };
document.getElementById('view-mode').onchange = e => { viewMode = e.target.value; updateUI(true); };
document.getElementById('main-metric').onchange = e => { cMain = e.target.value; updateUI(true); };
document.getElementById('detail-lvl').onchange = e => { cDetail = e.target.value; updateUI(true); };
document.getElementById('level-switcher').onchange = e => { cl = e.target.value; closePanel(); searchInput.value = ''; updateUI(true); };
document.getElementById('sub-metric').onchange = e => { cSub = e.target.value; updateMap(); saveUIState(); refreshCompareButtonState(); };

// Слухач для індексів (реагує на вибір як кластерів, так і похибок)
document.getElementById('index-metric').onchange = async e => { 
    cIndex = e.target.value; 
    document.body.style.cursor = 'wait';
    document.getElementById('index-metric').style.opacity = '0.5';
    
    await fetchIndexData(); // Тепер це розумна функція!
    
    document.body.style.cursor = 'default';
    document.getElementById('index-metric').style.opacity = '1';
    saveUIState();
    updateMap(); 
    if(selectedFeature) drawSideCharts(); 
};

// Делегування події для динамічного селекта 'comp-switcher'
document.getElementById('index-controls').addEventListener('change', (e) => {
    if (e.target.id === 'comp-switcher') {
        // Симулюємо зміну індексу, щоб запустити оновлення даних з новою розмірністю
        document.getElementById('index-metric').dispatchEvent(new Event('change'));
    }
});

// Запускаємо лоадер після завантаження сторінки
document.addEventListener('DOMContentLoaded', () => {
    loadFiltersConfig().then(() => {
        loadDataFromServer();
    });
});

// ==============================================================================
// 📷 РЕЖИМ ПРЕЗЕНТАЦІЇ (ZEN MODE)
// ==============================================================================
let isZenMode = false;

function toggleZenMode() {
    isZenMode = !isZenMode;
    const exitBtn = document.getElementById('btn-exit-zen');
    
    if (isZenMode) {
        document.body.classList.add('zen-mode');
        exitBtn.style.display = 'block';
        closePanel(); // Ховаємо бічну панель графіків, якщо вона відкрита
    } else {
        document.body.classList.remove('zen-mode');
        exitBtn.style.display = 'none';
    }
}

// Вихід по клавіші ESC
document.addEventListener('keydown', function(event) {
    if (event.key === "Escape" && isZenMode) {
        toggleZenMode();
    }
});