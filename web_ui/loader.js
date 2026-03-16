// --- ПЛАВНИЙ ПРОГРЕС-БАР ПО БАЙТАХ (ТОЧНИЙ ПІДРАХУНОК) ---
const downloadStats = {};

function updateByteProgress() {
    let currentLoaded = 0;
    let currentTotal = 0;
    let isTotalKnown = true;

    for (const key in downloadStats) {
        currentLoaded += downloadStats[key].loaded;
        if (downloadStats[key].total > 0) {
            currentTotal += downloadStats[key].total;
        } else {
            isTotalKnown = false; // Якщо хоч один файл прийшов без Content-Length
        }
    }

    const mbLoaded = (currentLoaded / (1024 * 1024)).toFixed(2);

    if (isTotalKnown && currentTotal > 0) {
        // Якщо сервер сказав точний розмір - рахуємо реальні відсотки
        let percent = Math.min(100, Math.round((currentLoaded / currentTotal) * 100));
        document.getElementById('progress-bar').style.width = percent + '%';
        document.getElementById('progress-text').innerText = percent + '%';
        document.getElementById('loading-details').innerText = `Завантажено ${mbLoaded} MB з ${(currentTotal / (1024*1024)).toFixed(2)} MB`;
    } else {
        // Якщо розмір невідомий (через Gzip) - показуємо безкінечний лоадер, але з реальними мегабайтами
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-bar').style.background = 'linear-gradient(90deg, #00ff88, #00bfff, #00ff88)';
        document.getElementById('progress-bar').style.backgroundSize = '200% 100%';
        document.getElementById('progress-bar').style.animation = 'pulse 2s linear infinite';
        
        document.getElementById('progress-text').innerText = 'Завантаження...';
        document.getElementById('loading-details').innerText = `Отримано даних: ${mbLoaded} MB`;
    }
}

async function fetchWithBytes(url, id) {
    downloadStats[id] = { loaded: 0, total: 0 }; 
    
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Помилка HTTP: ${response.status} для ${url}`);

    // ТРЕКАЄМО РЕАЛЬНУ ВАГУ З СЕРВЕРА
    const contentLength = response.headers.get('content-length');
    const xFileSize = response.headers.get('x-file-size'); // Шукаємо наш кастомний заголовок
    
    // Беремо X-File-Size (якщо він є), інакше пробуємо стандартний Content-Length
    console.log(`[Файл: ${id}] Content-Length:`, contentLength, '| X-File-Size:', xFileSize);
    const totalBytes = xFileSize || contentLength;
    downloadStats[id].total = totalBytes ? parseInt(totalBytes, 10) : 0;

    const reader = response.body.getReader();
    let loaded = 0;
    const chunks = [];

    // Читаємо потік байтів
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
            loaded += value.length;
            chunks.push(value);
            downloadStats[id].loaded = loaded;
            updateByteProgress(); 
        }
    }

    // Збираємо файл
    const body = new Uint8Array(loaded);
    let position = 0;
    for (let chunk of chunks) {
        body.set(chunk, position);
        position += chunk.length;
    }

    return JSON.parse(new TextDecoder("utf-8").decode(body));
}