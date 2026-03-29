window.i18n = {
    currentLang: localStorage.getItem('currentLang') || 'uk',
    dataTranslations: {},
    uiTranslations: {},
    tCache: {},

    async init() {
        const files = [
            'geography.json', 'incomes.json', 'programmatic.json',
            'economic.json', 'functional.json', 'groups.json', 'other.json'
        ];
        try {
            const ts = new Date().getTime();
            const requests = files.map(file => 
                fetch(`/api/translations/${file}?t=${ts}`)
                    .then(res => res.ok ? res.json() : {})
                    .catch(() => ({}))
            );
            const results = await Promise.all(requests);

            results.forEach(data => {
                this.dataTranslations = { ...this.dataTranslations, ...data };
            });

            const uiRes = await fetch(`/api/translations/ui_elements.json?t=${ts}`);
            if (uiRes.ok) this.uiTranslations = await uiRes.json();

            this.tCache = {};
            this.updateStaticUI();
        } catch (e) {
            console.error(e);
        }
    },

    cleanEngText(text, code) {
        if (!text) return '';
        let clean = text.trim();
        if (code && clean.startsWith(code + " - ")) {
            clean = clean.substring(code.length + 3).trim();
        }
        return clean;
    },

    hasCyrillic(str) {
        return /[А-Яа-яЄєІіЇїҐґ]/.test(str);
    },

    t(stringObj) {
        if (!stringObj || typeof stringObj !== 'string') return stringObj;
        let s = stringObj.trim();
        if (this.tCache[s]) return this.tCache[s];

        let result = s;

        if (this.uiTranslations[this.currentLang] && this.uiTranslations[this.currentLang][s]) {
            result = this.uiTranslations[this.currentLang][s];
        } else if (this.currentLang === 'en' && this.uiTranslations['uk']) {
            const reverseKey = Object.keys(this.uiTranslations['uk']).find(key => this.uiTranslations['uk'][key] === s);
            if (reverseKey) result = this.uiTranslations['en'][reverseKey] || reverseKey;
        }

        if (result === s && this.currentLang !== 'uk') {
            const normalizeStr = (str) => {
                if (!str) return '';
                return str.toLowerCase().replace(/['"`’]/g, '').replace(/[\u00A0\s]+/g, ' ').replace(/[.,\-_]/g, '').trim();
            };

            const normStringObj = normalizeStr(s);

            if (s.includes('Деталізація') || s.includes('Detailing')) {
                let parts = s.split(':');
                let label = parts[0].trim();
                let content = parts[1].trim();

                let categoryPart = "";
                if (content.includes('(') && content.includes(')')) {
                    let start = content.indexOf('(');
                    categoryPart = content.substring(start);
                    content = content.substring(0, start).trim();
                }

                let transLabel = this.t(label);
                let transContent = content;
                if (content.includes('знак')) {
                    let num = content.replace(/\D/g, '');
                    let suffix = (num === "1") ? this.t("знак") : this.t("знаки");
                    transContent = `${num} ${suffix}`;
                } else {
                    transContent = this.t(content);
                }

                let transCategory = categoryPart;
                if (categoryPart.startsWith('(')) {
                    let inner = categoryPart.slice(1, -1);
                    transCategory = `(${this.t(inner)})`;
                }

                result = `${transLabel}: ${transContent} ${transCategory}`;
            } else {
                const dashIdx = s.indexOf(' - ');
                if (dashIdx !== -1) {
                    const codePart = s.substring(0, dashIdx).trim();
                    const matchByCode = Object.values(this.dataTranslations).find(item => item.code == codePart);
                    if (matchByCode && matchByCode.eng && !this.hasCyrillic(matchByCode.eng)) {
                        result = `${codePart} - ${this.cleanEngText(matchByCode.eng, codePart)}`;
                    }
                } else {
                    const exactMatchKey = Object.keys(this.dataTranslations).find(k => normalizeStr(k) === normStringObj);
                    if (exactMatchKey) {
                        const entry = this.dataTranslations[exactMatchKey];
                        if (entry.eng && !this.hasCyrillic(entry.eng)) {
                            result = this.cleanEngText(entry.eng, entry.code);
                        }
                    }
                }
            }
        }

        this.tCache[s] = result;
        return result;
    },

    updateStaticUI() {
        if (!this.uiTranslations[this.currentLang]) return;
        this.tCache = {};

        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (this.uiTranslations[this.currentLang][key]) el.innerHTML = this.uiTranslations[this.currentLang][key];
        });

        const legendTitle = document.querySelector('.legend-title');
        if (legendTitle) legendTitle.innerText = this.t(legendTitle.innerText);

        const legendLabels = document.querySelectorAll('.legend-labels span');
        legendLabels.forEach(span => { span.innerText = this.t(span.innerText); });

        const selectsToUpdate = ['view-mode', 'level-switcher', 'main-metric', 'detail-lvl', 'index-metric', 'comp-switcher', 'sub-metric'];
        selectsToUpdate.forEach(id => {
            const select = document.getElementById(id);
            if (select) {
                Array.from(select.options).forEach(opt => {
                    if (opt.dataset.originalText) opt.textContent = this.t(opt.dataset.originalText);
                });
                Array.from(select.querySelectorAll('optgroup')).forEach(optg => {
                    if (optg.dataset.originalLabel) optg.label = this.t(optg.dataset.originalLabel);
                });
            }
        });

        if (typeof updateUI === 'function') updateUI(false);
        if (typeof deckgl !== 'undefined' && deckgl && typeof updateMap === 'function') updateMap();
        
        if (typeof selectedFeature !== 'undefined' && selectedFeature) {
            const titleEl = document.getElementById('sp-title');
            if (titleEl) titleEl.innerText = this.t(selectedFeature.properties.display_name);
            const subtitleEl = document.getElementById('sp-subtitle');
            if (subtitleEl) subtitleEl.innerText = this.t(subtitleEl.innerText);
            if (typeof drawSideCharts === 'function') drawSideCharts();
        }
        
        if (typeof updateCompareWidget === 'function') updateCompareWidget();
    },

    switchLanguage() {
        const toggle = document.getElementById('lang-toggle');
        if (toggle) {
            this.currentLang = toggle.checked ? 'en' : 'uk';
            localStorage.setItem('currentLang', this.currentLang);
            this.tCache = {};
            this.updateStaticUI();
        }
    },

    async loadFiltersConfig() {
        try {
            const response = await fetch('/api/filters_config');
            const config = await response.json();

            const populateSelect = (selectId, optionsData) => {
                const select = document.getElementById(selectId);
                if (!select) return;
                select.innerHTML = '';
                optionsData.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt.value;
                    option.dataset.originalText = opt.label;
                    option.textContent = this.t(opt.label);
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
                const groups = [
                    { label: "PCA K-Means Clusters", data: config.analytical_indexes },
                    { label: "Model Accuracy (Errors)", data: config.pca_errors }
                ];
                groups.forEach(g => {
                    const og = document.createElement('optgroup');
                    og.dataset.originalLabel = g.label;
                    og.label = this.t(g.label);
                    g.data.forEach(opt => {
                        const o = document.createElement('option');
                        o.value = opt.value;
                        o.dataset.originalText = opt.label;
                        o.textContent = this.t(opt.label);
                        og.appendChild(o);
                    });
                    indexSelect.appendChild(og);
                });
                indexSelect.value = typeof cIndex !== 'undefined' ? cIndex : 'income';
            }
        } catch (e) {}
    }
};

Object.defineProperty(window, 'currentLang', {
    get: () => window.i18n.currentLang,
    set: (v) => { window.i18n.currentLang = v; }
});

window.t = (s) => window.i18n.t(s);
window.switchLanguage = () => window.i18n.switchLanguage();