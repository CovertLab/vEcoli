/**
 * AI-Powered Documentation Search
 * Adds an "Ask AI" button that answers questions using the documentation.
 */

const CONFIG = {
    apiEndpoint: 'https://api.groq.com/openai/v1/chat/completions',
    model: 'llama-3.1-8b-instant',
    maxContextDocs: 5,
    maxTokens: 1000
};

let searchIndex = null;

function getApiKey() {
    const userKey = localStorage.getItem('ai_search_api_key');
    if (userKey) return userKey;
    const buildKey = '__GROQ_API_KEY__';
    if (buildKey && !buildKey.startsWith('__')) return buildKey;
    return null;
}

function saveApiKey(key) {
    localStorage.setItem('ai_search_api_key', key);
}

function clearApiKey() {
    localStorage.removeItem('ai_search_api_key');
}

async function loadSearchIndex() {
    if (searchIndex !== null) return searchIndex;
    
    const pathParts = window.location.pathname.split('/');
    const pathsToTry = [];
    
    for (let i = pathParts.length - 1; i >= 0; i--) {
        const basePath = pathParts.slice(0, i).join('/');
        pathsToTry.push(`${basePath}/_static/search_index.json`);
    }
    pathsToTry.push('_static/search_index.json', '../_static/search_index.json', '../../_static/search_index.json');
    
    for (const path of pathsToTry) {
        try {
            const response = await fetch(path);
            if (response.ok) {
                searchIndex = await response.json();
                console.log('AI Search: Loaded', searchIndex.length, 'documents');
                return searchIndex;
            }
        } catch (e) {}
    }
    console.error('AI Search: Failed to load search index');
    return [];
}

function findRelevantDocs(question, index) {
    const stopWords = new Set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'whom', 'how', 'why', 'when', 'where'
    ]);
    
    const keywords = question.toLowerCase()
        .split(/\s+/)
        .filter(w => w.length > 1 && !stopWords.has(w))
        .map(w => w.replace(/[^a-z0-9]/g, ''));
    
    const phrases = keywords.slice(0, -1).map((w, i) => w + ' ' + keywords[i + 1]);
    
    const scored = index.map(doc => {
        const titleLower = doc.title.toLowerCase();
        const contentLower = doc.content.toLowerCase();
        let score = 0;
        
        for (const keyword of keywords) {
            if (!keyword) continue;
            const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
            score += (titleLower.match(regex) || []).length * 20;
            score += (contentLower.match(regex) || []).length;
            if (titleLower.includes(keyword)) score += 50;
        }
        
        for (const phrase of phrases) {
            if ((titleLower + ' ' + contentLower).includes(phrase)) score += 15;
        }
        
        if (doc.content.length < 2000 && score > 0) score += 5;
        
        return { ...doc, score };
    });
    
    return scored.filter(d => d.score > 0).sort((a, b) => b.score - a.score).slice(0, CONFIG.maxContextDocs);
}

async function askLLM(question, relevantDocs) {
    const apiKey = getApiKey();
    if (!apiKey) throw new Error('NO_API_KEY');
    
    let context = relevantDocs.map(doc => 
        `## ${doc.title}\n${doc.content.slice(0, 2000)}`
    ).join('\n\n');
    
    const response = await fetch(CONFIG.apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: CONFIG.model,
            messages: [
                { role: 'system', content: 'You are a documentation assistant for Vivarium E. coli. Answer based ONLY on the provided context. If not found, say so. Be concise.' },
                { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` }
            ],
            max_tokens: CONFIG.maxTokens,
            temperature: 0.3
        })
    });
    
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

function createModal() {
    const modal = document.createElement('div');
    modal.id = 'ai-search-modal';
    modal.innerHTML = `
        <div class="ai-search-backdrop"></div>
        <div class="ai-search-dialog">
            <div class="ai-search-header">
                <h3>Ask AI about the Documentation</h3>
                <button class="ai-search-close">&times;</button>
            </div>
            <div class="ai-search-body">
                <input type="text" id="ai-search-input" placeholder="Ask a question..." autocomplete="off">
                <button id="ai-search-submit">Ask</button>
            </div>
            <div id="ai-search-result" class="ai-search-result"></div>
            <div id="ai-search-sources" class="ai-search-sources"></div>
            <div id="ai-search-settings" class="ai-search-settings">
                <details>
                    <summary>Settings</summary>
                    <div class="api-key-form">
                        <label>Groq API Key:</label>
                        <input type="password" id="ai-api-key-input" placeholder="Enter API key..." autocomplete="off">
                        <div class="api-key-buttons">
                            <button id="ai-api-key-save">Save</button>
                            <button id="ai-api-key-clear">Clear</button>
                        </div>
                        <p class="api-key-help">Get a free key at <a href="https://console.groq.com/keys" target="_blank">console.groq.com</a></p>
                    </div>
                </details>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    modal.querySelector('.ai-search-backdrop').addEventListener('click', closeModal);
    modal.querySelector('.ai-search-close').addEventListener('click', closeModal);
    modal.querySelector('#ai-search-submit').addEventListener('click', handleSearch);
    modal.querySelector('#ai-search-input').addEventListener('keypress', e => { if (e.key === 'Enter') handleSearch(); });
    modal.querySelector('#ai-api-key-save').addEventListener('click', () => {
        const key = document.getElementById('ai-api-key-input').value.trim();
        if (key) { saveApiKey(key); document.getElementById('ai-api-key-input').value = ''; alert('API key saved!'); }
    });
    modal.querySelector('#ai-api-key-clear').addEventListener('click', () => { clearApiKey(); alert('API key cleared.'); });
}

function openModal() {
    document.getElementById('ai-search-modal').classList.add('active');
    document.getElementById('ai-search-input').focus();
}

function closeModal() {
    document.getElementById('ai-search-modal').classList.remove('active');
}

async function handleSearch() {
    const input = document.getElementById('ai-search-input');
    const resultDiv = document.getElementById('ai-search-result');
    const sourcesDiv = document.getElementById('ai-search-sources');
    const question = input.value.trim();
    if (!question) return;
    
    resultDiv.innerHTML = '<p class="loading">Searching...</p>';
    sourcesDiv.innerHTML = '';
    let relevantDocs = [];
    
    try {
        const index = await loadSearchIndex();
        relevantDocs = findRelevantDocs(question, index);
        
        if (relevantDocs.length === 0) {
            resultDiv.innerHTML = '<p>No relevant documentation found.</p>';
            return;
        }
        
        resultDiv.innerHTML = '<p class="loading">Generating answer...</p>';
        const answer = await askLLM(question, relevantDocs);
        resultDiv.innerHTML = `<div class="answer">${formatAnswer(answer)}</div>`;
        showSources(sourcesDiv, relevantDocs);
    } catch (error) {
        if (error.message === 'NO_API_KEY') {
            resultDiv.innerHTML = '<p><strong>API key required.</strong> Add your Groq API key in Settings below.</p>';
            showSources(sourcesDiv, relevantDocs);
            document.querySelector('#ai-search-settings details').open = true;
        } else {
            resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    }
}

function formatAnswer(text) {
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function showSources(container, docs) {
    if (docs.length === 0) return;
    container.innerHTML = '<p><strong>Sources:</strong></p><ul>' +
        docs.map(doc => `<li><a href="${doc.path}">${doc.title}</a></li>`).join('') + '</ul>';
}

function addAskAIButton() {
    const searchBox = document.querySelector('.wy-side-nav-search') || document.querySelector('[role="search"]');
    if (!searchBox) return;
    
    const button = document.createElement('button');
    button.id = 'ask-ai-button';
    button.textContent = '🤖 Ask AI';
    button.addEventListener('click', openModal);
    
    const searchForm = searchBox.querySelector('form') || searchBox;
    searchForm.parentNode.insertBefore(button, searchForm.nextSibling);
}

function initAISearch() {
    createModal();
    addAskAIButton();
    loadSearchIndex();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAISearch);
} else {
    initAISearch();
}
