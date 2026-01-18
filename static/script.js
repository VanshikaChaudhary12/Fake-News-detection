const newsForm = document.getElementById('newsForm');
const newsText = document.getElementById('newsText');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const resultsPanel = document.getElementById('resultsPanel');
const resultDisplay = document.getElementById('resultDisplay');
const resultBadge = document.getElementById('resultBadge');
const badgeIcon = document.getElementById('badgeIcon');
const resultTitle = document.getElementById('resultTitle');
const resultSubtitle = document.getElementById('resultSubtitle');
const confidenceMeter = document.getElementById('confidenceMeter');
const confidenceValue = document.getElementById('confidenceValue');
const meterFill = document.getElementById('meterFill');
const resultMetrics = document.getElementById('resultMetrics');
const resultActions = document.getElementById('resultActions');
const responseTime = document.getElementById('responseTime');
const loadingModal = document.getElementById('loadingModal');
const themeToggle = document.getElementById('themeToggle');
const totalAnalyzed = document.getElementById('totalAnalyzed');
const historySection = document.getElementById('historySection');
const historyGrid = document.getElementById('historyGrid');
const detailModal = document.getElementById('detailModal');
const modalBody = document.getElementById('modalBody');

let currentResult = null;
let analysisHistory = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
let totalCount = parseInt(localStorage.getItem('totalAnalyzed') || '0');

const sampleArticles = [
    {
        content: "Scientists at MIT have developed a new method for detecting fake news using machine learning algorithms. The research, published in the journal Nature, shows promising results in identifying misinformation with 95% accuracy. The team used natural language processing techniques to analyze text patterns and linguistic features that distinguish authentic news from fabricated content."
    },
    {
        content: "BREAKING: Aliens have landed in New York City and are demanding to speak with world leaders immediately! Government sources confirm that extraterrestrial beings arrived in flying saucers this morning. Citizens are advised to stay indoors while negotiations begin. This unprecedented event will change human history forever!"
    }
];

document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    updateTotalCount();
    loadHistory();
    
    newsText.addEventListener('input', () => {
        const len = newsText.value.length;
        charCount.textContent = len;
        analyzeBtn.disabled = newsText.value.trim().length < 10;
    });
    
    clearBtn.addEventListener('click', () => {
        newsText.value = '';
        newsText.dispatchEvent(new Event('input'));
        resultsPanel.style.display = 'none';
    });
    
    newsText.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter' && !analyzeBtn.disabled) {
            e.preventDefault();
            newsForm.dispatchEvent(new Event('submit'));
        }
    });
});

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    themeToggle.innerHTML = newTheme === 'light' 
        ? '<i class="fas fa-moon"></i>' 
        : '<i class="fas fa-sun"></i>';
});

function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeToggle.innerHTML = savedTheme === 'light' 
        ? '<i class="fas fa-moon"></i>' 
        : '<i class="fas fa-sun"></i>';
}

newsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = newsText.value.trim();
    
    if (!text || text.length < 10) {
        alert('Please enter at least 10 characters for accurate analysis.');
        return;
    }
    
    showLoading();
    const startTime = Date.now();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `news_text=${encodeURIComponent(text)}`
        });
        
        const data = await response.json();
        const endTime = Date.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        
        hideLoading();
        
        if (data.error) {
            showError(data.message);
        } else {
            showResult(data, duration);
            saveToHistory(text, data);
            updateTotalCount();
        }
        
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        alert('Network error. Please check your connection and try again.');
    }
});

function showLoading() {
    loadingModal.style.display = 'flex';
    analyzeBtn.disabled = true;
    
    let stepIndex = 0;
    const steps = ['step1', 'step2', 'step3'];
    
    const interval = setInterval(() => {
        if (stepIndex < steps.length) {
            document.getElementById(steps[stepIndex]).classList.add('active');
            stepIndex++;
        } else {
            clearInterval(interval);
        }
    }, 500);
}

function hideLoading() {
    loadingModal.style.display = 'none';
    analyzeBtn.disabled = false;
    
    document.querySelectorAll('.progress-step').forEach(step => {
        step.classList.remove('active');
    });
}

function showResult(data, duration) {
    currentResult = { ...data, duration };
    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
    
    const isFake = data.prediction === 0;
    
    badgeIcon.className = `badge-icon ${isFake ? 'fake' : 'real'}`;
    badgeIcon.innerHTML = `<i class="fas ${isFake ? 'fa-triangle-exclamation' : 'fa-circle-check'}"></i>`;
    
    resultTitle.textContent = isFake ? 'Fake News Detected' : 'Real News Detected';
    resultSubtitle.textContent = data.message;
    
    confidenceMeter.style.display = 'block';
    confidenceValue.textContent = `${data.confidence_percentage}%`;
    
    setTimeout(() => {
        meterFill.style.width = `${data.confidence_percentage}%`;
    }, 300);
    
    resultMetrics.style.display = 'grid';
    responseTime.textContent = `${duration}s`;
    
    resultActions.style.display = 'grid';
}

function showError(message) {
    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
    
    badgeIcon.className = 'badge-icon';
    badgeIcon.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    badgeIcon.innerHTML = '<i class="fas fa-circle-xmark"></i>';
    
    resultTitle.textContent = 'Analysis Error';
    resultSubtitle.textContent = message;
    
    confidenceMeter.style.display = 'none';
    resultMetrics.style.display = 'none';
    resultActions.style.display = 'none';
}

function saveToHistory(text, data) {
    const historyItem = {
        id: Date.now(),
        text: text.substring(0, 150),
        prediction: data.prediction,
        label: data.label,
        confidence: data.confidence_percentage,
        timestamp: new Date().toISOString()
    };
    
    analysisHistory.unshift(historyItem);
    if (analysisHistory.length > 10) analysisHistory.pop();
    
    localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));
    
    totalCount++;
    localStorage.setItem('totalAnalyzed', totalCount.toString());
    
    loadHistory();
}

function loadHistory() {
    if (analysisHistory.length === 0) {
        historySection.style.display = 'none';
        return;
    }
    
    historySection.style.display = 'block';
    historyGrid.innerHTML = '';
    
    analysisHistory.forEach(item => {
        const div = document.createElement('div');
        div.className = `history-item ${item.prediction === 0 ? 'fake' : 'real'}`;
        div.innerHTML = `
            <div class="history-item-header">
                <span class="history-item-label">${item.label} (${item.confidence}%)</span>
                <span class="history-item-time">${formatTime(item.timestamp)}</span>
            </div>
            <div class="history-item-text">${item.text}...</div>
        `;
        div.onclick = () => {
            newsText.value = item.text;
            newsText.dispatchEvent(new Event('input'));
            window.scrollTo({ top: 0, behavior: 'smooth' });
        };
        historyGrid.appendChild(div);
    });
}

function clearHistory() {
    if (confirm('Clear all analysis history?')) {
        analysisHistory = [];
        localStorage.removeItem('analysisHistory');
        loadHistory();
    }
}

function updateTotalCount() {
    totalAnalyzed.textContent = totalCount;
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);
    
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
}

function copyResult() {
    if (!currentResult) return;
    
    const text = `Fake News Detection Result\n\nPrediction: ${currentResult.label}\nConfidence: ${currentResult.confidence_percentage}%\nResponse Time: ${currentResult.duration}s\nAnalysis: ${currentResult.message}`;
    
    navigator.clipboard.writeText(text).then(() => {
        alert('✅ Result copied to clipboard!');
    });
}

function shareResult() {
    if (!currentResult) return;
    
    const text = `I analyzed news content using AI: ${currentResult.label} (${currentResult.confidence_percentage}% confidence)`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Fake News Detection Result',
            text: text
        });
    } else {
        copyResult();
    }
}

function showDetails() {
    if (!currentResult) return;
    
    const isFake = currentResult.prediction === 0;
    
    modalBody.innerHTML = `
        <div style="margin-bottom: 25px;">
            <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <i class="fas fa-chart-pie"></i> Analysis Summary
            </h3>
            <div style="display: grid; gap: 12px;">
                <div class="metric-item">
                    <i class="fas fa-tag"></i>
                    <div class="metric-content">
                        <span class="metric-label">Classification</span>
                        <span class="metric-value">${currentResult.label}</span>
                    </div>
                </div>
                <div class="metric-item">
                    <i class="fas fa-percent"></i>
                    <div class="metric-content">
                        <span class="metric-label">Confidence Score</span>
                        <span class="metric-value">${currentResult.confidence_percentage}%</span>
                    </div>
                </div>
                <div class="metric-item">
                    <i class="fas fa-clock"></i>
                    <div class="metric-content">
                        <span class="metric-label">Processing Time</span>
                        <span class="metric-value">${currentResult.duration}s</span>
                    </div>
                </div>
                <div class="metric-item">
                    <i class="fas fa-exclamation-triangle"></i>
                    <div class="metric-content">
                        <span class="metric-label">Risk Level</span>
                        <span class="metric-value">${isFake ? 'High' : 'Low'}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div style="margin-bottom: 25px;">
            <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <i class="fas fa-cogs"></i> Technical Details
            </h3>
            <div style="display: grid; gap: 12px;">
                <div class="metric-item">
                    <i class="fas fa-robot"></i>
                    <div class="metric-content">
                        <span class="metric-label">Model</span>
                        <span class="metric-value">Logistic Regression</span>
                    </div>
                </div>
                <div class="metric-item">
                    <i class="fas fa-layer-group"></i>
                    <div class="metric-content">
                        <span class="metric-label">Features</span>
                        <span class="metric-value">TF-IDF (10,000)</span>
                    </div>
                </div>
                <div class="metric-item">
                    <i class="fas fa-hashtag"></i>
                    <div class="metric-content">
                        <span class="metric-label">N-grams</span>
                        <span class="metric-value">1-3 grams</span>
                    </div>
                </div>
                <div class="metric-item">
                    <i class="fas fa-filter"></i>
                    <div class="metric-content">
                        <span class="metric-label">Preprocessing</span>
                        <span class="metric-value">Lemmatization + Stopwords</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div style="padding: 20px; background: var(--glass-bg); border-radius: 12px; border: 1px solid var(--glass-border);">
            <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <i class="fas fa-lightbulb"></i> Recommendation
            </h3>
            <p style="color: var(--text-secondary); line-height: 1.6;">
                ${isFake 
                    ? '⚠️ This content shows characteristics of fake news. We recommend verifying the information through multiple trusted sources before sharing or acting on it. Look for official sources, check publication dates, and cross-reference with reputable news outlets.'
                    : '✅ This content appears to be legitimate news. However, always practice critical thinking and verify important information through multiple sources. Check the author credentials, publication date, and look for supporting evidence from other reliable sources.'}
            </p>
        </div>
    `;
    
    detailModal.style.display = 'flex';
}

function closeModal() {
    detailModal.style.display = 'none';
}

function downloadReport() {
    if (!currentResult) return;
    
    const report = `
╔═══════════════════════════════════════════════════════════╗
║         FAKE NEWS DETECTION ANALYSIS REPORT              ║
╚═══════════════════════════════════════════════════════════╝

Analysis Date: ${new Date().toLocaleString()}

═══════════════════════════════════════════════════════════
CLASSIFICATION RESULT
═══════════════════════════════════════════════════════════
Prediction: ${currentResult.label}
Confidence: ${currentResult.confidence_percentage}%
Processing Time: ${currentResult.duration}s
Risk Level: ${currentResult.prediction === 0 ? 'High' : 'Low'}

═══════════════════════════════════════════════════════════
TECHNICAL DETAILS
═══════════════════════════════════════════════════════════
Algorithm: Enhanced Logistic Regression
Features: TF-IDF Vectorization (10,000 features)
N-grams: 1-3 grams
Preprocessing: Lemmatization + Stopword Removal
Model Accuracy: 94%+

═══════════════════════════════════════════════════════════
ANALYZED TEXT
═══════════════════════════════════════════════════════════
${newsText.value}

═══════════════════════════════════════════════════════════
RECOMMENDATION
═══════════════════════════════════════════════════════════
${currentResult.prediction === 0 
    ? 'This content shows characteristics of fake news. Verify through multiple trusted sources before sharing.'
    : 'This content appears legitimate. However, always practice critical thinking and verify important information.'}

═══════════════════════════════════════════════════════════
Generated by AI-Based Fake News Detection System
B.Tech CSE Final Year Project
═══════════════════════════════════════════════════════════
    `.trim();
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fake-news-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

function loadSample(index) {
    if (index >= 0 && index < sampleArticles.length) {
        newsText.value = sampleArticles[index].content;
        newsText.dispatchEvent(new Event('input'));
        newsText.focus();
        resultsPanel.style.display = 'none';
    }
}

function analyzeURL() {
    alert('URL analysis feature coming soon! This will fetch article content from URLs and analyze it automatically.');
}

window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
        closeModal();
    }
});
