// State
let currentThreadId = localStorage.getItem('agentic_rag_thread_id');
let currentSources = {}; // Maps source ID (e.g. S1) to chunk data

// DOM Elements
const sidebar = document.getElementById('sidebar');
const openSidebarBtn = document.getElementById('openSidebarBtn');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const overlay = document.getElementById('overlay');

const kbStatusDot = document.getElementById('kbStatusDot');
const kbStatusText = document.getElementById('kbStatusText');
const kbDetails = document.getElementById('kbDetails');
const kbDocTitle = document.getElementById('kbDocTitle');
const kbChunkCount = document.getElementById('kbChunkCount');

const uploadForm = document.getElementById('uploadForm');
const kbFile = document.getElementById('kbFile');
const fileLabel = document.getElementById('fileLabel');
const uploadBtn = document.getElementById('uploadBtn');
const resetBtn = document.getElementById('resetBtn');

const chatContainer = document.getElementById('chatContainer');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesWrapper = document.getElementById('messagesWrapper');
const typingContainer = document.getElementById('typingContainer');
const typingStatus = document.getElementById('typingStatus');

const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');

const sourceModal = document.getElementById('sourceModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const sourceModalTitle = document.getElementById('sourceModalTitle');
const sourceModalBody = document.getElementById('sourceModalBody');
const suggestionBtns = document.querySelectorAll('.suggestion-btn');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    checkKbStatus();

    if (!currentThreadId) {
        currentThreadId = crypto.randomUUID();
        localStorage.setItem('agentic_rag_thread_id', currentThreadId);
    }
});

// Sidebar & Mobile Nav
openSidebarBtn.addEventListener('click', () => {
    sidebar.classList.add('open');
    overlay.classList.add('active');
});

function closeSidebar() {
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
}
closeSidebarBtn.addEventListener('click', closeSidebar);
overlay.addEventListener('click', closeSidebar);

// Knowledge Base Status
async function checkKbStatus() {
    try {
        const res = await fetch('/api/v1/knowledge-base');
        const data = await res.json();

        if (data.ready) {
            kbStatusDot.className = 'status-dot ready';
            kbStatusText.textContent = 'Active & Ready';
            kbDetails.style.display = 'block';
            kbDocTitle.textContent = data.title || data.source;
            kbChunkCount.textContent = data.chunk_count;
        } else {
            kbStatusDot.className = 'status-dot error';
            kbStatusText.textContent = 'Not Ready';
            kbDetails.style.display = 'none';
        }
    } catch (err) {
        kbStatusDot.className = 'status-dot error';
        kbStatusText.textContent = 'Error connecting';
    }
}

// Upload Handling
kbFile.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileLabel.querySelector('span').textContent = e.target.files[0].name;
        fileLabel.classList.add('has-file');
    }
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!kbFile.files.length) return;

    const file = kbFile.files[0];
    const formData = new FormData();
    formData.append('file', file);

    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';

    try {
        const res = await fetch('/api/v1/knowledge-base/upload', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (res.ok) {
            alert(`Success: ${data.message}`);
            startNewChat();
            checkKbStatus();
            closeSidebar();
        } else {
            alert(`Error: ${getErrorMessage(data, 'Upload failed.')}`);
        }
    } catch (err) {
        alert('Upload failed due to network error.');
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload & Index';
        kbFile.value = '';
        fileLabel.querySelector('span').textContent = 'Choose PDF or TXT';
        fileLabel.classList.remove('has-file');
    }
});

// Reset Document
resetBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to restore the default document? This will replace the current index.')) return;

    resetBtn.disabled = true;
    resetBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Restoring...';

    try {
        const res = await fetch('/api/v1/knowledge-base/default', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            alert('Default document restored successfully.');
            startNewChat();
            checkKbStatus();
            closeSidebar();
        } else {
            alert(`Error: ${getErrorMessage(data, 'Could not restore the document.')}`);
        }
    } catch (err) {
        alert('Reset failed.');
    } finally {
        resetBtn.disabled = false;
        resetBtn.innerHTML = '<i class="fas fa-undo"></i> Restore Default Document';
    }
});

// Chat Functionality
suggestionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        chatInput.value = btn.dataset.query;
        chatForm.dispatchEvent(new Event('submit'));
    });
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg) return;

    chatInput.value = '';
    welcomeScreen.style.display = 'none';

    addMessage(msg, 'user');
    showTyping('Thinking...');

    try {
        const res = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, thread_id: currentThreadId })
        });

        const data = await res.json();

        if (res.ok) {
            currentThreadId = data.thread_id;
            localStorage.setItem('agentic_rag_thread_id', currentThreadId);

            let answerText = data.answer;
            const paragraphs = answerText.split(/\n\n+/);
            answerText = paragraphs.map(p => `<p style="white-space: pre-line">${escapeHtml(p)}</p>`).join('');
            let sourcesHtml = '';

            if (data.sources && data.sources.length > 0) {
                data.sources.forEach(source => {
                    currentSources[source.id] = source;
                    const sourceId = escapeHtml(source.id);
                    const sourceTitle = escapeHtml(source.title);
                    const sourcePage = escapeHtml(source.page || 'N/A');
                    const citationRegex = new RegExp(`\\[${source.id}\\]`, 'g');
                    answerText = answerText.replace(
                        citationRegex,
                        `<button type="button" class="citation-badge" data-source-id="${sourceId}" aria-label="View source ${sourceId}">${sourceId}</button>`
                    );

                    sourcesHtml += `
                        <button type="button" class="source-tag" data-source-id="${sourceId}">
                            <i class="fas fa-file-alt"></i> ${sourceId} - ${sourceTitle} (Page ${sourcePage})
                        </button>
                    `;
                });

                answerText += `<div class="sources-container">${sourcesHtml}</div>`;
            }

            addMessage(answerText, 'bot', true);
        } else {
            addMessage(`Error: ${getErrorMessage(data, 'Something went wrong.')}`, 'bot');
        }
    } catch (err) {
        addMessage('Network error. Please try again.', 'bot');
    } finally {
        hideTyping();
    }
});

// New Chat
newChatBtn.addEventListener('click', () => {
    startNewChat();
});

function startNewChat() {
    if (currentThreadId) {
        fetch(`/api/v1/threads/${currentThreadId}`, { method: 'DELETE' }).catch(console.error);
    }

    currentThreadId = crypto.randomUUID();
    localStorage.setItem('agentic_rag_thread_id', currentThreadId);

    messagesWrapper.innerHTML = '';
    welcomeScreen.style.display = 'block';
    currentSources = {};
}

// UI Helpers
function addMessage(content, role, isHtml = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const icon = role === 'user' ? 'fa-user' : 'fa-robot';

    let contentHtml = '';
    if (isHtml) {
        contentHtml = content;
    } else {
        const paragraphs = content.split(/\n\n+/);
        contentHtml = paragraphs.map(p => `<p style="white-space: pre-line">${escapeHtml(p)}</p>`).join('');
    }

    msgDiv.innerHTML = `
        <div class="avatar">
            <i class="fas ${icon}"></i>
        </div>
        <div class="message-content">
            ${contentHtml}
        </div>
    `;

    messagesWrapper.appendChild(msgDiv);
    scrollToBottom();
}

function showTyping(text) {
    typingStatus.textContent = text;
    typingContainer.style.display = 'flex';
    sendBtn.disabled = true;
    chatInput.disabled = true;
    scrollToBottom();
}

function hideTyping() {
    typingContainer.style.display = 'none';
    sendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(unsafe) {
    return String(unsafe ?? '')
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

function getErrorMessage(data, fallback) {
    return data?.detail?.message || data?.message || fallback;
}

function showSource(sourceId, trigger) {
    const source = currentSources[sourceId];
    if (!source) return;

    lastModalTrigger = trigger;
    sourceModalTitle.textContent = `Source ${sourceId}: ${source.title}`;
    sourceModalBody.replaceChildren();

    const metadata = document.createElement('dl');
    metadata.className = 'source-metadata';
    addSourceDetail(metadata, 'Page', source.page);
    addSourceDetail(metadata, 'Section', source.section);

    if (source.source_url) {
        const link = document.createElement('a');
        link.className = 'source-link';
        link.href = source.source_url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = 'Open original source';
        sourceModalBody.append(metadata, link);
    } else {
        sourceModalBody.append(metadata);
    }

    sourceModal.classList.add('active');
    sourceModal.setAttribute('aria-hidden', 'false');
    closeModalBtn.focus();
}

function addSourceDetail(container, label, value) {
    if (!value) return;

    const row = document.createElement('div');
    const term = document.createElement('dt');
    const description = document.createElement('dd');
    term.textContent = label;
    description.textContent = value;
    row.append(term, description);
    container.append(row);
}

let lastModalTrigger = null;

messagesWrapper.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-source-id]');
    if (trigger) {
        showSource(trigger.dataset.sourceId, trigger);
    }
});

function closeSourceModal() {
    sourceModal.classList.remove('active');
    sourceModal.setAttribute('aria-hidden', 'true');
    lastModalTrigger?.focus();
    lastModalTrigger = null;
}

closeModalBtn.addEventListener('click', closeSourceModal);
sourceModal.addEventListener('click', (e) => {
    if (e.target === sourceModal) closeSourceModal();
});
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && sourceModal.classList.contains('active')) {
        closeSourceModal();
    }
});
