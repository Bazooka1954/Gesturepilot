const socket = io();

const connDot         = document.getElementById('conn-dot');
const connLabel       = document.getElementById('conn-label');
const cards           = document.getElementById('cards');
const toast           = document.getElementById('toast');
const toastKey        = document.getElementById('toast-key');
const toastLabel      = document.getElementById('toast-label');
const menuStateLabel  = document.getElementById('menu-state-label');
const gestureDisplay  = document.getElementById('gesture-name-display');
const ringFill        = document.getElementById('ring-fill');
const holdRingWrap    = document.getElementById('hold-ring-wrap');
const pinchHint       = document.getElementById('pinch-hint');

socket.on('connect', () => {
    connDot.style.background  = '#00c853';
    connDot.style.boxShadow   = '0 0 6px #00c853';
    connLabel.textContent     = 'CONNECTED';
});
socket.on('disconnect', () => {
    connDot.style.background  = '#ef5350';
    connDot.style.boxShadow   = '0 0 6px #ef5350';
    connLabel.textContent     = 'DISCONNECTED';
});

socket.on('menu_toggle', ({ open }) => {
    cards.classList.toggle('active', open);
    menuStateLabel.textContent = open ? '● ACTIVE' : '● INACTIVE';
    menuStateLabel.style.color = open ? '#00c853' : '#444';
    pinchHint.style.opacity    = open ? '0' : '1';

    document.querySelectorAll('.card').forEach(c => {
        c.classList.toggle('menu-open', open);
    });
});

socket.on('gesture_action', ({ gesture, key, label, url }) => {
    const card = document.getElementById(`card-${gesture}`);
    if (card) {
        card.classList.add('firing');
        setTimeout(() => card.classList.remove('firing'), 900);
    }

    showToast(key, label);

    if (url === '__chat__') {
        window.dispatchEvent(new CustomEvent('gesture-open-chat'));
    } else {
        window.open(url, '_blank');
    }
});

function showToast(key, label) {
    toastKey.textContent   = key;
    toastLabel.textContent = label;
    toast.classList.add('visible');
    clearTimeout(toast._hideTimer);
    toast._hideTimer = setTimeout(() => toast.classList.remove('visible'), 2200);
}


(() => {
    const API = '';

    const trigger      = document.getElementById('chat-trigger');
    const popup        = document.getElementById('chat-popup');
    const badge        = document.getElementById('chat-badge');
    const msgArea      = document.getElementById('cp-messages');
    const emptyEl      = document.getElementById('cp-empty');
    const input        = document.getElementById('cp-input');
    const sendBtn      = document.getElementById('cp-send-btn');
    const newBtn       = document.getElementById('cp-new-btn');


    let isOpen    = false;
    let chatId    = null;
    let streaming = false;
    let unread    = 0;

    trigger.addEventListener('click', () => {
        isOpen = !isOpen;
        trigger.classList.toggle('open', isOpen);
        popup.classList.toggle('open', isOpen);
        if (isOpen) {
            clearBadge();
            if (!chatId) startNewChat();
            setTimeout(() => input.focus(), 250);
        }
    });


    async function loadChat(id) {
        try {
            const res  = await fetch(`${API}/api/chats/${id}`);
            const data = await res.json();
            chatId = data.id;
            clearMessages();
            (data.messages || []).forEach(m => appendMsg(m.role, m.content));
            setTimeout(() => input.focus(), 100);
        } catch {
            appendError('⚠ Could not load chat');
        }
    }

    function addUnread() {
        if (isOpen) return;
        unread++;
        badge.textContent = unread;
        badge.classList.add('visible');
    }
    function clearBadge() {
        unread = 0;
        badge.classList.remove('visible');
    }

    async function startNewChat() {
        chatId = null;
        clearMessages();
        try {
            const res  = await fetch(`${API}/api/chats/new`, { method: 'POST' });
            const data = await res.json();
            chatId = data.id;
        } catch {
            appendError('⚠ Could not reach /api/chats/new');
        }
    }
    newBtn.addEventListener('click', startNewChat);

    function clearMessages() {
        msgArea.innerHTML = '';
        msgArea.appendChild(emptyEl);
        emptyEl.style.display = 'flex';
    }
    function hideEmpty() { emptyEl.style.display = 'none'; }

    function appendMsg(role, text) {
        hideEmpty();
        const wrap = document.createElement('div');
        wrap.className = `cp-msg ${role}`;
        const lbl = document.createElement('div');
        lbl.className = 'cp-role';
        lbl.textContent = role === 'user' ? 'YOU' : 'BOT';
        const bubble = document.createElement('div');
        bubble.className = 'cp-bubble';
        bubble.textContent = text;
        wrap.appendChild(lbl);
        wrap.appendChild(bubble);
        msgArea.appendChild(wrap);
        scrollBottom();
        return bubble;
    }

    function appendTyping() {
        hideEmpty();
        const wrap = document.createElement('div');
        wrap.className = 'cp-msg assistant';
        wrap.id = 'cp-typing-wrap';
        const t = document.createElement('div');
        t.className = 'cp-typing';
        t.innerHTML = '<span></span><span></span><span></span>';
        wrap.appendChild(t);
        msgArea.appendChild(wrap);
        scrollBottom();
    }
    function removeTyping() { document.getElementById('cp-typing-wrap')?.remove(); }

    function appendError(msg) {
        hideEmpty();
        const el = document.createElement('div');
        el.className = 'cp-error';
        el.textContent = msg;
        msgArea.appendChild(el);
        scrollBottom();
    }
    function scrollBottom() { msgArea.scrollTop = msgArea.scrollHeight; }

    async function sendMessage() {
        if (streaming || !chatId) return;
        const text = input.value.trim();
        if (!text) return;

        input.value = '';
        input.style.height = 'auto';
        setSending(true);
        appendMsg('user', text);
        appendTyping();

        try {
            const res = await fetch(`${API}/api/chats/${chatId}/send`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            removeTyping();
            const bubble = appendMsg('assistant', '');
            let fullText = '';
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const p = JSON.parse(line.slice(6));
                        if (p.token) { fullText += p.token; bubble.textContent = fullText; scrollBottom(); }
                        if (p.done)  addUnread();
                        if (p.error) appendError('⚠ ' + p.error);
                    } catch {}
                }
            }
        } catch {
            removeTyping();
            appendError('⚠ Send failed — is app.py running?');
        }

        setSending(false);
        input.focus();
    }

    function setSending(val) {
        streaming = val;
        sendBtn.disabled = val;
        input.disabled = val;
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 80) + 'px';
    });

    window.addEventListener('gesture-open-chat', () => {
        if (!isOpen) {
            isOpen = true;
            trigger.classList.add('open');
            popup.classList.add('open');
            clearBadge();
            if (!chatId) startNewChat();
            setTimeout(() => input.focus(), 250);
        }
    });
})();