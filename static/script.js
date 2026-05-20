/**
 * OncoCheck Web Interface – JavaScript
 * Client-side chat interface logic
 */

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loadingIndicator = document.getElementById('loadingIndicator');

// ─────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────

/**
 * Format text with markdown-like features
 */
function formatMessageText(text) {
    // Escape HTML
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Convert bold **text** to <strong>
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Convert italics *text* to <em>
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Convert headers
    formatted = formatted.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*?)$/gm, '<h1>$1</h1>');

    // Convert line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    // Convert lists
    formatted = formatted.replace(/^- (.*?)$/gm, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*?<\/li>)/s, '<ul>$1</ul>');

    return formatted;
}

/**
 * Add a message to the chat
 */
function addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isUser) {
        contentDiv.textContent = text;
    } else {
        contentDiv.innerHTML = formatMessageText(text);
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show/hide loading indicator
 */
function showLoading(show) {
    if (show) {
        loadingIndicator.classList.remove('hidden');
    } else {
        loadingIndicator.classList.add('hidden');
    }
}

/**
 * Disable/enable input
 */
function setInputEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
}

// ─────────────────────────────────────────────
// API Communication
// ─────────────────────────────────────────────

/**
 * Send message to backend API
 */
async function sendMessage(message) {
    if (!message.trim()) {
        alert('Please enter a message');
        return;
    }

    // Add user message to chat
    addMessage(message, true);
    userInput.value = '';
    setInputEnabled(false);
    showLoading(true);

    try {
        const response = await fetch('http://127.0.0.1:5000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'success') {
            addMessage(data.response, false);
        } else {
            addMessage(`Error: ${data.message || 'Unknown error'}`, false);
        }

    } catch (error) {
        console.error('Error:', error);
        addMessage(`⚠️ Connection error: ${error.message}. Please try again.`, false);
    } finally {
        showLoading(false);
        setInputEnabled(true);
        userInput.focus();
    }
}

// ─────────────────────────────────────────────
// Event Listeners
// ─────────────────────────────────────────────

/**
 * Send button click
 */
sendBtn.addEventListener('click', () => {
    const message = userInput.value.trim();
    if (message) {
        sendMessage(message);
    }
});

/**
 * Enter key to send (Ctrl+Enter)
 */
userInput.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
        }
    }
});

/**
 * Auto-resize textarea
 */
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

/**
 * Focus input on load
 */
document.addEventListener('DOMContentLoaded', () => {
    userInput.focus();
    console.log('OncoCheck Chat Interface loaded ✓');
});

// ─────────────────────────────────────────────
// Example Queries (Optional Enhancement)
// ─────────────────────────────────────────────

const exampleQueries = [
    'What are the main cancer risk factors?',
    'How is cancer risk assessed?',
    'What are the symptoms of high cancer risk?',
    'What lifestyle factors influence cancer risk?',
];
