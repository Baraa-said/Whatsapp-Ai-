// JavaScript for WhatsApp Simulation

const messageInput = document.getElementById('message-input');
const sendIcon = document.getElementById('send-icon');
const micIcon = document.getElementById('mic-icon');
const messagesContainer = document.getElementById('messages-container');

// Sound effects
const sentParams = { type: 'sent' }; // Placeholder for sound logic

// Event Listeners
messageInput.addEventListener('input', toggleInputIcons);
messageInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') sendMessage();
});
sendIcon.addEventListener('click', sendMessage);

// Toggle between Mic and Send icons
function toggleInputIcons() {
    if (messageInput.value.trim() !== "") {
        sendIcon.style.display = 'block';
        micIcon.style.display = 'none';
    } else {
        sendIcon.style.display = 'none';
        micIcon.style.display = 'block';
    }
}

// Send Message
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;

    // Add user message to UI
    addMessage(text, 'sent');
    messageInput.value = '';
    toggleInputIcons();
    scrollToBottom();

    // Call API
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to get response');
        }

        const data = await response.json();

        // Add bot response
        addMessage(data.answer, 'received', data.sources);
        scrollToBottom();

    } catch (error) {
        console.error('Error:', error);
        addMessage(`❌ Error: ${error.message}`, 'received');
        scrollToBottom();
    }
}

// Add message to UI
function addMessage(text, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Format text with basic markdown-like support (bold)
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
    formattedText = formattedText.replace(/\n/g, '<br>');

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const tick = type === 'sent' ? '<span class="message-tick"><i class="material-icons" style="font-size: 16px;">done_all</i></span>' : '';

    contentDiv.innerHTML = `
        ${formattedText}
        <span class="message-time">${time} ${tick}</span>
    `;

    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
}

// Scroll to bottom
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Upload Files
async function uploadFiles(files) {
    if (files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    showToast('Uploading documents...');

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            showToast(`✅ ${data.message}`);
            addMessage(`Document upload complete! I processed ${data.count} files.`, 'received');
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        showToast('❌ Error uploading files');
        console.error(error);
    }
}

// Clear chat
async function clearChat() {
    if (confirm('Are you sure you want to clear the chat?')) {
        try {
            await fetch('/api/reset', { method: 'POST' });
            messagesContainer.innerHTML = '';
            // Add intro message again
            addMessage("Hello! I'm your AI Assistant. Upload documents using the paperclip icon to start chatting! 📄", 'received');
        } catch (error) {
            console.error(error);
        }
    }
}

// Toast notification
function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = "toast show";
    setTimeout(function () { toast.className = toast.className.replace("show", ""); }, 3000);
}

// Initial scroll
scrollToBottom();
