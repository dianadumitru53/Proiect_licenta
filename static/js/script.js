document.getElementById("send-btn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("userInput").addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});

function addMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message-container', sender === 'user' ? 'user-container' : 'bot-container');

    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    messageElement.textContent = message;

    const iconElement = document.createElement('span');
    iconElement.classList.add('icon', sender === 'user' ? 'user-icon' : 'bot-icon');

    messageContainer.appendChild(iconElement);
    messageContainer.appendChild(messageElement);
    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    if (userInput === "") return;

    addMessage('user', userInput);

    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.style.display = 'block';

    setTimeout(() => {
        fetch('/get', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput })
        })
        .then(response => response.json())
        .then(data => {
            typingIndicator.style.display = 'none';
            addMessage('bot', data.response);
            document.getElementById("userInput").value = "";
            document.getElementById("chat-box").scrollTop = document.getElementById("chat-box").scrollHeight;
        });
    }, 1000); // Afișează indicatorul timp de 1 secundă
}

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    document.querySelector('.welcome-container').classList.toggle('dark-mode');
    document.querySelector('.chat-container').classList.toggle('dark-mode');
    document.querySelector('.settings-panel').classList.toggle('dark-mode');
}

function adjustFontSize(size) {
    const messages = document.querySelectorAll('.message');
    messages.forEach(message => {
        message.style.fontSize = size + 'px';
    });
}
