(function () {

    // Target confirmed via DevTools: div.bubble-wrap[role="log"] inside #chatbot
    // scrollHeight > clientHeight = real scroll container.
    // autoscroll=False must be set in app.py — Gradio's own autoscroll
    // resets scrollTop after every update and overwrites our JS.

    function getContainer() {
        return document.querySelector('#chatbot .bubble-wrap');
    }

    function scrollToBottom() {
        const el = getContainer();
        if (el) el.scrollTop = el.scrollHeight;
    }

    // Keep scrolling every 100ms while content is still growing (streaming).
    // Stops 1.5s after scrollHeight stabilises.
    function pinUntilStable() {
        const el = getContainer();
        if (!el) return;

        scrollToBottom();

        let last = el.scrollHeight;
        let stable = 0;

        const id = setInterval(() => {
            const el = getContainer();
            if (!el) { clearInterval(id); return; }

            scrollToBottom();

            if (el.scrollHeight === last) {
                if (++stable >= 15) clearInterval(id);  // 15 × 100ms = 1.5s stable
            } else {
                last   = el.scrollHeight;
                stable = 0;
            }
        }, 100);
    }

    function hookTriggers() {
        // Enter key in textbox
        const textarea = document.querySelector('#question-box textarea');
        if (textarea && !textarea._sh) {
            textarea._sh = true;
            textarea.addEventListener('keydown', e => {
                if (e.key === 'Enter' && !e.shiftKey) pinUntilStable();
            });
        }

        // Followup buttons — Gradio replaces these after each response
        document.querySelectorAll('#followups button').forEach(btn => {
            if (!btn._sh) {
                btn._sh = true;
                btn.addEventListener('click', pinUntilStable);
            }
        });
    }

    function attach() {
        const chatbot = document.querySelector('#chatbot');
        if (!chatbot) { setTimeout(attach, 300); return; }

        // Watch for new message rows — fires when user sends or answer arrives
        let prevRows = 0;
        new MutationObserver(() => {
            const rows = chatbot.querySelectorAll('.message-row').length;
            if (rows !== prevRows) {
                prevRows = rows;
                pinUntilStable();
            }
            hookTriggers();   // re-hook after Gradio replaces buttons
        }).observe(chatbot, { childList: true, subtree: true });

        hookTriggers();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attach);
    } else {
        attach();
    }
})();

// ── Header carousel ───────────────────────────────────────────────────────────
// Script lives here (not in header.html) because gr.HTML sanitises <script> tags.
// Waits for the carousel DOM to be ready before initialising.

(function () {
    function initCarousel() {
        var slides = document.getElementById('arc-slides');
        var dotsEl = document.getElementById('arc-dots');
        var prev   = document.getElementById('arc-prev');
        var next   = document.getElementById('arc-next');
        if (!slides || !dotsEl || !prev || !next) {
            setTimeout(initCarousel, 200);
            return;
        }

        var total = slides.children.length;
        var cur   = 0;
        var timer;

        // Build dots
        for (var i = 0; i < total; i++) {
            var d = document.createElement('button');
            d.className = i === 0 ? 'active' : '';
            d.onclick = (function (n) { return function () { go(n); }; })(i);
            dotsEl.appendChild(d);
        }

        function go(n) {
            cur = (n + total) % total;
            slides.style.transform = 'translateX(-' + cur * 100 + '%)';
            dotsEl.querySelectorAll('button').forEach(function (d, i) {
                if (i === cur) {
                    d.classList.add('active');
                } else {
                    d.classList.remove('active');
                }
            });
            resetTimer();
        }

        function resetTimer() {
            clearInterval(timer);
            timer = setInterval(function () { go(cur + 1); }, 4500);
        }

        prev.onclick = function () { go(cur - 1); };
        next.onclick = function () { go(cur + 1); };
        resetTimer();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCarousel);
    } else {
        initCarousel();
    }
})();

// ── Character counter ─────────────────────────────────────────────────────────
// Display remaining characters next to the "Question" label.

(function () {
    const MAX_LENGTH = 150; // Must match max_length in app.py
    
    function initCharCounter() {
        const container = document.querySelector('#question-box');
        if (!container) return;
        
        const textarea = container.querySelector('textarea');
        if (!textarea) return;
        
        // Find existing label or look for the element that contains "Question"
        let labelContainer = container.querySelector('[data-testid="block-info"]');
        
        // If not found, try to find the label by looking for text nodes
        if (!labelContainer) {
            // Try finding the label in the container's structure
            const labels = container.querySelectorAll('label');
            if (labels.length > 0) {
                labelContainer = labels[0].parentElement;
            }
        }
        
        // If still not found, just create a container above the textarea
        if (!labelContainer) {
            labelContainer = document.createElement('div');
            labelContainer.className = 'label-container';
            
            const label = document.createElement('span');
            label.textContent = 'Question';
            labelContainer.appendChild(label);
            
            textarea.parentElement.insertBefore(labelContainer, textarea);
        }
        
        // Create or get counter element
        let counter = labelContainer.querySelector('.char-counter');
        if (!counter) {
            counter = document.createElement('div');
            counter.className = 'char-counter';
            labelContainer.appendChild(counter);
        }
        
        // Enforce maxlength at browser level too
        textarea.maxLength = MAX_LENGTH;

        function updateCount() {
            if (textarea.value.length > MAX_LENGTH) {
                textarea.value = textarea.value.slice(0, MAX_LENGTH);
            }
            const len = textarea.value.length;
            const remaining = MAX_LENGTH - len;
            counter.textContent = `${len} / ${MAX_LENGTH}`;
            
            // Update classes for styling
            counter.classList.remove('warning', 'critical');
            if (remaining < 50) {
                counter.classList.add('critical');
            } else if (remaining < 100) {
                counter.classList.add('warning');
            }
        }
        
        if (!textarea._counterSetup) {
            textarea._counterSetup = true;
            textarea.addEventListener('input', updateCount);
            textarea.addEventListener('change', updateCount);
            textarea.addEventListener('keyup', updateCount);

            // Keep in sync when Gradio or app clears content programmatically
            textarea._counterInterval = setInterval(updateCount, 200);
            updateCount();  // initial
        }
    }
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCharCounter);
    } else {
        initCharCounter();
    }
    
    // Retry periodically in case Gradio recreates the element
    setInterval(initCharCounter, 500);
})();