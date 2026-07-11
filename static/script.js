// Frontend Script for Fault-Tolerant Hybrid Agent

document.addEventListener("DOMContentLoaded", () => {
    // Remove intro screen after animation completes
    const introScreen = document.getElementById("intro-screen");
    if (introScreen) {
        setTimeout(() => {
            introScreen.remove();
        }, 4000);
    }

    // DOM Elements
    const ollamaStatusPill = document.getElementById("ollama-status");
    const apiStatusPill = document.getElementById("api-status");
    const modeStatusPill = document.getElementById("mode-status");
    
    const apiKeyInput = document.getElementById("api-key-input");
    const saveKeyBtn = document.getElementById("save-key-btn");
    const offlineToggle = document.getElementById("offline-toggle");
    
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const micBtn = document.getElementById("audio-mic-btn");
    
    const audioFeedback = document.getElementById("audio-feedback");
    const cancelAudioBtn = document.getElementById("cancel-audio-btn");
    
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const nb2OutputWrapper = document.getElementById("nb2-output-wrapper");
    const nb2Image = document.getElementById("nb2-image");
    const nb2StatusLabel = document.getElementById("nb2-status-label");
    const nb2Overlay = document.getElementById("nb2-overlay");

    const videoOutputWrapper = document.getElementById("video-output-wrapper");
    const videoPlayer = document.getElementById("video-player");
    const videoTurnLabel = document.getElementById("video-turn-label");
    const videoControlsInfo = document.getElementById("video-controls-info");
    const editPip = document.getElementById("edit-pip");
    const pipNb2Image = document.getElementById("pip-nb2-image");
    const pipRenderingLabel = document.getElementById("pip-rendering-label");

    const storyboardOutputWrapper = document.getElementById("storyboard-output-wrapper");
    const storyboardReason = document.getElementById("storyboard-reason");
    const storyboardGrid = document.getElementById("storyboard-grid");

    const logConsole = document.getElementById("log-console");

    const nodes = {
        sense: document.getElementById("node-sense"),
        decide: document.getElementById("node-decide"),
        nb2: document.getElementById("node-nb2"),
        act: document.getElementById("node-act"),
        check: document.getElementById("node-check"),
        recover: document.getElementById("node-recover")
    };
    
    const connectors = {
        conn1: document.getElementById("conn-1"),
        conn2: document.getElementById("conn-2"),
        conn3: document.getElementById("conn-3"),
        conn4: document.getElementById("conn-4"),
        conn5: document.getElementById("conn-5")
    };

    let savedApiKey = localStorage.getItem("gemini_api_key") || "";
    if (savedApiKey) {
        apiKeyInput.value = savedApiKey;
        updateApiStatusIndicator(true);
    }

    let currentInteractionId = null;
    let sceneContext = null;  // accumulated description for coherent NB2 edit previews
    let pollInterval = null;
    let displayedLogCount = 0;
    let pendingOfflinePrompt = null; // stores prompt queued while offline
    let lastVideoUrl = null;         // stores last successfully generated video URL
    let isOfflineQueuing = false;
    
    // ── Offline / Online detection ──────────────────────────────────────────────
    function handleOnline() {
        if (pendingOfflinePrompt && isOfflineQueuing) {
            appendLog("Network", "Connection restored! Retrying queued prompt...", "success-log");
            appendChatMessage("agent", "**Connection restored!** Sending your queued prompt now...");
            const { text, isFollowUp } = pendingOfflinePrompt;
            pendingOfflinePrompt = null;
            isOfflineQueuing = false;
            chatInput.value = text;
            sendMessage();
        }
    }
    
    function handleOffline() {
        appendLog("Network", "Connection lost! Prompt will be queued and retried automatically.", "warning-log");
    }
    
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    // Replay button
    document.getElementById("replay-btn").addEventListener("click", () => {
        videoPlayer.currentTime = 0;
        videoPlayer.play().catch(() => {});
    });

    checkSystemStatus();
    setInterval(checkSystemStatus, 10000);

    saveKeyBtn.addEventListener("click", () => {
        const key = apiKeyInput.value.trim();
        if (key) {
            localStorage.setItem("gemini_api_key", key);
            savedApiKey = key;
            appendLog("System", "Gemini API Key saved locally.", "success-log");
            updateApiStatusIndicator(true);
            saveKeyBtn.textContent = "Saved!";
            saveKeyBtn.style.background = "var(--green-glow)";
            setTimeout(() => { saveKeyBtn.textContent = "Save"; saveKeyBtn.style.background = ""; }, 1500);
        } else {
            localStorage.removeItem("gemini_api_key");
            savedApiKey = "";
            updateApiStatusIndicator(false);
            appendLog("System", "Gemini API Key cleared.", "warning-log");
        }
    });

    offlineToggle.addEventListener("change", async () => {
        const isOffline = offlineToggle.checked;
        try {
            const res = await fetch("/api/toggle-offline", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ simulate_offline: isOffline })
            });
            const data = await res.json();
            updateModeStatusIndicator(data.simulate_offline);
            appendLog("System", `Simulation mode toggled to: ${data.simulate_offline ? "OFFLINE" : "ONLINE"}`, "warning-log");
        } catch (err) { console.error(err); }
    });

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
        });
    });

    // ── Real microphone recording via Web Speech API ─────────────────────────
    let recognition = null;
    micBtn.addEventListener("click", () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            appendLog("Audio", "Speech recognition not supported in this browser.", "warning-log");
            return;
        }
        
        if (recognition) {
            recognition.stop();
            return;
        }
        
        recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.interimResults = true;
        recognition.continuous = false;
        
        audioFeedback.style.display = "flex";
        chatInput.disabled = true;
        micBtn.style.color = "var(--color-danger)";
        audioFeedback.querySelector("#audio-timer").textContent = "Listening...";
        
        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(r => r[0].transcript)
                .join("");
            audioFeedback.querySelector("#audio-timer").textContent = `"${transcript}"...`;
            chatInput.value = transcript;
        };
        
        recognition.onend = () => {
            recognition = null;
            audioFeedback.style.display = "none";
            chatInput.disabled = false;
            micBtn.style.color = "";
            if (chatInput.value.trim()) {
                appendLog("Audio Ingestion", `Transcribed: "${chatInput.value.trim()}"`, "success-log");
            }
        };
        
        recognition.onerror = (e) => {
            recognition = null;
            audioFeedback.style.display = "none";
            chatInput.disabled = false;
            micBtn.style.color = "";
            appendLog("Audio", `Microphone error: ${e.error}`, "warning-log");
        };
        
        recognition.start();
    });

    cancelAudioBtn.addEventListener("click", () => {
        if (recognition) { recognition.stop(); recognition = null; }
        audioFeedback.style.display = "none";
        chatInput.disabled = false;
        micBtn.style.color = "";
    });

    sendBtn.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    async function checkSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const status = await res.json();
            
            if (status.ollama_online) {
                ollamaStatusPill.querySelector(".status-dot").className = "status-dot success";
                ollamaStatusPill.querySelector(".status-label").textContent = "Local Gemma: Active";
            } else {
                ollamaStatusPill.querySelector(".status-dot").className = "status-dot danger";
                ollamaStatusPill.querySelector(".status-label").textContent = "Local Gemma: Offline";
            }
            
            offlineToggle.checked = status.simulate_offline;
            updateModeStatusIndicator(status.simulate_offline);
            updateApiStatusIndicator(savedApiKey || status.gemini_key_set);
        } catch (err) {}
    }

    function updateApiStatusIndicator(hasKey) {
        if (hasKey) {
            apiStatusPill.querySelector(".status-dot").className = "status-dot success";
            apiStatusPill.querySelector(".status-label").textContent = "Gemini API: Configured";
        } else {
            apiStatusPill.querySelector(".status-dot").className = "status-dot danger";
            apiStatusPill.querySelector(".status-label").textContent = "Gemini API: Unconfigured";
        }
    }

    // Mode Toggle status
    function updateModeStatusIndicator(isOffline) {
        if (isOffline) {
            modeStatusPill.querySelector(".status-dot").className = "status-dot danger";
            modeStatusPill.querySelector(".status-label").textContent = "Mode: Simulated Offline";
        } else {
            modeStatusPill.querySelector(".status-dot").className = "status-dot success";
            modeStatusPill.querySelector(".status-label").textContent = "Mode: Online";
        }
    }

    function appendLog(source, msg, logClass = "system-log") {
        const time = new Date().toLocaleTimeString();
        const line = document.createElement("div");
        line.className = `log-line ${logClass}`;
        line.innerHTML = `[${time}] <strong>${source}</strong>: ${msg}`;
        logConsole.appendChild(line);
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    function renderNewLogs(logs) {
        for (let i = displayedLogCount; i < logs.length; i++) {
            let l = logs[i];
            let logClass = "system-log";
            if (l.status === "success") logClass = "success-log";
            if (l.status === "warning") logClass = "warning-log";
            if (l.status === "error") logClass = "error-log";
            appendLog(l.step, l.message, logClass);
        }
        displayedLogCount = logs.length;
    }

    function resetPipelineUi() {
        Object.values(nodes).forEach(n => n && (n.className = "flow-node"));
        Object.values(connectors).forEach(c => c && (c.className = "flow-connector"));
    }

    async function syncPipelineNodes(logs, isComplete, source) {
        const sleep = ms => new Promise(res => setTimeout(res, ms));
        
        nodes.sense.classList.add("success");
        connectors.conn1.classList.add("active");
        nodes.decide.classList.add("success");
        connectors.conn2.classList.add("active");
        
        const hasNb2Log = logs.some(l => l.step === "NB2_Gen");
        if (hasNb2Log) {
            nodes.nb2.classList.add("success");
        }
        connectors.conn3.classList.add("active");
        
        const hasOmniLog = logs.some(l => l.step === "Omni_Gen");
        if (hasOmniLog && !isComplete) {
            nodes.act.classList.add("active"); // Animating pulse while generating async
        }
        
        if (isComplete) {
            nodes.act.classList.remove("active");
            nodes.act.classList.add("success");
            connectors.conn4.classList.add("active");
            
            nodes.check.classList.add("active");
            await sleep(500);
            nodes.check.classList.remove("active");
            
            if (source === "cloud") {
                nodes.check.classList.add("success");
            } else {
                nodes.check.classList.add("error");
                connectors.conn5.classList.add("active");
                nodes.recover.classList.add("success");
            }
        }
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        const isFollowUp = !!currentInteractionId;

        appendChatMessage("user", text);
        chatInput.value = "";
        document.querySelector("[data-tab='preview']").click();

        resetPipelineUi();
        displayedLogCount = 0;
        if (pollInterval) clearInterval(pollInterval);

        chatInput.style.borderColor = "";
        chatInput.placeholder = "Describe the video action or edit...";

        if (!isFollowUp) {
            previewPlaceholder.style.display = "flex";
            nb2OutputWrapper.style.display = "none";
            videoOutputWrapper.style.display = "none";
            videoControlsInfo.style.display = "none";
            storyboardOutputWrapper.style.display = "none";
            editPip.style.display = "none";
        } else {
            editPip.style.display = "block";
            pipNb2Image.src = "";
            pipRenderingLabel.style.display = "none";
        }

        const isOfflineSim = offlineToggle.checked;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: text,
                    api_key: savedApiKey,
                    simulate_offline: isOfflineSim,
                    interaction_id: currentInteractionId,
                    scene_context: sceneContext
                })
            });
            const data = await res.json();

            renderNewLogs(data.logs);
            previewPlaceholder.style.display = "none";

            // Sync storyboard (offline path completes synchronously)
            if (data.status === "completed") {
                editPip.style.display = "none";
                videoOutputWrapper.style.display = "none";
                videoControlsInfo.style.display = "none";
                await syncPipelineNodes(data.logs, true, data.source);
                renderCompletedJob(data, isFollowUp, text);
                return;
            }

            // Async video path
            if (isFollowUp) {
                if (data.nb2_image_url) {
                    const img = new Image();
                    img.onload = () => {
                        pipNb2Image.src = data.nb2_image_url;
                        pipRenderingLabel.style.display = "block";
                    };
                    img.src = data.nb2_image_url;
                }
            } else {
                nb2OutputWrapper.style.display = "flex";
                if (data.nb2_image_url) {
                    const img = new Image();
                    img.onload = () => { nb2Image.src = data.nb2_image_url; };
                    img.src = data.nb2_image_url;
                }
                nb2Overlay.style.display = "flex";
            }

            syncPipelineNodes(data.logs, false, null);

            // Poll at 800ms for fast response
            pollInterval = setInterval(async () => {
                try {
                    const jobRes = await fetch(`/api/job/${data.job_id}`);
                    const jobData = await jobRes.json();
                    renderNewLogs(jobData.logs);
                    if (jobData.status === "completed") {
                        clearInterval(pollInterval);
                        await syncPipelineNodes(jobData.logs, true, jobData.source);
                        renderCompletedJob(jobData, isFollowUp, text);
                    }
                } catch (e) { /* retry next tick */ }
            }, 800);

        } catch (err) {
            console.error(err);
            const isActuallyOffline = !navigator.onLine;
            
            if (isActuallyOffline) {
                // Store prompt and show previous video if available
                isOfflineQueuing = true;
                pendingOfflinePrompt = { text, isFollowUp };
                appendLog("Network", "No connection detected. Prompt queued — will retry when back online.", "warning-log");
                
                // Show small toast notification
                showToast("📶 No internet — your video will arrive when the connection is back.");
                
                // Show last generated video cleanly if there is one
                if (lastVideoUrl) {
                    previewPlaceholder.style.display = "none";
                    videoOutputWrapper.style.display = "flex";
                    videoControlsInfo.style.display = "flex";
                    videoPlayer.src = lastVideoUrl;
                    videoPlayer.load();
                    videoTurnLabel.textContent = "⚡ Offline — showing last generated video";
                } else {
                    previewPlaceholder.style.display = "flex";
                    previewPlaceholder.querySelector("h3").textContent = "Offline — Queued";
                    previewPlaceholder.querySelector("p").textContent = "Your video will arrive when the internet connection is back.";
                }
            } else {
                previewPlaceholder.style.display = "flex";
                previewPlaceholder.querySelector("h3").textContent = "Critical Error";
                previewPlaceholder.querySelector("p").textContent = err.message;
            }
        }
    }

    function renderCompletedJob(data, wasFollowUp, promptText) {
        if (data.interaction_id) currentInteractionId = data.interaction_id;

        // Update scene context to keep styles unified
        if (!sceneContext) {
            sceneContext = promptText;
        } else {
            sceneContext = sceneContext + ", then " + promptText;
        }

        chatInput.placeholder = currentInteractionId
            ? "Describe an edit... (e.g. 'Make the sky purple')"
            : "Describe the video...";
        chatInput.style.borderColor = currentInteractionId ? "var(--purple-glow)" : "";

        nb2OutputWrapper.style.display = "none";
        nb2Overlay.style.display = "none";
        editPip.style.display = "none";
        pipNb2Image.src = "";

        if (data.source === "cloud") {
            videoOutputWrapper.style.display = "flex";
            videoControlsInfo.style.display = "flex";
            videoOutputWrapper.classList.remove("fade-in");
            void videoOutputWrapper.offsetWidth;
            videoOutputWrapper.classList.add("fade-in");

            videoPlayer.src = data.video_url;
            lastVideoUrl = data.video_url; // save for offline fallback
            videoPlayer.load();
            videoPlayer.addEventListener("canplay", () => {
                videoPlayer.play().catch(() => {});
            }, { once: true });

            videoTurnLabel.textContent = wasFollowUp
                ? `Edit #${sceneContext.split(", then ").length} rendered via Gemini Omni Flash`
                : "Base video rendered via Gemini Omni Flash";

            const msg = wasFollowUp
                ? "Edit applied! New version is playing. Send another prompt to keep editing."
                : "Base video ready! Send a follow-up to do style transfers or element swaps.";
            appendChatMessage("agent", msg);

        } else {
            videoOutputWrapper.style.display = "none";
            videoControlsInfo.style.display = "none";
            storyboardOutputWrapper.style.display = "flex";
            storyboardReason.textContent = data.storyboard.fallback_reason || "Connection dropped. Storyboard fallback generated locally.";

            storyboardGrid.innerHTML = "";
            data.storyboard.scenes.forEach(scene => {
                const card = document.createElement("div");
                card.className = "storyboard-card";
                card.innerHTML = `
                    <div class="scene-num">Scene #${scene.scene_number}</div>
                    <div class="scene-title">${scene.title}</div>
                    <div class="scene-section-title">Visual Layout</div>
                    <div class="scene-detail">${scene.description}</div>
                    <div class="scene-section-title">Action</div>
                    <div class="scene-detail">${scene.action}</div>
                `;
                storyboardGrid.appendChild(card);
            });
            appendChatMessage("agent", "**Network Drop Intercepted!** Local Gemma 4 orchestrator recovered and executed the offline storyboard fallback.");
        }
    }

    function appendChatMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}`;
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
        msgDiv.innerHTML = `<div class="message-content">${formattedText}</div>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showToast(message, duration = 4000) {
        const existing = document.getElementById("flashdraft-toast");
        if (existing) existing.remove();

        const toast = document.createElement("div");
        toast.id = "flashdraft-toast";
        toast.textContent = message;
        document.body.appendChild(toast);

        // Trigger enter animation
        requestAnimationFrame(() => toast.classList.add("toast-visible"));

        setTimeout(() => {
            toast.classList.remove("toast-visible");
            setTimeout(() => toast.remove(), 400);
        }, duration);
    }
});
