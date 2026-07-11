# FlashDraft

**A local-first AI creative director.** Gemma 4 runs entirely on-device as the "brain," deciding when to hand off to cloud-based generation — NB2 Lite for instant images, Omni Flash for conversational video. Built for the Google DeepMind Bangalore Hackathon.

---

## The problem

AI video production tools today force a bad trade-off:

- **Every iteration costs money.** Fix a typo, tweak a prompt — you're burning cloud API credits even during brainstorming.
- **They break the moment your connection does.** No offline recovery, no resume — you lose your session.
- **Your creative IP leaves your machine.** Scripts, plot twists, unreleased campaigns — all sent to a third-party server just to draft an idea.

FlashDraft fixes this by keeping the thinking local and only paying for the cloud when it counts.

## Who it's for

- **Filmmakers & directors** blocking out scenes before committing to expensive production or VFX.
- **Location scouts** working in the field with unreliable connectivity, visualizing a site under different lighting/weather.
- **Indie game devs & storyboard artists** iterating on concept sequences without per-generation cloud costs eating their budget.

## USP

**A fault-tolerant, privacy-first AI director**, not another API wrapper:

1. **Cost-free iteration, premium rendering** — brainstorming and storyboarding run locally and for free on Gemma 4. Expensive cloud calls (NB2 Lite, Omni Flash) only fire when the shot is ready.
2. **Network-resilient continuity** — if the connection drops mid-session, the local agent doesn't crash. It falls back to a local text storyboard and resumes rendering the moment connectivity returns.
3. **IP isolation** — the creative core (script, plot, character logic) never leaves the device. Only minimal, isolated visual instructions go to the cloud.

> Our USP is a hybrid-agent architecture that puts a local Gemma 4 "Director" on your device for cost-free, private brainstorming and network resilience, while leveraging NB2 Lite and Omni Flash in the cloud only to render the final video.

## Architecture

```
        ┌─────────────────────────────┐
        │   Gemma 4 E4B (on-device)   │
        │   sense → decide → act → check
        └──────────────┬──────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
   [Tool] NB2 Lite            [Tool] Omni Flash
   fast base image             conversational video
   generation (~4s)            edit + motion
          │                           │
          └─────────────┬─────────────┘
                         │
              network error caught?
                         │
                  ┌──────┴──────┐
                  │  local text  │
                  │  storyboard  │
                  │  fallback    │
                  └──────────────┘
```

**Loop:**
- **Sense** — Gemma 4 parses the user's request (text/voice) entirely offline.
- **Decide** — checks local state/history, decides which tool to call and what minimal instruction it needs.
- **Act** — dispatches to NB2 Lite (image) and/or Omni Flash (video/motion edit).
- **Check** — verifies the result; if a cloud call fails (or the network drops), catches it and falls back to a local, detailed storyboard instead of crashing, resuming the cloud render once connectivity returns.

## Tech stack

| Layer | Tool |
|---|---|
| Local orchestrator | Gemma 4 E4B, run via [LM Studio](https://lmstudio.ai) (local OpenAI-compatible server) or Ollama |
| Agent framework | Antigravity SDK (Python) |
| Fast image generation | Nano Banana 2 Lite (NB2 Lite) |
| Conversational video | Gemini Omni Flash (`gemini-omni-flash-preview`) |
| Client | Python (`openai` SDK pointed at `localhost`) |

### 4. Add cloud keys (for NB2 Lite / Omni Flash)
- Use a **billing-disabled** project for NB2 Lite (free tier, ~1,000–1,500 req/day).
- Use a **separate, billing-enabled** project for Omni Flash (`$0.10`/sec of video). Keeping these separate avoids losing your free tier on incidental calls once billing is enabled on a project.

## Built for

Google DeepMind Bangalore Hackathon 
