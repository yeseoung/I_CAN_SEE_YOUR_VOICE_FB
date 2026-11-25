// SORI 화상회의 화면 (React, 빌드 없이 htm으로 작성). 서버 API·WebSocket 형식은 예전 app.html과 같다.
import { createElement, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import htm from "htm";
import { Hangul, selftest } from "/static/hangul.js";

const html = htm.bind(createElement);
if (location.search.includes("selftest")) selftest();

// ---------- 공통 ----------
const SPACE_MS = 1500;  // 손을 이만큼 내리고 있으면 띄어쓰기
const RTC_CONFIG = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
const actionsRe = /^[ㄱ-ㅎㅏ-ㅣ]$/;  // 자모 한 글자만 자막에 추가
const MODELS = [["v1", "v1 (기존)"], ["v2", "v2 (11/17)"], ["v2.1", "v2.1 (11/19)"]];
const wsBase = () => `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}`;

const prefs = {
    get signDefault() { return localStorage.getItem("pref.signDefault") !== "0"; },
    set signDefault(on) { localStorage.setItem("pref.signDefault", on ? "1" : "0"); },
    get model() { const m = localStorage.getItem("pref.signModel"); return ["v2", "v2.1"].includes(m) ? m : "v1"; },
    set model(m) { localStorage.setItem("pref.signModel", m); },
};

// 손 검출은 브라우저에서 하고 서버에는 랜드마크만 보낸다
let tracker = null;
import("/static/hands.js")
    .then(m => m.createHandTracker())
    .then(t => { tracker = t; })
    .catch(error => console.error("손 검출기 로드 실패:", error));

// 탭마다 메모리에만 둔다: 탭 두 개로 다른 계정 로그인 가능
let authToken = null;
let onExpired = () => {};

// 세션 토큰을 붙여 API 호출. 401이면 세션이 끝난 것(서버 재시작 등)이라 로그인 화면으로 보낸다
async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${authToken}` } });
    if (res.status === 401) {
        onExpired("세션이 만료되었습니다. 다시 로그인하세요.");
        throw new Error("unauthorized");
    }
    return res;
}

const initial = (name) => (name?.[0] || "U").toUpperCase();

// ---------- 로그인 ----------
function Login({ message, onLogin }) {
    const [username, setUsername] = useState("demo");
    const [password, setPassword] = useState("demo");
    const [msg, setMsg] = useState({ text: message || "", info: false });
    const [busy, setBusy] = useState(false);

    useEffect(() => setMsg({ text: message || "", info: false }), [message]);

    const submit = async (e) => {
        e.preventDefault();
        setMsg({ text: "", info: false });
        setBusy(true);
        try {
            const res = await fetch("/api/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: username.trim(), password }),
            });
            const data = await res.json();
            if (!res.ok || !data.ok) { setMsg({ text: data.error || "로그인에 실패했습니다.", info: false }); return; }
            authToken = data.token;
            onLogin({ name: data.name, role: data.role });
        } catch (err) {
            setMsg({ text: "서버에 연결할 수 없습니다.", info: false });
        } finally {
            setBusy(false);
        }
    };

    // 소셜 로그인은 UI만 — 기능은 추후 연결
    const social = (provider) =>
        setMsg({ text: `${provider} 로그인은 준비 중이에요. 체험 계정으로 로그인해 주세요.`, info: true });

    return html`
    <main class="login">
        <section class="login-hero">
            <div class="wordmark">SORI</div>
            <div class="hero-body">
                <h1>손으로 말하면<br />자막으로 전해집니다</h1>
                <p>지문자를 실시간으로 한글 자막으로 바꿔 주는 1:1 화상회의입니다.</p>
                <div class="demo-tile" aria-hidden="true">
                    <span class="demo-name">상대방</span>
                    <p class="caption demo-caption">수어<span class="demo-swap"><span class="composing f1">ㄹ</span><span class="composing f2">로</span></span></p>
                </div>
            </div>
            <div class="hero-foot">손 모양 인식은 내 브라우저에서, 글자 판별은 서버에서 합니다.</div>
        </section>

        <section class="login-panel">
            <div class="login-box">
                <h2>로그인</h2>
                <p class="lead">회의를 열거나 코드로 참여하려면 로그인하세요.</p>
                <form onSubmit=${submit}>
                    <label class="field">
                        <span>아이디</span>
                        <input value=${username} onInput=${e => setUsername(e.target.value)} autoComplete="username" />
                    </label>
                    <label class="field">
                        <span>비밀번호</span>
                        <input type="password" value=${password} onInput=${e => setPassword(e.target.value)} autoComplete="current-password" />
                    </label>
                    <button class="btn btn-primary btn-block" type="submit" disabled=${busy}>${busy ? "로그인 중…" : "로그인"}</button>
                </form>
                <p class=${"msg" + (msg.info ? " info" : "")} role="status">${msg.text}</p>
                <div class="sep">또는</div>
                <div class="social">
                    <button type="button" class="btn btn-line btn-block" onClick=${() => social("Google")}>
                        <svg viewBox="0 0 48 48" aria-hidden="true"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/></svg>
                        Google로 계속하기
                    </button>
                    <button type="button" class="btn btn-apple btn-block" onClick=${() => social("Apple")}>
                        <svg viewBox="0 0 384 512" fill="currentColor" aria-hidden="true"><path d="M318.7 268.7c-.2-36.7 16.4-64.4 50-84.8-18.8-26.9-47.2-41.7-84.7-44.6-35.5-2.8-74.3 20.7-88.5 20.7-15 0-49.4-19.7-76.4-19.7C63.3 141.2 4 184.8 4 273.5q0 39.3 14.4 81.2c12.8 36.7 59 126.7 107.2 125.2 25.2-.6 43-17.9 75.8-17.9 31.8 0 48.3 17.9 76.4 17.9 48.6-.7 90.4-82.5 102.6-119.3-65.2-30.7-61.7-90-61.7-91.9zm-56.6-164.2c27.3-32.4 24.8-61.9 24-72.5-24.1 1.4-52 16.4-67.9 34.9-17.5 19.8-27.8 44.3-25.6 71.9 26.1 2 49.9-11.4 69.5-34.3z"/></svg>
                        Apple로 계속하기
                    </button>
                </div>
                <p class="hint">체험 계정: demo / demo, alice / 1234, bob / 1234</p>
            </div>
        </section>
    </main>`;
}

// ---------- 로그인 이후 공통 헤더 ----------
function Header({ user, screen, go, onLogout }) {
    const nav = (id, label) => html`
        <button class=${"navlink" + (screen === id ? " active" : "")} aria-current=${screen === id ? "page" : null} onClick=${() => go(id)}>${label}</button>`;
    return html`
    <header class="appbar">
        <div class="appbar-inner">
            <button class="wordmark wordmark-btn" onClick=${() => go("lobby")}>SORI</button>
            <nav class="topnav">${nav("lobby", "홈")}${nav("mypage", "마이페이지")}</nav>
            <div class="userchip">
                <span class="avatar" aria-hidden="true">${initial(user.name)}</span>
                <span class="uname">${user.name}</span>
                <button class="link-btn" onClick=${onLogout}>로그아웃</button>
            </div>
        </div>
    </header>`;
}

// ---------- 홈 ----------
function Lobby({ user, onEnter }) {
    const [code, setCode] = useState("");
    const [msg, setMsg] = useState("");
    const [meetings, setMeetings] = useState(null);  // null: 불러오는 중, "error": 실패

    const load = async () => {
        setMeetings(null);
        try {
            const res = await api("/api/meetings");
            setMeetings((await res.json()).meetings || []);
        } catch (e) {
            setMeetings("error");
        }
    };
    useEffect(() => { load(); }, []);

    const enter = async (c) => { const err = await onEnter(c); if (err) setMsg(err); };

    const create = async () => {
        setMsg("");
        try {
            const res = await api("/api/meetings", { method: "POST" });
            enter((await res.json()).code);
        } catch (err) {
            setMsg("회의를 개설할 수 없습니다.");
        }
    };

    const join = async (e) => {
        e.preventDefault();
        const c = code.trim();
        setMsg("");
        if (!c) { setMsg("회의 코드를 입력하세요."); return; }
        try {
            const res = await api(`/api/meetings/${encodeURIComponent(c)}`);
            if (!res.ok) { setMsg("존재하지 않는 회의 코드입니다."); return; }
            enter(c);
        } catch (err) {
            setMsg("서버에 연결할 수 없습니다.");
        }
    };

    let list;
    if (meetings === null) list = html`<p class="ml-empty">불러오는 중…</p>`;
    else if (meetings === "error") list = html`<p class="ml-empty">목록을 불러오지 못했습니다. 새로고침을 눌러 다시 시도하세요.</p>`;
    else if (!meetings.length) list = html`<p class="ml-empty">아직 열린 회의가 없어요. 새 회의를 시작해 보세요.</p>`;
    else list = html`<ul class="ml">${meetings.map(m => html`
        <li class="ml-row" key=${m.code}>
            <span class=${"ml-live" + (m.live ? " on" : "")}>${m.live ? "진행 중" : "종료"}</span>
            <span class="ml-code">${m.code}</span>
            <span class="ml-host">개설자 ${m.host}</span>
            <span class="ml-count">${m.participants}명 참여</span>
            <button class="btn btn-line btn-sm" onClick=${() => enter(m.code)}>참여</button>
        </li>`)}</ul>`;

    return html`
    <main class="page">
        <h1 class="greet">${user.name} 님, 안녕하세요</h1>
        <p class="greet-sub">새 회의를 열거나, 받은 코드로 참여하세요.</p>
        <div class="paths">
            <section class="path path-new">
                <h2>새 회의 시작</h2>
                <p>방을 만들고 코드를 공유하면 바로 통화가 열립니다.</p>
                <button class="btn btn-caption btn-block" onClick=${create}>새 회의 시작</button>
            </section>
            <section class="path">
                <h2>코드로 참여</h2>
                <p>받은 회의 코드를 입력하세요.</p>
                <form class="join-row" onSubmit=${join}>
                    <input class="code-input" aria-label="회의 코드" value=${code} onInput=${e => setCode(e.target.value)}
                        placeholder="a1b2c3" autoCapitalize="off" spellCheck="false" />
                    <button class="btn btn-primary" type="submit">참여</button>
                </form>
            </section>
        </div>
        <p class="msg" role="alert">${msg}</p>
        <div class="section-head">
            <h2>회의 목록</h2>
            <button class="link-btn" onClick=${load}>새로고침</button>
        </div>
        ${list}
    </main>`;
}

// ---------- 마이페이지 ----------
function MyPage({ user }) {
    const [signDefault, setSignDefault] = useState(prefs.signDefault);
    const toggle = (e) => { prefs.signDefault = e.target.checked; setSignDefault(e.target.checked); };
    const kind = user.role === "admin" ? "관리자 계정" : "체험 계정";
    return html`
    <main class="page page-narrow">
        <h1 class="page-title">마이페이지</h1>
        <section class="profile">
            <span class="avatar avatar-lg" aria-hidden="true">${initial(user.name)}</span>
            <div>
                <div class="profile-name">${user.name}</div>
                <div class="profile-meta">@${user.name}</div>
            </div>
        </section>
        <dl class="info">
            <div><dt>아이디</dt><dd>${user.name}</dd></div>
            <div><dt>계정 유형</dt><dd>${kind}</dd></div>
            <div><dt>가입일</dt><dd>2026. 11. 5.</dd></div>
        </dl>
        <h2 class="sub-title">환경 설정</h2>
        <label class="setting">
            <span>회의에 들어가면 수어 자막 켜기</span>
            <input type="checkbox" class="switch" checked=${signDefault} onChange=${toggle} />
        </label>
    </main>`;
}

// ---------- 회의실 ----------
function Tile({ name, stream, local, caption, videoRef }) {
    const setVideo = (el) => {
        if (videoRef) videoRef.current = el;
        if (!el) return;
        el.muted = !!local;  // 내 소리는 내 스피커로 다시 내지 않는다
        if (el.srcObject !== stream) el.srcObject = stream;
    };
    return html`
    <figure class=${"tile" + (local ? " local" : "")}>
        ${stream ? html`<video ref=${setVideo} autoPlay playsInline></video>`
                 : html`<div class="tile-wait">영상 연결 중…</div>`}
        <figcaption class="tile-name">${name}</figcaption>
        ${caption}
    </figure>`;
}

function Caption({ text, composing, live }) {
    if (!text && !composing) return null;
    return html`<p class="caption" aria-live=${live ? "polite" : null}>${text}${composing && html`<span class="composing">${composing}</span>`}</p>`;
}

function Room({ code, stream, user, onLeave }) {
    const [peers, setPeers] = useState([]);          // [{id, name, stream}]
    const [captions, setCaptions] = useState({});    // peerId -> 자막
    const [own, setOwn] = useState(["", ""]);        // 내 자막 [확정, 조합 중]
    const [signOn, setSignOn] = useState(prefs.signDefault);
    const [model, setModel] = useState(prefs.model);
    const [copied, setCopied] = useState(false);
    const localVideo = useRef(null);
    const roomSocket = useRef(null);
    const composer = useRef(null);
    if (!composer.current) composer.current = new Hangul();
    const hand = useRef({ lastSeen: 0, spaced: false });

    // 내 자막은 내 영상 아래에 보이고, 다른 참가자에게 전송된다
    const publishCaption = () => {
        const c = composer.current;
        setOwn(c.parts());
        const ws = roomSocket.current;
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "caption", text: c.text() }));
    };

    // 시그널링 + WebRTC
    useEffect(() => {
        const pcs = new Map();      // peerId -> RTCPeerConnection
        const names = new Map();    // peerId -> 이름
        let signalQueue = Promise.resolve();
        const ws = new WebSocket(`${wsBase()}/ws/room/${encodeURIComponent(code)}?token=${encodeURIComponent(authToken)}`);
        roomSocket.current = ws;

        // 상대 타일은 입장 알림을 받자마자 만든다. 영상(WebRTC) 연결이 늦거나 실패해도 상대 자막은 보여야 한다
        const upsert = (id, s) => setPeers(ps => {
            const i = ps.findIndex(p => p.id === id);
            if (i < 0) return [...ps, { id, name: names.get(id) || "참가자", stream: s || null }];
            if (!s) return ps;
            const next = [...ps]; next[i] = { ...next[i], stream: s }; return next;
        });
        const sendSignal = (to, data) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "signal", to, data }));
        };
        const createPeer = (peerId, isInitiator) => {
            const pc = new RTCPeerConnection(RTC_CONFIG);
            pcs.set(peerId, pc);
            stream.getTracks().forEach(track => pc.addTrack(track, stream));
            pc.onicecandidate = (e) => { if (e.candidate) sendSignal(peerId, { candidate: e.candidate }); };
            pc.ontrack = (e) => upsert(peerId, e.streams[0]);
            if (isInitiator) {
                pc.createOffer()
                    .then(offer => pc.setLocalDescription(offer))
                    .then(() => sendSignal(peerId, { sdp: pc.localDescription }));
            }
            return pc;
        };
        const handleSignal = async (from, data) => {
            const pc = pcs.get(from) || createPeer(from, false);
            if (data.sdp) {
                await pc.setRemoteDescription(data.sdp);
                if (data.sdp.type === "offer") {
                    await pc.setLocalDescription(await pc.createAnswer());
                    sendSignal(from, { sdp: pc.localDescription });
                }
            } else if (data.candidate) {
                try { await pc.addIceCandidate(data.candidate); } catch (e) { console.warn("ICE 후보 추가 실패", e); }
            }
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "welcome") {
                // 기존 참가자들에게 내가 먼저 연결을 건다(offer 발신)
                for (const p of msg.peers) {
                    names.set(p.peerId, p.name);
                    upsert(p.peerId);
                    createPeer(p.peerId, true);
                }
            } else if (msg.type === "peer-joined") {
                // 새 참가자는 자기가 offer를 보내 온다. 여기서는 이름만 기억.
                names.set(msg.peerId, msg.name);
                upsert(msg.peerId);
            } else if (msg.type === "signal") {
                // 시그널은 도착 순서대로 하나씩 처리한다. 동시에 처리하면 offer 적용이 끝나기 전에
                // 도착한 ICE 후보가 버려져 다른 네트워크끼리 연결이 안 된다
                signalQueue = signalQueue.then(() => handleSignal(msg.from, msg.data)).catch(console.error);
            } else if (msg.type === "caption") {
                setCaptions(c => ({ ...c, [msg.from]: msg.text }));
            } else if (msg.type === "peer-left") {
                pcs.get(msg.peerId)?.close();
                pcs.delete(msg.peerId);
                names.delete(msg.peerId);
                setPeers(ps => ps.filter(p => p.id !== msg.peerId));
                setCaptions(c => { const { [msg.peerId]: _, ...rest } = c; return rest; });
            }
        };

        const closeWs = () => ws.close();
        window.addEventListener("beforeunload", closeWs);
        return () => {
            window.removeEventListener("beforeunload", closeWs);
            pcs.forEach(pc => pc.close());
            ws.onmessage = null;
            ws.close();
            roomSocket.current = null;
            stream.getTracks().forEach(t => t.stop());
        };
    }, []);

    // 수어 인식 (/ws/sign): 켜져 있는 동안 연결하고, 모델을 바꾸면 새 모델로 다시 연결한다
    useEffect(() => {
        if (!signOn) return;
        let ws = null, timer = null, retry = null, busy = false, done = false;
        const stop = () => {
            clearInterval(timer); timer = null;
            if (ws) { ws.onclose = null; ws.close(); ws = null; }
            busy = false;
        };
        const start = () => {
            ws = new WebSocket(`${wsBase()}/ws/sign?model=${model}`);
            ws.onmessage = (event) => {
                busy = false;
                const data = JSON.parse(event.data);
                const now = performance.now();
                const h = hand.current;
                if (!data.detected) {
                    // 손 내림: 오래 내리고 있으면 띄어쓰기 한 번
                    if (!h.spaced && composer.current.text() && (now - h.lastSeen) > SPACE_MS) {
                        composer.current.space(); publishCaption(); h.spaced = true;
                    }
                    return;
                }
                h.lastSeen = now; h.spaced = false;
                // 서버가 새로 확정한 글자만 넣는다 (같은 글자 3번 연속, 로컬 웹캠과 같은 규칙)
                if (data.new && actionsRe.test(data.new)) {
                    composer.current.input(data.new);
                    publishCaption();
                }
            };
            // 서버 오류나 네트워크 문제로 끊기면 1초 뒤 다시 연결한다
            ws.onclose = () => { stop(); if (!done) retry = setTimeout(start, 1000); };
            timer = setInterval(() => {
                const video = localVideo.current;
                if (!video || video.readyState < 2 || !ws || ws.readyState !== WebSocket.OPEN || busy || !tracker) return;
                busy = true;
                try { ws.send(JSON.stringify(tracker.detect(video))); } catch (e) { busy = false; }
            }, 33);  // 학습 데이터와 같은 30fps: 10프레임 창 = 0.33초
        };
        start();
        return () => { done = true; clearTimeout(retry); stop(); };
    }, [signOn, model]);

    const toggleSign = () => {
        if (signOn) { composer.current.reset(); publishCaption(); }
        setSignOn(!signOn);
    };
    const clearCaption = () => { composer.current.reset(); publishCaption(); };
    const changeModel = (e) => { prefs.model = e.target.value; setModel(e.target.value); };
    const copyCode = async () => {
        try { await navigator.clipboard.writeText(code); } catch (e) { return; }
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
    };

    const solo = peers.length === 0;
    return html`
    <main class="room">
        <header class="room-top">
            <span class="conn"><span class="dot"></span>연결됨</span>
            <span class="codechip">
                <span class="codechip-label">회의 코드</span>
                <span class="codechip-code">${code}</span>
                <button class=${"copy" + (copied ? " copied" : "")} onClick=${copyCode} aria-label="회의 코드 복사" title="코드 복사">
                    ${copied ? "복사됨" : "복사"}
                </button>
            </span>
        </header>

        <section class=${"stage" + (solo ? " solo" : "")}>
            ${peers.map(p => html`
                <${Tile} key=${p.id} name=${p.name} stream=${p.stream}
                    caption=${html`<${Caption} text=${captions[p.id] || ""} live />`} />`)}
            <${Tile} key="local" local name=${`${user.name} (나)`} stream=${stream} videoRef=${localVideo}
                caption=${html`<${Caption} text=${own[0]} composing=${own[1]} />`} />
            ${solo && html`<p class="waiting">상대가 들어오기를 기다리는 중이에요. 회의 코드 <b>${code}</b>를 공유하세요.</p>`}
        </section>

        <footer class="dock">
            <button class=${"pill" + (signOn ? " on" : "")} aria-pressed=${signOn} onClick=${toggleSign}>
                <span class="pill-dot"></span>수어 자막 ${signOn ? "켜짐" : "꺼짐"}
            </button>
            <label class="pill pill-select">
                <span>모델</span>
                <select value=${model} onChange=${changeModel} aria-label="수어 인식 모델">
                    ${MODELS.map(([v, label]) => html`<option key=${v} value=${v}>${label}</option>`)}
                </select>
            </label>
            <button class="pill" onClick=${clearCaption}>자막 지우기</button>
            <button class="pill pill-leave" onClick=${onLeave}>나가기</button>
        </footer>
    </main>`;
}

// ---------- 앱 ----------
function App() {
    const [user, setUser] = useState(null);          // {name, role}
    const [screen, setScreen] = useState("lobby");   // lobby | mypage
    const [room, setRoom] = useState(null);          // {code, stream}
    const [loginMsg, setLoginMsg] = useState("");

    const endSession = (message = "") => {
        authToken = null;
        setUser(null); setRoom(null); setLoginMsg(message);
    };
    onExpired = endSession;

    // 카메라를 먼저 연다. 실패하면 원인을 담은 문구를 돌려줘 홈 화면에 보여 준다
    const enterRoom = async (code) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            setRoom({ code, stream });
        } catch (err) {
            // HTTPS가 아니면 브라우저가 카메라 API 자체를 막는다(localhost·127.0.0.1은 예외)
            return !window.isSecureContext
                ? "이 주소에서는 카메라를 쓸 수 없습니다. https:// 또는 http://localhost:8000 으로 접속하세요."
                : err.name === "NotAllowedError" ? "카메라/마이크 권한이 필요합니다. 주소창의 카메라 아이콘에서 허용하세요."
                : `카메라/마이크를 열 수 없습니다 (${err.name}). 다른 앱이 카메라를 쓰고 있는지 확인하세요.`;
        }
    };

    const logout = () => {
        fetch("/api/logout", { method: "POST", headers: { Authorization: `Bearer ${authToken}` } }).catch(() => {});
        endSession();
    };

    if (!user) return html`<${Login} message=${loginMsg} onLogin=${u => { setUser(u); setScreen("lobby"); setLoginMsg(""); }} />`;
    if (room) return html`<${Room} key=${room.code} code=${room.code} stream=${room.stream} user=${user}
        onLeave=${() => { setRoom(null); setScreen("lobby"); }} />`;
    return html`
        <${Header} user=${user} screen=${screen} go=${setScreen} onLogout=${logout} />
        ${screen === "mypage" ? html`<${MyPage} user=${user} />` : html`<${Lobby} user=${user} onEnter=${enterRoom} />`}`;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
