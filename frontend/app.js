/**
 * VoxTriad Screener – Frontend Application Logic
 *
 * Responsibilities:
 *  1. Initialise the Supabase JS client (with offline mock fallback)
 *  2. Handle email/password authentication (sign-in / sign-up / dev bypass)
 *  3. Control the MediaRecorder API to capture microphone audio
 *  4. Send the recorded Blob to the FastAPI /api/audio/upload endpoint
 *  5. Fetch clinical reports (Groq AI / rule-based fallback)
 *  6. Query pgvector ANN acoustic similarities
 */

'use strict';

// =============================================================================
// 1. CONFIGURATION
// =============================================================================

// Environment variables: resolved dynamically from window or backend /api/config
var SUPABASE_URL = typeof window !== 'undefined' && window.SUPABASE_URL ? window.SUPABASE_URL : '';
var SUPABASE_ANON_KEY = typeof window !== 'undefined' && window.SUPABASE_ANON_KEY ? window.SUPABASE_ANON_KEY : '';

/** Base URL of the FastAPI backend (dynamically resolves to loopback host). */
const API_BASE_URL = (typeof window !== 'undefined' && window.API_BASE_URL)
  ? window.API_BASE_URL
  : (typeof window !== 'undefined' && window.location.hostname === '127.0.0.1')
    ? 'http://127.0.0.1:8000'
    : 'http://localhost:8000';

// =============================================================================
// 2. SUPABASE CLIENT INITIALISATION (with offline mock fallback)
// =============================================================================

function createMockSupabase() {
  console.warn('[VoxTriad] Supabase is unavailable or credentials failed. Initializing offline mock client.');
  return {
    auth: {
      async getSession() {
        return { data: { session: null }, error: new Error('Supabase not loaded') };
      },
      onAuthStateChange() {
        return { data: { subscription: { unsubscribe() {} } } };
      },
      async signOut() {
        return { error: null };
      },
      async signInWithPassword() {
        return { data: { session: null }, error: new Error('Supabase not loaded') };
      },
      async signUp() {
        return { data: { session: null }, error: new Error('Supabase not loaded') };
      },
      async signInWithOAuth() {
        return { data: { session: null }, error: new Error('Supabase not loaded') };
      }
    }
  };
}

let supabaseClient = (typeof window !== 'undefined' && window.supabaseClient)
  ? window.supabaseClient
  : null;

if (!supabaseClient) {
  try {
    if (typeof window !== 'undefined' && window.supabase && typeof window.supabase.createClient === 'function' && SUPABASE_URL && SUPABASE_ANON_KEY) {
      supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: true,
        },
      });
      console.info('[VoxTriad] Supabase client successfully initialized in app.js.');
    } else {
      console.warn('[VoxTriad] Supabase credentials not found or SDK not available. Falling back to offline mock client.');
      supabaseClient = createMockSupabase();
    }
  } catch (err) {
    console.warn('[VoxTriad] Exception while initializing Supabase client:', err);
    supabaseClient = createMockSupabase();
  }
}

// Asynchronously bootstrap client from backend /api/config if not pre-configured
if (typeof window !== 'undefined' && (!SUPABASE_URL || !SUPABASE_ANON_KEY)) {
  fetch(`${API_BASE_URL}/api/config`)
    .then((res) => (res.ok ? res.json() : null))
    .then((cfg) => {
      if (cfg && cfg.supabase_url && cfg.supabase_anon_key && window.supabase && typeof window.supabase.createClient === 'function') {
        SUPABASE_URL = cfg.supabase_url;
        SUPABASE_ANON_KEY = cfg.supabase_anon_key;
        supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
          auth: {
            persistSession: true,
            autoRefreshToken: true,
            detectSessionInUrl: true,
          },
        });
        window.SUPABASE_URL = SUPABASE_URL;
        window.SUPABASE_ANON_KEY = SUPABASE_ANON_KEY;
        window.supabaseClient = supabaseClient;
        window.supabase = supabaseClient;
        console.info('[VoxTriad] Supabase client dynamically configured from backend /api/config.');
      }
    })
    .catch(() => {});
}

// =============================================================================
// 3. DOM REFERENCES (Declared at module scope; bound in initApp)
// =============================================================================

// Auth
let authSection, authForm, authEmailInput, authPasswordInput, authError;
let authCardTitle, authCardDesc, toggleAuthPrompt;
let btnLogin, btnLoginText, btnLoginSpinner, btnToggleRegister, btnBypassAuth;

// Navbar
let navUserInfo, navUserEmail, btnLogout;

// Dashboard
let dashboardSection, dashSessionId;

// Recorder controls
let btnRecord, btnStop, iconMic, iconStop, recordStatus, recordTimer, recordRing1, recordRing2;

// File Browse controls
let btnBrowseFile, audioFileInput;

// Waveform
let waveformIdle, waveformBars;

// Playback + upload
let playbackPanel, audioPlayback, btnUpload, btnUploadText, btnUploadSpinner, uploadResult, uploadResultText;

// API status
let apiStatusDot, apiStatusText;

// Clinical Results section
let resultsSection, riskRingArc, riskPctLabel, riskBadge;
let barSvc, barRf, barGb, lblSvc, lblRf, lblGb;
let lblShimmer, barShimmer, lblJitter, barJitter, lblHnr, barHnr;
let uncertaintyBanner, uncertaintyTitle, uncertaintyDesc, uncertaintyBar, uncertaintyPct;
let reportRecordingId, reportModelUsed, reportLlmStatus, llmSourceBadge;
let summarySkeleton, summaryText, btnRefreshReport;

// pgvector Similarity
let similarityResultsContainer, similarityLoading, similarityCountBadge;

// NLP Linguistic Analysis
let tabAcoustic, tabLinguistic;
let panelAcoustic, panelLinguistic;
let guideAcoustic, guideLinguistic;
let btnNlpRecord, btnNlpStop, nlpIconMic, nlpIconStop;
let nlpRecordStatus, nlpRecordTimer, nlpRecordRing1, nlpRecordRing2;
let nlpWaveformIdle, nlpWaveformBars;
let nlpPlaybackPanel, nlpAudioPlayback, btnNlpAnalyze, btnNlpAnalyzeText, btnNlpAnalyzeSpinner;
let nlpResultsSection, nlpLdiBar, nlpLdiLabel, nlpSynBar, nlpSynLabel, nlpCohBar, nlpCohLabel;
let nlpAudioCtx, nlpAnalyser, nlpDataArray, nlpAnimationFrame;
let nlpMarkersList, nlpTranscriptBox, nlpSummaryText, nlpSummarySkeleton;
let nlpErrorBox, nlpErrorText;

// Vision Analysis
let tabVisual;
let panelVisual, guideVisual;
let webcamPreview, btnVisionRecord, btnVisionStop, visionIconCam, visionIconStop;
let visionRecordStatus, visionRecordTimer, visionRecordRing1, visionRecordRing2;
let btnVisionAnalyze, btnVisionAnalyzeText, btnVisionAnalyzeSpinner, visionUploadResult;
let visionResultsSection, visionMarVal, visionBlinkVal, visionHypomimiaBadge;
let visionErrorBox, visionErrorText, apiStatusDotVision, apiStatusTextVision;

// =============================================================================
// 4. APPLICATION STATE
// =============================================================================

/** @type {'idle' | 'recording' | 'stopped'} */
let recorderState = 'idle';

/** @type {MediaRecorder | null} */
let mediaRecorder = null;

/** @type {Blob[]} */
let audioChunks = [];

/** @type {Blob | null} */
let recordedBlob = null;

/** @type {MediaStream | null} */
let activeStream = null;

/** Timer interval reference */
let timerInterval = null;

/** Seconds elapsed during recording */
let elapsedSeconds = 0;

/** Whether the auth form is in sign-in or sign-up mode */
let isSignUpMode = false;

/** Track the latest recording_id so Re-analyse can call without re-upload */
let _lastRecordingId = null;

/**
 * @type {'acoustic' | 'linguistic'} Current active analysis mode
 */
let _analysisMode = 'acoustic';

// NLP-specific recorder state (mirrors acoustic recorder, independent)
/** @type {'idle' | 'recording' | 'stopped'} */
let nlpRecorderState = 'idle';
/** @type {MediaRecorder | null} */
let nlpMediaRecorder = null;
/** @type {Blob[]} */
let nlpAudioChunks = [];
/** @type {Blob | null} */
let nlpRecordedBlob = null;
/** @type {MediaStream | null} */
let nlpActiveStream = null;
let nlpTimerInterval = null;
let nlpElapsedSeconds = 0;

// Vision-specific recorder state
/** @type {'idle' | 'recording' | 'stopped'} */
let visionRecorderState = 'idle';
/** @type {MediaRecorder | null} */
let visionMediaRecorder = null;
/** @type {Blob[]} */
let visionChunks = [];
/** @type {Blob | null} */
let visionRecordedBlob = null;
/** @type {MediaStream | null} */
let visionActiveStream = null;
let visionTimerInterval = null;
let visionElapsedSeconds = 0;

// =============================================================================
// 5. UTILITIES
// =============================================================================

/**
 * Format elapsed seconds as M:SS.
 * @param {number} s - Seconds elapsed.
 * @returns {string}
 */
function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = String(s % 60).padStart(2, '0');
  return `${m}:${sec}`;
}

/**
 * Show or hide an element by toggling the 'hidden' class.
 * @param {HTMLElement} el
 * @param {boolean} visible
 */
function setVisible(el, visible) {
  if (!el) return;
  if (visible) {
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

// =============================================================================
// 5c. PDF EXPORT
// =============================================================================

/**
 * Export a results section to a downloadable clinical PDF.
 *
 * Uses html2pdf.js (loaded via CDN) which renders the element through
 * html2canvas and encodes the result into a jsPDF letter-format document.
 *
 * @param {string} elementId    - ID of the DOM element to capture.
 * @param {string} modalityName - Human-readable label used in the filename
 *                                (e.g. 'Acoustic', 'Linguistic', 'Visual').
 */
function exportToPDF(elementId, modalityName) {
  const element = document.getElementById(elementId);
  if (!element) {
    console.warn(`[VoxTriad] exportToPDF: element #${elementId} not found.`);
    return;
  }

  if (typeof html2pdf === 'undefined') {
    console.error('[VoxTriad] html2pdf.js is not loaded. Check the CDN script tag.');
    alert('PDF export library is not available. Please check your internet connection and reload.');
    return;
  }

  const timestamp = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const opt = {
    margin:      0.5,
    filename:    `VoxTriad_${modalityName}_Clinical_Report_${timestamp}.pdf`,
    image:       { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true, logging: false },
    jsPDF:       { unit: 'in', format: 'letter', orientation: 'portrait' },
  };

  console.info(`[VoxTriad] Generating PDF for #${elementId} → ${opt.filename}`);
  html2pdf().set(opt).from(element).save();
}

// =============================================================================
// 5b. ANALYSIS MODE SWITCHER
// =============================================================================

/**
 * Switch the dashboard between Acoustic and Linguistic modes.
 * Updates tab button classes, panel visibility, and sidebar guides.
 * @param {'acoustic' | 'linguistic'} mode
 */
function setAnalysisMode(mode) {
  _analysisMode = mode;

  const isAcoustic = mode === 'acoustic';
  const isLinguistic = mode === 'linguistic';
  const isVisual = mode === 'visual';

  // ── Tabs ──
  if (tabAcoustic) {
    tabAcoustic.setAttribute('aria-selected', isAcoustic ? 'true' : 'false');
    tabAcoustic.className = 'subnav-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-xs ' + 
        (isAcoustic ? 'text-cyan-300' : 'text-slate-500 hover:bg-white/[.03] hover:text-cyan-200');
  }
  if (tabLinguistic) {
    tabLinguistic.setAttribute('aria-selected', isLinguistic ? 'true' : 'false');
    tabLinguistic.className = 'subnav-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-xs ' + 
        (isLinguistic ? 'text-cyan-300' : 'text-slate-500 hover:bg-white/[.03] hover:text-cyan-200');
  }
  if (tabVisual) {
    tabVisual.setAttribute('aria-selected', isVisual ? 'true' : 'false');
    tabVisual.className = 'subnav-item flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-xs ' + 
        (isVisual ? 'text-cyan-300' : 'text-slate-500 hover:bg-white/[.03] hover:text-cyan-200');
  }

  // ── Panels ──
  setVisible(panelAcoustic, isAcoustic);
  setVisible(panelLinguistic, isLinguistic);
  setVisible(panelVisual, isVisual);

  // ── Sidebar guides ──
  setVisible(guideAcoustic, isAcoustic);
  setVisible(guideLinguistic, isLinguistic);
  setVisible(guideVisual, isVisual);

  // ── Cleanup Webcam ──
  if (!isVisual) {
    if (visionActiveStream) {
      visionActiveStream.getTracks().forEach(t => t.stop());
      visionActiveStream = null;
    }
    if (webcamPreview) {
      webcamPreview.srcObject = null;
    }
    if (visionRecorderState === 'recording' && typeof stopVisionRecording === 'function') {
      stopVisionRecording();
    }
  }

  // ── Hide results when switching ──
  if (!isAcoustic && resultsSection) resultsSection.classList.add('hidden');
  if (!isLinguistic && nlpResultsSection) nlpResultsSection.classList.add('hidden');
  if (!isVisual && visionResultsSection) visionResultsSection.classList.add('hidden');
}

/**
 * Display an error message in the auth error box.
 * @param {string} message
 */
function showAuthError(message) {
  if (!authError) return;
  authError.textContent = message;
  setVisible(authError, true);
}

/**
 * Clear the auth error box.
 */
function clearAuthError() {
  if (!authError) return;
  authError.textContent = '';
  setVisible(authError, false);
}

/**
 * Toggle the login button's loading state.
 * @param {boolean} loading
 */
function setLoginLoading(loading) {
  if (btnLogin) btnLogin.disabled = loading;
  setVisible(btnLoginText, !loading);
  setVisible(btnLoginSpinner, loading);
}

/**
 * Generate a short pseudo-unique session identifier for display.
 * @returns {string}
 */
function generateSessionId() {
  return `VTX-${Date.now().toString(36).toUpperCase()}-${Math.random().toString(36).slice(2, 6).toUpperCase()}`;
}

/**
 * Escape HTML characters to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// =============================================================================
// 5b. VIDEO BACKGROUND CROSSFADE
// =============================================================================

/**
 * Crossfade between background videos based on the active UI section.
 * @param {'auth' | 'dashboard'} section
 */
function switchBgVideo(section) {
  const landing = document.getElementById('bg-video-landing');
  const dashboard = document.getElementById('bg-video-dashboard');
  if (!landing || !dashboard) return;

  if (section === 'dashboard') {
    landing.style.opacity = '0';
    dashboard.style.opacity = '0.30';
  } else {
    landing.style.opacity = '0.40';
    dashboard.style.opacity = '0';
  }
}

/**
 * Attach error handlers to video elements so the page degrades gracefully
 * if video files are missing – the solid black body remains as fallback.
 */
function initVideoBackgrounds() {
  const videos = document.querySelectorAll('#video-bg-container video');
  videos.forEach(v => {
    v.addEventListener('error', () => {
      console.warn(`[VoxTriad] Video failed to load: ${v.src}`);
      v.style.display = 'none';
      // Re-apply gradient fallback on body
      document.body.classList.add('gradient-bg');
    });
  });
}

// =============================================================================
// 6. ANIMATED WAVEFORM (CSS-based decorative bars during recording)
// =============================================================================

const WAVEFORM_BAR_COUNT = 40;

function buildWaveformBars() {
  if (!waveformBars) return;
  waveformBars.innerHTML = '';
  for (let i = 0; i < WAVEFORM_BAR_COUNT; i++) {
    const bar = document.createElement('div');
    bar.className = 'waveform-bar flex-1';
    const dur = (0.4 + Math.random() * 0.8).toFixed(2);
    const delay = (Math.random() * 0.8).toFixed(2);
    bar.style.cssText = `
      --dur: ${dur}s;
      animation-delay: ${delay}s;
      min-height: 4px;
      max-height: 100%;
    `;
    waveformBars.appendChild(bar);
  }
}

/**
 * Switch the waveform display between idle placeholder and animated bars.
 * @param {'idle' | 'active'} mode
 */
function setWaveformMode(mode) {
  if (!waveformBars || !waveformIdle) return;
  if (mode === 'active') {
    setVisible(waveformIdle, false);
    waveformBars.classList.remove('hidden');
    waveformBars.classList.add('flex');
  } else {
    setVisible(waveformIdle, true);
    waveformBars.classList.add('hidden');
    waveformBars.classList.remove('flex');
  }
}

// =============================================================================
// 7. API HEALTH CHECK
// =============================================================================

async function checkApiHealth() {
  if (!apiStatusDot || !apiStatusText) return;
  try {
    let res = null;
    try {
      res = await fetch(`${API_BASE_URL}/api/health`, { signal: AbortSignal.timeout(4000) });
    } catch {
      // Fallback: try alternative loopback address (127.0.0.1 vs localhost) or /health endpoint
      const altHost = API_BASE_URL.includes('localhost') ? 'http://127.0.0.1:8000' : 'http://localhost:8000';
      try {
        res = await fetch(`${altHost}/api/health`, { signal: AbortSignal.timeout(4000) });
      } catch {
        res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(4000) });
      }
    }

    if (res && res.ok) {
      const data = await res.json().catch(() => ({}));
      const isOnline = data.status === 'online' || data.status === 'ok' || res.status === 200;
      if (isOnline) {
        apiStatusDot.className = 'w-2 h-2 rounded-full bg-emerald-400 shadow-sm shadow-emerald-500/50';
        apiStatusText.textContent = 'online';
        apiStatusText.className = 'text-xs font-mono text-emerald-400';
        
        if (document.getElementById('api-status-dot-nlp')) document.getElementById('api-status-dot-nlp').className = apiStatusDot.className;
        if (document.getElementById('api-status-text-nlp')) {
          document.getElementById('api-status-text-nlp').textContent = 'online';
          document.getElementById('api-status-text-nlp').className = apiStatusText.className;
        }
        if (apiStatusDotVision) apiStatusDotVision.className = apiStatusDot.className;
        if (apiStatusTextVision) {
          apiStatusTextVision.textContent = 'online';
          apiStatusTextVision.className = apiStatusText.className;
        }
        return;
      }
    }
    throw new Error(`HTTP ${res ? res.status : 'offline'}`);
  } catch (err) {
    apiStatusDot.className = 'w-2 h-2 rounded-full bg-red-400';
    apiStatusText.textContent = 'offline';
    apiStatusText.className = 'text-xs font-mono text-red-400';
    
    if (document.getElementById('api-status-dot-nlp')) document.getElementById('api-status-dot-nlp').className = apiStatusDot.className;
    if (document.getElementById('api-status-text-nlp')) {
      document.getElementById('api-status-text-nlp').textContent = 'offline';
      document.getElementById('api-status-text-nlp').className = apiStatusText.className;
    }
    if (apiStatusDotVision) apiStatusDotVision.className = apiStatusDot.className;
    if (apiStatusTextVision) {
      apiStatusTextVision.textContent = 'offline';
      apiStatusTextVision.className = apiStatusText.className;
    }
  }
}

// =============================================================================
// 8. AUTHENTICATION
// =============================================================================

/**
 * Sign the user in using Supabase Auth.
 * @param {string} email
 * @param {string} password
 */
async function handleSignIn(email, password) {
  clearAuthError();
  setLoginLoading(true);

  try {
    const { data, error } = await supabaseClient.auth.signInWithPassword({ email, password });
    if (error) throw error;
    onAuthSuccess(data.user);
  } catch (err) {
    showAuthError(err.message || 'Sign-in failed. Please try again.');
  } finally {
    setLoginLoading(false);
  }
}

/**
 * Sign the user up (create a new account).
 * @param {string} email
 * @param {string} password
 */
async function handleSignUp(email, password) {
  clearAuthError();
  setLoginLoading(true);

  try {
    const { data, error } = await supabaseClient.auth.signUp({ email, password });
    if (error) throw error;

    if (data.user && !data.user.confirmed_at) {
      showAuthError('Account created! Please check your email to confirm your address before signing in.');
    } else {
      onAuthSuccess(data.user);
    }
  } catch (err) {
    showAuthError(err.message || 'Sign-up failed. Please try again.');
  } finally {
    setLoginLoading(false);
  }
}

/**
 * Transition the UI into the authenticated (dashboard) state.
 * @param {import('@supabase/supabase-js').User | { email: string, id: string }} user
 */
function onAuthSuccess(user) {
  // Hide fallback verifying session loader
  const authStatus = document.getElementById('auth-status');
  if (authStatus) authStatus.classList.add('hidden');

  // Display authenticated user's email in navbar
  const userEmail = user?.email || 'clinician@hospital.org';
  if (navUserEmail) {
    navUserEmail.textContent = userEmail;
    navUserEmail.title = userEmail;
  }
  if (navUserInfo) {
    setVisible(navUserInfo, true);
    navUserInfo.classList.add('flex');
  }

  // Update dynamic user profile card (Sidebar / Drawer)
  const profileNameEl = document.getElementById('user-profile-name');
  const profileInitialsEl = document.getElementById('user-profile-initials');

  if (profileNameEl && profileInitialsEl) {
    let fullName = user?.user_metadata?.full_name || user?.user_metadata?.name || '';
    if (!fullName && user?.email) {
      fullName = user.email.split('@')[0];
    }
    
    let initials = 'DR';
    if (fullName) {
      const parts = fullName.trim().split(/\s+/);
      if (parts.length > 1) {
        initials = (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
      } else if (parts[0].length > 0) {
        initials = parts[0].substring(0, 2).toUpperCase();
      }
    }

    profileNameEl.textContent = fullName || 'User';
    profileInitialsEl.textContent = initials;
  }

  // Swap sections: hide auth card completely, show screening dashboard
  if (authSection) setVisible(authSection, false);
  if (dashboardSection) setVisible(dashboardSection, true);

  // Crossfade video background to dashboard
  switchBgVideo('dashboard');

  // Set session display ID
  if (dashSessionId) dashSessionId.textContent = generateSessionId();

  // Check API health
  checkApiHealth();
  
  // Load Patient History if element exists
  loadPatientHistory(user.id);
}

function getRiskBadge(riskLevel) {
  const level = (riskLevel || 'pending').toLowerCase();
  
  if (level === 'high') {
    return `<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border text-red-400 bg-red-500/10 border-red-500/20"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>High</span>`;
  } else if (level === 'medium') {
    return `<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border text-amber-400 bg-amber-500/10 border-amber-500/20"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>Medium</span>`;
  } else if (level === 'low') {
    return `<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border text-emerald-400 bg-emerald-500/10 border-emerald-500/20"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>Low</span>`;
  } else {
    return `<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border text-slate-400 bg-slate-800 border-slate-700/50"><span class="w-1.5 h-1.5 rounded-full bg-current"></span>Pending</span>`;
  }
}

async function loadPatientHistory(authId) {
  const tbody = document.getElementById('patient-history-tbody');
  if (!tbody) return; // Fail gracefully if not on dashboard page
  
  try {
    const response = await fetch(`http://127.0.0.1:8000/api/patients?auth_id=${encodeURIComponent(authId)}`);
    if (!response.ok) throw new Error("Failed to fetch history");
    const data = await response.json();
    
    tbody.innerHTML = ''; // clear
    
    if (!data || data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="px-6 py-8 text-center text-slate-500">No patients found.</td></tr>`;
      return;
    }
    
    data.forEach(item => {

      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-800/30 transition-colors group cursor-pointer';
      tr.innerHTML = `
          <td class="px-6 py-4 whitespace-nowrap font-mono text-xs text-slate-400 group-hover:text-primary transition-colors">${item.manual_id || item.id}</td>
          <td class="px-6 py-4 font-medium text-white">${item.full_name}</td>
          <td class="px-6 py-4 text-slate-400">${item.age}</td>
          <td class="px-6 py-4 whitespace-nowrap">${getRiskBadge(item.acoustic_risk)}</td>
          <td class="px-6 py-4 whitespace-nowrap">${getRiskBadge(item.linguistic_risk)}</td>
          <td class="px-6 py-4 whitespace-nowrap text-right">${getRiskBadge(item.video_risk)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading patient history:", err);
    tbody.innerHTML = `<tr><td colspan="5" class="px-6 py-8 text-center text-rose-500">Error loading patients.</td></tr>`;
  }
}

async function updatePatientRisk(modality, riskLevel) {
  if (!window.currentPatient || !window.currentPatient.id) return;
  try {
    const response = await fetch(`${API_BASE_URL}/api/patients/${window.currentPatient.id}/risk`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modality, risk_level: riskLevel })
    });
    if (!response.ok) {
      console.warn(`[VoxTriad] Failed to update ${modality} risk on backend.`);
    }
  } catch (err) {
    console.warn(`[VoxTriad] Exception updating ${modality} risk`, err);
  }
}

/**
 * Sign out the current user and redirect to login.html.
 */
async function handleSignOut() {
  const authStatus = document.getElementById('auth-status');
  const authStatusText = document.getElementById('auth-status-text');
  if (authStatus) authStatus.classList.remove('hidden');
  if (authStatusText) authStatusText.textContent = 'Signing out…';

  try {
    const client = (typeof window !== 'undefined' && (window.supabaseClient || window.supabaseApp))
      ? (window.supabaseClient || window.supabaseApp)
      : supabaseClient;
    if (client && client.auth && typeof client.auth.signOut === 'function') {
      await client.auth.signOut();
    }
  } catch (err) {
    console.warn('[VoxTriad] Sign out warning:', err);
  }

  // Reset navbar
  if (navUserInfo) {
    setVisible(navUserInfo, false);
    navUserInfo.classList.remove('flex');
  }
  if (navUserEmail) navUserEmail.textContent = '';

  // Clear all cached patient and session data
  localStorage.clear();
  sessionStorage.clear();

  // Forcefully redirect to login.html
  window.location.href = 'login.html';
}

/**
 * Update the auth card DOM between Sign In and Sign Up views.
 */
function updateAuthView() {
  if (isSignUpMode) {
    if (authCardTitle) authCardTitle.textContent = 'Clinician Registration';
    if (authCardDesc) authCardDesc.textContent = 'Create an account to access the screening dashboard';
    if (btnLoginText) btnLoginText.textContent = 'Create Account';
    if (toggleAuthPrompt) toggleAuthPrompt.textContent = 'Already have an account?';
    if (btnToggleRegister) btnToggleRegister.textContent = 'Sign in';
  } else {
    if (authCardTitle) authCardTitle.textContent = 'Clinician Sign In';
    if (authCardDesc) authCardDesc.textContent = 'Access your screening dashboard';
    if (btnLoginText) btnLoginText.textContent = 'Sign In';
    if (toggleAuthPrompt) toggleAuthPrompt.textContent = 'No account yet?';
    if (btnToggleRegister) btnToggleRegister.textContent = 'Create one';
  }
}

/**
 * Central handler for auth submission (triggered by submit event or button click).
 * @param {Event} [e]
 */
async function submitAuthForm(e) {
  if (e) {
    e.preventDefault();
    e.stopPropagation();
  }
  const email = authEmailInput ? authEmailInput.value.trim() : '';
  const password = authPasswordInput ? authPasswordInput.value : '';

  if (!email || !password) {
    showAuthError('Please enter your email and password.');
    return;
  }

  if (isSignUpMode) {
    await handleSignUp(email, password);
  } else {
    await handleSignIn(email, password);
  }
}

// =============================================================================
// 9. MEDIA RECORDER – RECORDING PIPELINE
// =============================================================================

/**
 * Request microphone access and start a MediaRecorder session.
 */
async function startRecording() {
  if (recorderState === 'recording') return;
  const errBanner = document.getElementById('audio-error-banner');
  if (errBanner) errBanner.classList.add('hidden');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    activeStream = stream;
    audioChunks = [];
    recorderState = 'recording';
    elapsedSeconds = 0;

    const mimeType = getSupportedMimeType();
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener('stop', onRecordingStop);
    mediaRecorder.onstop = onRecordingStop;
    mediaRecorder.start(250);

    setRecordingUIActive(true);

    timerInterval = setInterval(() => {
      elapsedSeconds++;
      if (recordTimer) recordTimer.textContent = formatTime(elapsedSeconds);
    }, 1000);

    if (recordStatus) recordStatus.textContent = 'Recording… speak clearly into the microphone';
    setWaveformMode('active');

  } catch (err) {
    console.error('[VoxTriad] Microphone access error:', err);
    if (recordStatus) recordStatus.textContent = `Microphone error: ${err.message}. Please grant microphone permission.`;
  }
}

/**
 * Stop an active recording session.
 * @param {boolean} [keepBlob=true]
 */
function stopRecording(keepBlob = true) {
  if (recorderState !== 'recording') return;

  clearInterval(timerInterval);
  timerInterval = null;
  recorderState = 'stopped';

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }

  if (activeStream) {
    activeStream.getTracks().forEach(track => track.stop());
    activeStream = null;
  }

  setRecordingUIActive(false);
  setWaveformMode('idle');
  if (recordStatus) {
    recordStatus.textContent = keepBlob ? 'Recording complete. Review and upload below.' : 'Recording cancelled.';
  }
}

/**
 * Fired when MediaRecorder has fully stopped; decodes audio and converts to WAV.
 */
async function onRecordingStop() {
  if (audioChunks.length === 0) {
    if (recordStatus) recordStatus.textContent = 'No audio captured. Please try again.';
    return;
  }

  const mimeType = mediaRecorder?.mimeType || 'audio/webm';
  const webmBlob = new Blob(audioChunks, { type: mimeType });
  audioChunks = [];

  try {
    const arrayBuffer = await webmBlob.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioCtx();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    recordedBlob = audioBufferToWav(audioBuffer);

    const blobUrl = URL.createObjectURL(recordedBlob);
    if (audioPlayback) audioPlayback.src = blobUrl;
    console.log('[VoxTriad] Converted recording to pure audio/wav');
  } catch (err) {
    console.error('WAV conversion failed:', err);
    recordedBlob = webmBlob; // Fallback
    const blobUrl = URL.createObjectURL(recordedBlob);
    if (audioPlayback) audioPlayback.src = blobUrl;
  }

  setVisible(playbackPanel, true);
  setVisible(uploadResult, false);
}

function getSupportedMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4',
  ];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return null;
}

// =============================================================================
// 10. RECORDER UI HELPERS
// =============================================================================

function setRecordingUIActive(active) {
  const idleState = document.getElementById('state-idle');
  const recState = document.getElementById('state-recording');
  const analyzingState = document.getElementById('state-analyzing');
  if (active) {
    if (idleState) idleState.classList.add('hidden');
    if (recState) { recState.classList.remove('hidden'); recState.classList.add('flex'); }
    if (analyzingState) { analyzingState.classList.add('hidden'); analyzingState.classList.remove('flex'); }
  } else {
    if (recState) { recState.classList.add('hidden'); recState.classList.remove('flex'); }
  }

  setVisible(iconMic, !active);
  setVisible(iconStop, active);
  setVisible(recordRing1, active);
  setVisible(recordRing2, active);

  if (btnRecord) {
    if (active) {
      btnRecord.classList.replace('bg-cyan-300', 'bg-rose-400');
      btnRecord.classList.replace('shadow-cyan-400/30', 'shadow-rose-400/30');
    } else {
      btnRecord.classList.replace('bg-rose-400', 'bg-cyan-300');
      btnRecord.classList.replace('shadow-rose-400/30', 'shadow-cyan-400/30');
    }
    btnRecord.setAttribute('aria-label', active ? 'Stop recording' : 'Start recording');
  }

  setVisible(btnStop, active);
  setVisible(recordTimer, active);
}

function resetRecorderUI() {
  recorderState = 'idle';
  elapsedSeconds = 0;
  recordedBlob = null;
  audioChunks = [];

  const idleState = document.getElementById('state-idle');
  const recState = document.getElementById('state-recording');
  const analyzingState = document.getElementById('state-analyzing');
  if (idleState) idleState.classList.remove('hidden');
  if (recState) { recState.classList.add('hidden'); recState.classList.remove('flex'); }
  if (analyzingState) { analyzingState.classList.add('hidden'); analyzingState.classList.remove('flex'); }

  setVisible(playbackPanel, false);
  setVisible(uploadResult, false);
  setVisible(btnStop, false);
  setVisible(recordTimer, false);

  if (btnUpload) {
    btnUpload.disabled = false;
    btnUpload.className = "w-full flex items-center justify-center gap-2 rounded-xl py-3 text-xs font-semibold text-cyan-200 transition-all border border-cyan-400/20 bg-cyan-400/10 hover:bg-cyan-400/20";
    if (btnUploadText) btnUploadText.innerHTML = "Analyse Audio";
  }

  setVisible(iconMic, true);
  setVisible(iconStop, false);
  setVisible(recordRing1, false);
  setVisible(recordRing2, false);

  if (recordStatus) recordStatus.textContent = 'Click the button to start recording';
  if (recordTimer) recordTimer.textContent = '0:00';
  if (audioPlayback) audioPlayback.src = '';
  if (uploadResultText) uploadResultText.textContent = '';

  if (similarityResultsContainer) {
    similarityResultsContainer.innerHTML = `
      <div class="col-span-full py-6 text-center text-slate-500 text-xs">
        Awaiting recording analysis to search nearest cohort neighbors…
      </div>
    `;
  }
  if (similarityCountBadge) setVisible(similarityCountBadge, false);
  if (similarityLoading) setVisible(similarityLoading, false);

  setWaveformMode('idle');
}

// =============================================================================
// 11. CLINICAL RESULTS & RISK RING
// =============================================================================

const RISK_CIRCUMFERENCE = 99.9; // ≈ 2π × 15.9 (SVG units)

function _riskColor(prob) {
  if (prob < 0.35) return { stroke: '#34d399', badge: 'bg-emerald-500/20 text-emerald-300', label: 'Low Risk' };
  if (prob < 0.65) return { stroke: '#fbbf24', badge: 'bg-amber-500/20 text-amber-300',   label: 'Moderate Risk' };
  return              { stroke: '#f87171', badge: 'bg-red-500/20 text-red-300',           label: 'High Risk' };
}

function _uncertaintyStyle(tier) {
  const styles = {
    high_confidence:    { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-300', title: 'High Confidence', desc: 'Analysis reliability is high. Results may be used to support clinical decision-making.' },
    low:                { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-300', title: 'High Confidence', desc: 'Analysis reliability is high. Results may be used to support clinical decision-making.' },
    review_recommended: { bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   text: 'text-amber-300',   title: 'Review Recommended', desc: 'Moderate uncertainty detected. A clinician review of the recording quality is advised.' },
    mandatory_review:   { bg: 'bg-red-500/10',     border: 'border-red-500/30',     text: 'text-red-300',     title: 'Re-recording Required', desc: 'Data quality is insufficient for reliable screening. Please record again in a quieter environment.' },
  };
  return styles[tier] || styles.high_confidence;
}

function populateResultsDashboard(uploadJson, reportJson) {
  if (!riskRingArc || !riskPctLabel || !riskBadge) return;

  // ── Risk ring ──────────────────────────────────────────────────────────────
  const prob = uploadJson.pd_probability ?? reportJson.pd_probability ?? null;

  if (prob !== null) {
    const pct  = Math.round(prob * 100);
    const offset = 100 - pct; // SVG has stroke-dasharray="100 100"
    const col  = _riskColor(prob);

    riskRingArc.setAttribute('stroke', col.stroke);
    riskRingArc.setAttribute('stroke-dashoffset', offset);
    riskPctLabel.textContent = `${pct}%`;
    riskPctLabel.style.color = col.stroke;

    riskBadge.className = `px-3 py-1.5 rounded-full text-xs font-semibold ${col.badge}`;
    riskBadge.textContent = col.label;
    riskBadge.classList.remove('hidden');
    
    updatePatientRisk('acoustic', col.label.toLowerCase());
  } else {
    riskPctLabel.textContent = 'N/A';
    riskBadge.textContent = 'Ensemble not trained';
    riskBadge.className = 'px-3 py-1.5 rounded-full text-xs font-semibold bg-slate-700/60 text-slate-400';
    riskBadge.classList.remove('hidden');
  }

  // ── Per-model bars ─────────────────────────────────────────────────────────
  const ind = uploadJson.individual_probs || {};
  function setBar(bar, lbl, val) {
    if (!bar || !lbl) return;
    const p = val !== undefined ? Math.round(val * 100) : null;
    bar.style.width = p !== null ? `${p}%` : '0%';
    lbl.textContent = p !== null ? `${p}%` : '–';
  }
  setBar(barSvc, lblSvc, ind.svc);
  setBar(barRf,  lblRf,  ind.rf);
  setBar(barGb,  lblGb,  ind.gb);

  // ── Acoustic Features ──────────────────────────────────────────────────────
  const af = uploadJson.acoustic_features || {};
  if (lblShimmer && barShimmer) {
    const s = af.shimmer !== undefined ? (af.shimmer * 100).toFixed(1) : 0;
    lblShimmer.textContent = `${s}%`;
    // Cap visual width for UI aesthetic purposes
    barShimmer.style.width = `${Math.min(s * 5, 100)}%`;
  }
  if (lblJitter && barJitter) {
    const j = af.jitter !== undefined ? (af.jitter * 100).toFixed(2) : 0;
    lblJitter.textContent = `${j}%`;
    barJitter.style.width = `${Math.min(j * 10, 100)}%`;
  }
  if (lblHnr && barHnr) {
    const h = af.hnr !== undefined ? af.hnr.toFixed(1) : 0;
    lblHnr.textContent = `${h} dB`;
    barHnr.style.width = `${Math.min((h / 30) * 100, 100)}%`;
  }

  // ── Uncertainty banner ─────────────────────────────────────────────────────
  const tier = uploadJson.uncertainty_tier || reportJson.uncertainty_tier || 'low';
  const unc  = uploadJson.uncertainty_score ?? reportJson.uncertainty_score ?? 0.18;
  const st   = _uncertaintyStyle(tier);

  if (uncertaintyBanner && uncertaintyTitle && uncertaintyDesc) {
    uncertaintyBanner.className = `rounded-xl p-4 border text-sm font-medium flex items-start gap-3 transition-all duration-500 ${st.bg} ${st.border} ${st.text}`;
    uncertaintyBanner.style.display = '';
    uncertaintyTitle.textContent = st.title;
    uncertaintyDesc.textContent  = st.desc;
  }

  const certaintyPct = unc !== null ? Math.round((1 - unc) * 100) : 82;
  const barColor = (tier === 'high_confidence' || tier === 'low') ? 'bg-emerald-500' : tier === 'mandatory_review' ? 'bg-red-500' : 'bg-amber-500';
  if (uncertaintyBar) {
    uncertaintyBar.className = `h-full rounded-full transition-all duration-700 ${barColor}`;
    uncertaintyBar.style.width = `${certaintyPct}%`;
  }
  if (uncertaintyPct) uncertaintyPct.textContent = `${certaintyPct}%`;

  // ── Metadata ───────────────────────────────────────────────────────────────
  const recId = uploadJson.recording_id || reportJson.recording_id || '—';
  if (reportRecordingId) {
    reportRecordingId.textContent = recId.length > 12 ? recId.slice(0, 8) + '…' : recId;
    reportRecordingId.title = recId;
  }
  if (reportModelUsed) reportModelUsed.textContent  = reportJson.model_used || '—';
  if (reportLlmStatus) {
    reportLlmStatus.textContent  = reportJson.llm_available
      ? (reportJson.fallback_used ? 'Rule-based fallback' : 'Groq AI')
      : 'Offline';
  }

  // ── LLM source badge ───────────────────────────────────────────────────────
  if (llmSourceBadge) {
    if (reportJson.fallback_used) {
      llmSourceBadge.textContent  = 'Rule-based';
      llmSourceBadge.className    = 'text-xs px-2 py-0.5 rounded-full font-medium bg-amber-500/20 text-amber-300';
    } else {
      llmSourceBadge.textContent  = 'Groq AI';
      llmSourceBadge.className    = 'text-xs px-2 py-0.5 rounded-full font-medium bg-cyan-500/20 text-cyan-300';
    }
    llmSourceBadge.classList.remove('hidden');
  }

  // ── Summary text ───────────────────────────────────────────────────────────
  if (summarySkeleton) summarySkeleton.classList.add('hidden');
  if (summaryText) {
    summaryText.textContent = reportJson.summary_text || 'No summary available.';
    summaryText.classList.remove('hidden');
  }
}

async function fetchClinicalReport(recordingId, uploadJson = {}, forceRefresh = false) {
  const uploadData = (uploadJson && Object.keys(uploadJson).length > 0) ? uploadJson : (_lastUploadResult || {});
  let token = null;
  try {
    const { data: { session } } = await supabaseClient.auth.getSession();
    token = session?.access_token ?? null;
  } catch (e) {
    console.warn('[VoxTriad] Could not get session for clinical report:', e);
  }

  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  setVisible(resultsSection, true);
  if (summarySkeleton) summarySkeleton.classList.remove('hidden');
  if (summaryText) summaryText.classList.add('hidden');
  if (llmSourceBadge) llmSourceBadge.classList.add('hidden');

  try {
    const url = `${API_BASE_URL}/api/report/${recordingId}${forceRefresh ? '?force_refresh=true' : ''}`;
    const response = await fetch(url, { method: 'GET', headers });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const reportJson = await response.json();
    populateResultsDashboard(uploadData, reportJson);

    if (recordStatus) recordStatus.textContent = '✓ Analysis complete – clinical summary generated.';
  } catch (err) {
    console.error('[VoxTriad] fetchClinicalReport failed:', err);
    if (summarySkeleton) summarySkeleton.classList.add('hidden');
    if (summaryText) {
      summaryText.textContent = `Could not retrieve clinical summary:\n${err.message}\n\nEnsure the backend is running and GROQ_API_KEY is configured.`;
      summaryText.classList.remove('hidden');
    }
    if (recordStatus) recordStatus.textContent = 'Report fetch failed – see console.';
  }
}

// =============================================================================
// 12. PHASE 5: PGVECTOR ACOUSTIC SIMILARITY
// =============================================================================

async function fetchSimilarRecordings(recordingId) {
  if (!recordingId || !similarityResultsContainer) return;

  let token = null;
  try {
    const { data: { session } } = await supabaseClient.auth.getSession();
    token = session?.access_token ?? null;
  } catch (e) {
    console.warn('[VoxTriad] Could not get session for similarity search:', e);
  }

  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  if (similarityLoading) setVisible(similarityLoading, true);
  if (similarityCountBadge) setVisible(similarityCountBadge, false);
  similarityResultsContainer.innerHTML = '';

  try {
    const url = `${API_BASE_URL}/api/similar/${recordingId}?top_k=6`;
    const response = await fetch(url, { method: 'GET', headers });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const similarityData = await response.json();
    const rawMatches = Array.isArray(similarityData.results) ? similarityData.results : [];

    const filteredMatches = rawMatches.filter(m => m.recording_id !== recordingId);
    const displayMatches = filteredMatches.length > 0 ? filteredMatches : rawMatches;

    renderSimilarityResults(displayMatches);
  } catch (err) {
    console.error('[VoxTriad] fetchSimilarRecordings failed:', err);
    similarityResultsContainer.innerHTML = `
      <div class="col-span-full py-4 px-4 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
        <p class="text-xs text-red-300 font-mono">Similarity search notice: ${escapeHtml(err.message)}</p>
        <p class="text-[11px] text-slate-500 mt-1">Requires pgvector match_recordings RPC deployed in Supabase.</p>
      </div>
    `;
  } finally {
    if (similarityLoading) setVisible(similarityLoading, false);
  }
}

function renderSimilarityResults(matches) {
  if (!similarityResultsContainer) return;
  similarityResultsContainer.innerHTML = '';

  if (!matches || matches.length === 0) {
    if (similarityCountBadge) {
      similarityCountBadge.textContent = '0 matches';
      setVisible(similarityCountBadge, true);
    }
    similarityResultsContainer.innerHTML = `
      <div class="col-span-full py-8 text-center text-slate-400 bg-surface-800/30 rounded-xl border border-dashed border-slate-700/60 p-6">
        <svg class="w-8 h-8 text-slate-600 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
        <p class="text-sm font-medium text-slate-300">No prior cohort recordings found</p>
        <p class="text-xs text-slate-500 mt-1">As additional patient audio samples are uploaded to the database, nearest-neighbor clusters will appear here.</p>
      </div>
    `;
    return;
  }

  if (similarityCountBadge) {
    similarityCountBadge.textContent = `${matches.length} match${matches.length > 1 ? 'es' : ''}`;
    setVisible(similarityCountBadge, true);
  }

  matches.forEach((item, index) => {
    const rawSim = typeof item.similarity === 'number'
      ? item.similarity
      : (parseFloat(item.similarity) || 0);
    const simPct = (rawSim * 100).toFixed(1) + '%';
    const barWidth = Math.max(0, Math.min(100, rawSim * 100)).toFixed(1) + '%';

    let riskBadgeClass = 'bg-slate-700/60 text-slate-400 border-slate-600/50';
    let riskText = 'Risk: N/A';

    if (item.risk_score !== null && item.risk_score !== undefined) {
      const score = Number(item.risk_score);
      const scorePct = Math.round(score * 100);

      if (score < 0.35) {
        riskBadgeClass = 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30';
        riskText = `Low Risk (${scorePct}%)`;
      } else if (score < 0.65) {
        riskBadgeClass = 'bg-amber-500/15 text-amber-300 border-amber-500/30';
        riskText = `Moderate Risk (${scorePct}%)`;
      } else {
        riskBadgeClass = 'bg-red-500/15 text-red-300 border-red-500/30';
        riskText = `High Risk (${scorePct}%)`;
      }
    }

    const shortId = item.recording_id
      ? `${item.recording_id.slice(0, 8)}…${item.recording_id.slice(-4)}`
      : 'Anonymous';

    const card = document.createElement('div');
    card.className = 'glass bg-slate-900/30 backdrop-blur-lg border border-white/10 hover:border-cyan-500/40 rounded-xl p-4 transition-all duration-200 flex flex-col justify-between gap-3 shadow-lg hover:shadow-cyan-500/20';
    card.innerHTML = `
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2 min-w-0">
          <span class="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-bold flex items-center justify-center shrink-0">
            #${index + 1}
          </span>
          <span class="font-mono text-xs text-slate-300 truncate" title="${escapeHtml(item.recording_id || '')}">
            ${escapeHtml(shortId)}
          </span>
        </div>
        <span class="px-2 py-0.5 rounded-full text-[11px] font-semibold border ${riskBadgeClass} shrink-0">
          ${riskText}
        </span>
      </div>

      <div class="space-y-1.5 mt-1">
        <div class="flex justify-between items-center text-xs">
          <span class="text-slate-400">Cosine Similarity</span>
          <span class="font-mono font-bold text-white">${simPct}</span>
        </div>
        <div class="w-full h-1.5 rounded-full bg-slate-700/70 overflow-hidden">
          <div class="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-700" style="width: ${barWidth}"></div>
        </div>
      </div>
    `;

    similarityResultsContainer.appendChild(card);
  });
}

// =============================================================================
// 13. AUDIO UPLOAD PIPELINE
// =============================================================================

/** Track the latest upload response so fetchClinicalReport can access it */
let _lastUploadResult = null;

async function uploadAudio() {
  if (!recordedBlob) {
    console.warn('[VoxTriad] uploadAudio called with no recordedBlob.');
    if (recordStatus) recordStatus.textContent = 'Please record audio before uploading.';
    return;
  }

  const idleState = document.getElementById('state-idle');
  const recState = document.getElementById('state-recording');
  const analyzingState = document.getElementById('state-analyzing');
  if (idleState) idleState.classList.add('hidden');
  if (recState) { recState.classList.add('hidden'); recState.classList.remove('flex'); }
  if (analyzingState) { analyzingState.classList.remove('hidden'); analyzingState.classList.add('flex'); }

  let isSuccess = false;
  if (btnUpload) {
    btnUpload.disabled = true;
    btnUpload.className = "w-full flex items-center justify-center gap-2 rounded-xl py-3 text-xs font-semibold text-cyan-200 transition-all border border-cyan-400/20 bg-cyan-400/10 hover:bg-cyan-400/20";
    if (btnUploadText) btnUploadText.innerHTML = "Analyse Audio";
  }
  setVisible(btnUploadText, false);
  setVisible(btnUploadSpinner, true);
  setVisible(resultsSection, false);
  if (recordStatus) recordStatus.textContent = 'Analyzing vocal biomarkers... Please wait.';

  try {
    const { data: { session }, error: sessionError } = await supabaseClient.auth.getSession();
    if (sessionError || !session) {
      throw new Error('Authentication session missing. Please sign in again.');
    }

    const formData = new FormData();
    formData.append('file', recordedBlob, 'recording.wav');

    // Send to FastAPI backend (?deep=true ensures wav2vec2 embedding is extracted and stored)
    const response = await fetch(`${API_BASE_URL}/api/audio/upload?deep=true`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${session.access_token}`
      },
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(`Backend Error: ${errData.detail || errData.message || response.statusText}`);
    }

    const result = await response.json();

    // Safely extract the ID regardless of which key FastAPI uses
    const recId = result.recording_id || result.id || result.recordingId || (result.data && result.data.id);

    if (!recId) {
      console.error('[VoxTriad] Backend response missing ID. Raw response:', result);
      throw new Error('Upload succeeded, but backend did not return a valid recording ID.');
    }

    console.log('[VoxTriad] Upload successful, ID:', recId);

    if (uploadResultText) {
      uploadResultText.textContent = JSON.stringify(result, null, 2);
    }
    setVisible(uploadResult, true);

    _lastRecordingId = recId;
    _lastUploadResult = result;

    if (recordStatus) recordStatus.textContent = '✓ Upload successful – generating clinical report & similarity matches…';

    // Trigger downstream AI and pgvector similarity search using the resolved ID
    if (typeof fetchClinicalReport === 'function') fetchClinicalReport(recId, result);
    if (typeof fetchSimilarRecordings === 'function') fetchSimilarRecordings(recId);

    isSuccess = true;
    if (btnUpload) {
      btnUpload.className = "w-full flex items-center justify-center gap-2 rounded-xl py-3 text-xs font-semibold transition-all bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 cursor-default";
      if (btnUploadText) btnUploadText.innerHTML = '<i data-lucide="check-circle" class="w-4 h-4"></i> Analysis Complete';
      if (window.lucide && window.lucide.createIcons) window.lucide.createIcons();
      btnUpload.disabled = true;
    }

  } catch (error) {
    console.error("Audio Processing Error:", error);
    const errorBanner = document.getElementById('audio-error-banner');
    const errorText = document.getElementById('audio-error-text');
    
    if (errorBanner && errorText) {
      // 1. Get the raw error message
      let displayMessage = error.message;
      
      // 2. Strip out "Feature extraction error: 400: ", "Backend Error: ", etc.
      displayMessage = displayMessage.replace(/^(Feature extraction error:|Backend Error:)?\s*\d*:\s*/i, '').trim();
      // Also catch "Backend Error: " without a status code number
      displayMessage = displayMessage.replace(/^Backend Error:\s*/i, '').trim();
      
      // 3. Handle offline network errors
      if (displayMessage === 'Failed to fetch') {
          displayMessage = "Network error. Please check if the backend server is running.";
      }
      
      // 4. Inject the clean message
      errorText.textContent = displayMessage || "An unknown error occurred while processing the audio.";
      errorBanner.classList.remove('hidden');
    }
  } finally {
    // Ensure UI always resets to idle so user can try again
    const stateAnalyzing = document.getElementById('state-analyzing');
    const stateIdle = document.getElementById('state-idle');
    if (stateAnalyzing) { stateAnalyzing.classList.add('hidden'); stateAnalyzing.classList.remove('flex'); }
    if (stateIdle) { stateIdle.classList.remove('hidden'); stateIdle.classList.add('flex'); }
    
    // Reset button states
    if (!isSuccess && btnUpload) {
      btnUpload.disabled = false;
      if (btnUploadText) btnUploadText.innerHTML = "Analyse Audio";
    }
    setVisible(btnUploadText, true);
    setVisible(btnUploadSpinner, false);
  }
}

// =============================================================================
// 13b. NLP LINGUISTIC ANALYSIS PIPELINE
// =============================================================================

/**
 * Build decorative waveform bars for the NLP recorder panel.
 */
function buildNlpWaveformBars() {
  if (!nlpWaveformBars) return;
  nlpWaveformBars.innerHTML = '';
  for (let i = 0; i < 40; i++) {
    const bar = document.createElement('div');
    bar.className = 'waveform-bar flex-1';
    const dur = (0.4 + Math.random() * 0.8).toFixed(2);
    bar.style.cssText = `--dur:${dur}s; animation-delay:${(Math.random() * 0.8).toFixed(2)}s; min-height:4px; max-height:100%;`;
    nlpWaveformBars.appendChild(bar);
  }
}

function setNlpWaveformMode(mode) {
  if (!nlpWaveformBars || !nlpWaveformIdle) return;
  if (mode === 'active') {
    setVisible(nlpWaveformIdle, false);
    nlpWaveformBars.classList.remove('hidden');
    nlpWaveformBars.classList.add('flex');
  } else {
    setVisible(nlpWaveformIdle, true);
    nlpWaveformBars.classList.add('hidden');
    nlpWaveformBars.classList.remove('flex');
  }
}

function setNlpRecordingUIActive(active) {
  const stateIdle = document.getElementById('ling-state-idle');
  const stateActive = document.getElementById('ling-state-active');
  if (stateIdle && stateActive) {
    if (active) {
      stateIdle.classList.add('hidden');
      stateActive.classList.remove('hidden');
      stateActive.classList.add('flex');
    } else {
      stateActive.classList.add('hidden');
      stateActive.classList.remove('flex');
    }
  }

  setVisible(nlpIconMic, !active);
  setVisible(nlpIconStop, active);
  setVisible(nlpRecordRing1, active);
  setVisible(nlpRecordRing2, active);
  setVisible(btnNlpStop, active);
  setVisible(nlpRecordTimer, active);
  
  const nlpStopText = document.getElementById('nlp-stop-text');
  if (nlpStopText) {
    if (active) nlpStopText.classList.remove('hidden');
    else nlpStopText.classList.add('hidden');
  }

  const recIndicator = document.getElementById('nlp-rec-indicator');
  if (recIndicator) {
    if (active) recIndicator.classList.remove('hidden');
    else recIndicator.classList.add('hidden');
  }
  
  const listeningText = document.getElementById('nlp-listening-text');
  if (listeningText) {
    if (active) listeningText.classList.remove('hidden');
    else listeningText.classList.add('hidden');
  }

  if (btnNlpRecord) {
    if (active) {
      btnNlpRecord.classList.replace('bg-cyan-300', 'bg-rose-400');
      btnNlpRecord.classList.replace('shadow-cyan-400/30', 'shadow-rose-400/30');
    } else {
      btnNlpRecord.classList.replace('bg-rose-400', 'bg-cyan-300');
      btnNlpRecord.classList.replace('shadow-rose-400/30', 'shadow-cyan-400/30');
    }
    btnNlpRecord.setAttribute('aria-label', active ? 'Stop recording' : 'Start recording');
  }
}

function resetNlpRecorderUI() {
  const stateIdle = document.getElementById('ling-state-idle');
  const stateActive = document.getElementById('ling-state-active');
  if (stateIdle && stateActive) {
    stateActive.classList.add('hidden');
    stateActive.classList.remove('flex');
    stateIdle.classList.remove('hidden');
  }
  
  const errorBanner = document.getElementById('nlp-error-banner');
  if (errorBanner) {
    errorBanner.classList.add('hidden');
  }
  
  nlpRecorderState = 'idle';
  nlpElapsedSeconds = 0;
  nlpRecordedBlob = null;
  nlpAudioChunks = [];
  setVisible(nlpPlaybackPanel, false);
  setVisible(btnNlpStop, false);
  setVisible(nlpRecordTimer, false);
  setVisible(nlpIconMic, true);
  setVisible(nlpIconStop, false);
  setVisible(nlpRecordRing1, false);
  setVisible(nlpRecordRing2, false);
  if (nlpRecordStatus) nlpRecordStatus.textContent = 'Click to start passage recording';
  if (nlpRecordTimer)  nlpRecordTimer.textContent = '0:00';
  if (nlpAudioPlayback) nlpAudioPlayback.src = '';
  setNlpWaveformMode('idle');
}

async function startNlpRecording() {
  if (nlpRecorderState === 'recording') return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: false, noiseSuppression: false },
    });
    nlpActiveStream = stream;
    nlpAudioChunks = [];

    // Trigger canvas visualizer
    try {
      nlpAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = nlpAudioCtx.createMediaStreamSource(stream);
      nlpAnalyser = nlpAudioCtx.createAnalyser();
      nlpAnalyser.fftSize = 2048;
      source.connect(nlpAnalyser);
      nlpDataArray = new Uint8Array(nlpAnalyser.frequencyBinCount);
      drawNlpWaveform();
    } catch (e) {
      console.warn('[VoxTriad] Canvas visualizer could not be started:', e);
    }

    const mimeType = getSupportedMimeType();
    nlpMediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    nlpMediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) nlpAudioChunks.push(e.data);
    };

    nlpMediaRecorder.onstop = async () => {
      const rawBlob = new Blob(nlpAudioChunks, { type: nlpMediaRecorder.mimeType || 'audio/webm' });
      nlpAudioChunks = [];
      // Convert to WAV for backend compatibility
      try {
        const arrayBuffer = await rawBlob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        nlpRecordedBlob = audioBufferToWav(audioBuffer);
      } catch {
        nlpRecordedBlob = rawBlob; // fall back to raw if WAV conversion fails
      }
      const nlpAudioPlayer = document.getElementById('nlp-audio-playback');
      if (nlpAudioPlayer && nlpRecordedBlob) {
          nlpAudioPlayer.src = URL.createObjectURL(nlpRecordedBlob);
      }
      setVisible(nlpPlaybackPanel, true);
      nlpRecorderState = 'stopped';
      setNlpWaveformMode('idle');
      if (nlpRecordStatus) nlpRecordStatus.textContent = '✓ Recording complete – review playback and click Analyse Passage.';
    };

    nlpMediaRecorder.start(250);
    nlpRecorderState = 'recording';
    setNlpRecordingUIActive(true);
    setNlpWaveformMode('active');
    if (nlpRecordStatus) nlpRecordStatus.textContent = 'Recording passage… Speak clearly and follow the text above.';

    nlpElapsedSeconds = 0;
    if (nlpRecordTimer) nlpRecordTimer.textContent = '0:00';
    nlpTimerInterval = setInterval(() => {
      nlpElapsedSeconds++;
      if (nlpRecordTimer) nlpRecordTimer.textContent = formatTime(nlpElapsedSeconds);
    }, 1000);

  } catch (err) {
    console.error('[VoxTriad] NLP microphone access failed:', err);
    if (nlpRecordStatus) nlpRecordStatus.textContent = `Microphone access denied: ${err.message}`;
  }
}

function stopNlpRecording() {
  if (nlpTimerInterval) { clearInterval(nlpTimerInterval); nlpTimerInterval = null; }
  if (nlpMediaRecorder && nlpMediaRecorder.state !== 'inactive') nlpMediaRecorder.stop();
  if (nlpActiveStream) { nlpActiveStream.getTracks().forEach(t => t.stop()); nlpActiveStream = null; }
  
  if (nlpAnimationFrame) { cancelAnimationFrame(nlpAnimationFrame); nlpAnimationFrame = null; }
  if (nlpAudioCtx && nlpAudioCtx.state !== 'closed') { nlpAudioCtx.close().catch(()=>{}); nlpAudioCtx = null; }
  
  setNlpRecordingUIActive(false);
}

function drawNlpWaveform() {
  const canvas = document.getElementById('nlp-waveform-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  // Dynamically size canvas to match parent
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  const width = canvas.width;
  const height = canvas.height;
  
  nlpAnimationFrame = requestAnimationFrame(drawNlpWaveform);
  nlpAnalyser.getByteTimeDomainData(nlpDataArray);
  
  // Fade effect
  ctx.fillStyle = 'rgba(11, 17, 33, 0.2)';
  ctx.fillRect(0, 0, width, height);
  
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#22d3ee'; // cyan-400
  ctx.beginPath();
  
  const sliceWidth = width * 1.0 / nlpAnalyser.frequencyBinCount;
  let x = 0;
  
  for (let i = 0; i < nlpAnalyser.frequencyBinCount; i++) {
    const v = nlpDataArray[i] / 128.0;
    const y = v * height / 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    x += sliceWidth;
  }
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

/**
 * Send the recorded passage audio blob to POST /api/nlp/analyze.
 * Renders results into the NLP results section on success.
 * @param {Blob} audioBlob
 */
async function uploadLinguisticAudio(audioBlob) {
  if (!audioBlob) {
    console.warn('[VoxTriad] uploadLinguisticAudio called with no blob.');
    return;
  }

  // ── Loading state ──
  if (btnNlpAnalyze) btnNlpAnalyze.disabled = true;
  setVisible(btnNlpAnalyzeText, false);
  setVisible(btnNlpAnalyzeSpinner, true);
  if (nlpResultsSection) nlpResultsSection.classList.add('hidden');
  if (nlpErrorBox) nlpErrorBox.classList.add('hidden');
  if (nlpRecordStatus) nlpRecordStatus.textContent = 'Uploading passage audio for linguistic analysis…';

  try {
    const formData = new FormData();
    formData.append('file', audioBlob, 'passage.wav');

    const response = await fetch(`${API_BASE_URL}/api/nlp/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      const errText = document.getElementById('nlp-error-text');
      if (errText) {
        errText.innerText = data.detail || "An error occurred. Please try again.";
      }
      const errBanner = document.getElementById('nlp-error-banner');
      if (errBanner) {
        errBanner.classList.remove('hidden');
      }
      
      // Reset UI button state
      if (btnNlpAnalyze) btnNlpAnalyze.disabled = false;
      setVisible(btnNlpAnalyzeText, true);
      setVisible(btnNlpAnalyzeSpinner, false);
      
      return; // Stop success logic from executing
    }

    const result = await response.json();
    console.info('[VoxTriad] NLP analysis complete:', result);

    renderNlpResults(result);
    if (nlpRecordStatus) nlpRecordStatus.textContent = '✓ Linguistic analysis complete.';

  } catch (err) {
    console.error('[VoxTriad] uploadLinguisticAudio failed:', err);
    if (nlpErrorText) nlpErrorText.textContent = err.message;
    if (nlpErrorBox)  nlpErrorBox.classList.remove('hidden');
    if (nlpRecordStatus) nlpRecordStatus.textContent = `Analysis failed: ${err.message}`;
  } finally {
    if (btnNlpAnalyze) btnNlpAnalyze.disabled = false;
    setVisible(btnNlpAnalyzeText, true);
    setVisible(btnNlpAnalyzeSpinner, false);
  }
}

/**
 * Populate the NLP results section from the /api/nlp/analyze JSON response.
 * @param {{ transcript: string, linguistic_analysis: Object }} result
 */
function renderNlpResults(result) {
  const la = result.linguistic_analysis || {};

  // ── Transcript ──
  const nlpTranscriptVal = document.getElementById('nlp-transcript-val');
  if (nlpTranscriptVal) {
    nlpTranscriptVal.textContent = result.transcript || '(no transcript returned)';
  }

  // ── ML Features ──
  const ml = la.ml_features || {};
  const ttr = typeof ml.ttr === 'number' ? ml.ttr : 0;
  const asl = typeof ml.avg_sentence_length === 'number' ? ml.avg_sentence_length : 0;
  const fwr = typeof ml.filler_word_ratio === 'number' ? ml.filler_word_ratio : 0;

  // Type-Token Ratio (0 to 1)
  const ttrBar = document.getElementById('nlp-ttr-bar');
  const ttrLabel = document.getElementById('nlp-ttr-val');
  if (ttrBar) ttrBar.style.width = `${Math.min(100, ttr * 100).toFixed(1)}%`;
  if (ttrLabel) ttrLabel.textContent = ttr.toFixed(2);

  // Avg Sentence Length (Scale ~ 0 to 30)
  const aslBar = document.getElementById('nlp-asl-bar');
  const aslLabel = document.getElementById('nlp-asl-val');
  if (aslBar) aslBar.style.width = `${Math.min(100, (asl / 30) * 100).toFixed(1)}%`;
  if (aslLabel) aslLabel.textContent = asl.toFixed(1);

  // Filler Word Ratio (Scale ~ 0 to 0.15)
  const fwrBar = document.getElementById('nlp-fwr-bar');
  const fwrLabel = document.getElementById('nlp-fwr-val');
  if (fwrBar) fwrBar.style.width = `${Math.min(100, (fwr / 0.15) * 100).toFixed(1)}%`;
  if (fwrLabel) fwrLabel.textContent = fwr.toFixed(3);

  // ── Risk Probability Donut ──
  const risk = typeof la.risk_probability === 'number' ? la.risk_probability : 0;
  const riskPct = (risk * 100).toFixed(0);
  const nlpRiskVal = document.getElementById('nlp-risk-val');
  if (nlpRiskVal) nlpRiskVal.textContent = `${riskPct}%`;
  
  const riskChart = document.getElementById('nlp-risk-chart-circle');
  if (riskChart) {
    const offset = 100 - risk * 100;
    riskChart.style.strokeDashoffset = offset;
    
    riskChart.classList.remove('text-cyan-400', 'text-amber-400', 'text-rose-500');
    if (risk < 0.4) riskChart.classList.add('text-cyan-400');
    else if (risk < 0.65) riskChart.classList.add('text-amber-400');
    else riskChart.classList.add('text-rose-500');
  }

  const linguisticRiskLevel = risk < 0.4 ? 'low' : risk < 0.65 ? 'medium' : 'high';
  updatePatientRisk('linguistic', linguisticRiskLevel);

  // ── Identified Markers ──
  const nlpMarkersList = document.getElementById('nlp-markers-list');
  if (nlpMarkersList) {
    nlpMarkersList.innerHTML = '';
    const markers = Array.isArray(la.identified_markers) ? la.identified_markers : [];
    if (markers.length === 0) {
      nlpMarkersList.innerHTML = '<span class="text-xs text-emerald-400 font-medium">No Parkinsonian markers identified</span>';
    } else {
      markers.forEach(marker => {
        const chip = document.createElement('span');
        chip.className = 'text-xs px-3 py-1 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30 font-medium inline-block mr-2 mb-2';
        chip.textContent = marker;
        nlpMarkersList.appendChild(chip);
      });
    }
  }

  // ── Clinical Summary ──
  const nlpSummarySkeleton = document.getElementById('nlp-summary-skeleton');
  const nlpAiSummary = document.getElementById('nlp-ai-summary');
  if (nlpSummarySkeleton) nlpSummarySkeleton.classList.add('hidden');
  if (nlpAiSummary) {
    nlpAiSummary.textContent = la.clinical_summary || 'No clinical summary returned.';
  }

  // ── Reveal results section ──
  const nlpResultsSection = document.getElementById('nlp-results-section');
  if (nlpResultsSection) {
    nlpResultsSection.classList.remove('hidden');
    nlpResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// =============================================================================
// 13c. VISION ANALYSIS PIPELINE
// =============================================================================

function setVisionRecordingUIActive(active) {
  const faceStateIdle = document.getElementById('face-state-idle');
  const faceStateActive = document.getElementById('face-state-active');
  const visionUploadResult = document.getElementById('vision-upload-result');
  const faceVideoPlayback = document.getElementById('face-video-playback');
  const timerOverlay = document.getElementById('face-state-timer-overlay');

  if (active) {
    if (faceStateIdle) faceStateIdle.classList.add('hidden');
    if (faceStateActive) faceStateActive.classList.remove('hidden');
    if (visionUploadResult) visionUploadResult.classList.add('hidden');
    if (timerOverlay) timerOverlay.classList.remove('hidden');
    if (webcamPreview) webcamPreview.classList.remove('hidden');
    if (faceVideoPlayback) faceVideoPlayback.src = '';
  } else {
    if (faceStateIdle) faceStateIdle.classList.add('hidden');
    if (faceStateActive) faceStateActive.classList.add('hidden');
    if (visionUploadResult) visionUploadResult.classList.remove('hidden');
    if (timerOverlay) timerOverlay.classList.add('hidden');
    if (webcamPreview) webcamPreview.classList.add('hidden');
    
    if (visionRecordedBlob && faceVideoPlayback) {
      faceVideoPlayback.src = URL.createObjectURL(visionRecordedBlob);
    }
  }
}

function resetVisionRecorderUI() {
  visionRecorderState = 'idle';
  visionElapsedSeconds = 0;
  visionRecordedBlob = null;
  visionChunks = [];
  
  const faceStateIdle = document.getElementById('face-state-idle');
  const faceStateActive = document.getElementById('face-state-active');
  const visionUploadResult = document.getElementById('vision-upload-result');
  const timerOverlay = document.getElementById('face-state-timer-overlay');
  const resultsSection = document.getElementById('vision-results-section');
  const visionErrorMessage = document.getElementById('vision-error-message');
  
  if (faceStateIdle) faceStateIdle.classList.remove('hidden');
  if (faceStateActive) faceStateActive.classList.add('hidden');
  if (visionUploadResult) visionUploadResult.classList.add('hidden');
  if (timerOverlay) timerOverlay.classList.add('hidden');
  if (resultsSection) resultsSection.classList.add('hidden');
  
  if (visionErrorMessage) {
    visionErrorMessage.textContent = '';
    visionErrorMessage.classList.add('hidden');
  }
  if (webcamPreview) {
    webcamPreview.classList.remove('hidden');
    webcamPreview.srcObject = null;
  }
  
  if (visionRecordTimer)  visionRecordTimer.textContent = '0:00';
}

async function startVisionRecording() {
  if (visionRecorderState === 'recording') return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: false
    });
    visionActiveStream = stream;
    if (webcamPreview) {
      webcamPreview.srcObject = stream;
    }
    visionChunks = [];

    const mimeType = 'video/webm;codecs=vp8';
    visionMediaRecorder = MediaRecorder.isTypeSupported(mimeType)
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    visionMediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) visionChunks.push(e.data);
    };

    visionMediaRecorder.onstop = () => {
      visionRecordedBlob = new Blob(visionChunks, { type: visionMediaRecorder.mimeType || 'video/webm' });
      visionChunks = [];
      setVisible(visionUploadResult, true);
      visionRecorderState = 'stopped';
      if (visionRecordStatus) visionRecordStatus.textContent = '✓ Recording complete – review preview and click Analyse Video.';
    };

    visionMediaRecorder.start(250);
    visionRecorderState = 'recording';
    setVisionRecordingUIActive(true);
    if (visionRecordStatus) visionRecordStatus.textContent = 'Recording video… Maintain a neutral expression.';

    visionElapsedSeconds = 0;
    if (visionRecordTimer) visionRecordTimer.textContent = '0:00';
    visionTimerInterval = setInterval(() => {
      visionElapsedSeconds++;
      if (visionRecordTimer) visionRecordTimer.textContent = formatTime(visionElapsedSeconds);
    }, 1000);

  } catch (err) {
    console.error('[VoxTriad] Webcam access failed:', err);
    if (visionRecordStatus) visionRecordStatus.textContent = `Webcam access denied: ${err.message}`;
  }
}

function stopVisionRecording() {
  if (visionTimerInterval) { clearInterval(visionTimerInterval); visionTimerInterval = null; }
  if (visionMediaRecorder && visionMediaRecorder.state !== 'inactive') visionMediaRecorder.stop();
  
  if (visionActiveStream) {
    visionActiveStream.getTracks().forEach(track => track.stop());
    visionActiveStream = null;
  }
  
  setVisionRecordingUIActive(false);
}

async function uploadVisionVideo(videoBlob) {
  if (!videoBlob) {
    console.warn('[VoxTriad] uploadVisionVideo called with no blob.');
    return;
  }

  if (btnVisionAnalyze) btnVisionAnalyze.disabled = true;
  setVisible(btnVisionAnalyzeText, false);
  setVisible(btnVisionAnalyzeSpinner, true);
  if (visionResultsSection) visionResultsSection.classList.add('hidden');
  if (visionErrorBox) visionErrorBox.classList.add('hidden');
  if (visionRecordStatus) visionRecordStatus.textContent = 'Uploading video for facial analysis…';

  try {
    const formData = new FormData();
    formData.append('file', videoBlob, 'vision.webm');

    const response = await fetch(`${API_BASE_URL}/api/vision/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.info('[VoxTriad] Vision analysis complete:', result);

    renderVisionResults(result);
    if (visionResultsSection) {
      visionResultsSection.classList.remove('hidden');
      visionResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    if (visionRecordStatus) visionRecordStatus.textContent = '✓ Visual analysis complete.';

  } catch (err) {
    console.error('[VoxTriad] uploadVisionVideo failed:', err);
    if (visionErrorText) visionErrorText.textContent = err.message;
    if (visionErrorBox)  visionErrorBox.classList.remove('hidden');
    if (visionRecordStatus) visionRecordStatus.textContent = `Analysis failed: ${err.message}`;
    
    const visionErrorMessage = document.getElementById('vision-error-message');
    if (visionErrorMessage) {
      visionErrorMessage.textContent = err.message;
      visionErrorMessage.classList.remove('hidden');
    }
  } finally {
    if (btnVisionAnalyze) btnVisionAnalyze.disabled = false;
    setVisible(btnVisionAnalyzeText, true);
    setVisible(btnVisionAnalyzeSpinner, false);
  }
}

function renderVisionResults(result) {
  const metrics = result.metrics || {};
  const varVal = typeof metrics.mouth_movement_variance === 'number' ? metrics.mouth_movement_variance : 0;
  const bpm = typeof metrics.blink_rate_per_minute === 'number' ? metrics.blink_rate_per_minute : 0;
  const symmetry = typeof metrics.smile_symmetry_index === 'number' ? metrics.smile_symmetry_index : 0;
  const hypo = (bpm < 14) || (symmetry < 0.65) || (varVal < 0.002 && bpm < 18);

  const elVar = document.getElementById('vision-var-val');
  const elBlink = document.getElementById('vision-blink-val');
  const elSym = document.getElementById('vision-symmetry-val');
  const elSummary = document.getElementById('face-ai-summary');

  const barVar = document.getElementById('vision-var-bar');
  const barBlink = document.getElementById('vision-blink-bar');
  const barSym = document.getElementById('vision-symmetry-bar');

  if (elVar) elVar.textContent = varVal.toFixed(4);
  if (elBlink) elBlink.textContent = bpm.toFixed(1) + ' bpm';
  if (elSym) elSym.textContent = symmetry.toFixed(2);
  if (elSummary) elSummary.textContent = result.llm_summary || 'No AI summary available.';

  setTimeout(() => {
    if (barVar) barVar.style.width = Math.min((varVal / 0.01) * 100, 100) + '%';
    if (barBlink) barBlink.style.width = Math.min((bpm / 30) * 100, 100) + '%';
    if (barSym) barSym.style.width = (symmetry * 100) + '%';
  }, 50);

  if (visionHypomimiaBadge) {
    if (hypo) {
      visionHypomimiaBadge.textContent = 'Hypomimia Detected';
      visionHypomimiaBadge.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border text-red-400 bg-red-500/10 border-red-500/20';
    } else {
      visionHypomimiaBadge.textContent = 'Normal';
      visionHypomimiaBadge.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
    }
  }

  const videoRiskLevel = hypo ? 'high' : 'low';
  updatePatientRisk('video', videoRiskLevel);
}

// =============================================================================
// 14. INITIALIZATION & EVENT WIRING (DOMContentLoaded)
// =============================================================================

function initApp() {
  console.info('[VoxTriad] Initializing DOM elements and event handlers.');
  try {

  // Bind Auth DOM elements
  authSection        = document.getElementById('auth-section');
  authForm           = document.getElementById('auth-form');
  authEmailInput     = document.getElementById('auth-email');
  authPasswordInput  = document.getElementById('auth-password');
  authError          = document.getElementById('auth-error');
  authCardTitle      = document.getElementById('auth-card-title');
  authCardDesc       = document.getElementById('auth-card-desc');
  toggleAuthPrompt   = document.getElementById('toggle-auth-prompt');
  btnLogin           = document.getElementById('btn-login');
  btnLoginText       = document.getElementById('btn-login-text');
  btnLoginSpinner    = document.getElementById('btn-login-spinner');
  btnToggleRegister  = document.getElementById('btn-toggle-register');
  btnBypassAuth      = document.getElementById('btn-bypass-auth');

  // Bind Navbar DOM elements
  navUserInfo        = document.getElementById('nav-user-info');
  navUserEmail       = document.getElementById('nav-user-email');
  btnLogout          = document.getElementById('btn-logout');

  // Bind Dashboard DOM elements
  dashboardSection   = document.getElementById('dashboard-section');
  dashSessionId      = document.getElementById('dash-session-id');

  // Bind Recorder DOM elements
  btnRecord          = document.getElementById('btn-record');
  btnStop            = document.getElementById('btn-stop');
  iconMic            = document.getElementById('icon-mic');
  iconStop           = document.getElementById('icon-stop');
  recordStatus       = document.getElementById('record-status');
  recordTimer        = document.getElementById('record-timer');
  recordRing1        = document.getElementById('record-ring-1');
  recordRing2        = document.getElementById('record-ring-2');

  // Bind File Browse DOM elements
  btnBrowseFile      = document.getElementById('btn-browse-file');
  audioFileInput     = document.getElementById('audio-file-input');

  // Bind Waveform DOM elements
  waveformIdle       = document.getElementById('waveform-idle');
  waveformBars       = document.getElementById('waveform-bars');

  // Bind Playback + Upload DOM elements
  playbackPanel      = document.getElementById('playback-panel');
  audioPlayback      = document.getElementById('audio-playback');
  btnUpload          = document.getElementById('btn-upload');
  btnUploadText      = document.getElementById('btn-upload-text');
  btnUploadSpinner   = document.getElementById('btn-upload-spinner');
  uploadResult       = document.getElementById('upload-result');
  uploadResultText   = document.getElementById('upload-result-text');

  // Bind API status DOM elements
  apiStatusDot       = document.getElementById('api-status-dot');
  apiStatusText      = document.getElementById('api-status-text');

  // Bind Clinical Results DOM elements
  resultsSection      = document.getElementById('results-section');
  riskRingArc         = document.getElementById('risk-chart-circle');
  riskPctLabel        = document.getElementById('risk-pct-label');
  riskBadge           = document.getElementById('risk-badge');
  barSvc              = document.getElementById('bar-svc');
  barRf               = document.getElementById('bar-rf');
  barGb               = document.getElementById('bar-gb');
  lblSvc              = document.getElementById('lbl-svc');
  lblRf               = document.getElementById('lbl-rf');
  lblGb               = document.getElementById('lbl-gb');
  lblShimmer          = document.getElementById('lbl-shimmer');
  barShimmer          = document.getElementById('bar-shimmer');
  lblJitter           = document.getElementById('lbl-jitter');
  barJitter           = document.getElementById('bar-jitter');
  lblHnr              = document.getElementById('lbl-hnr');
  barHnr              = document.getElementById('bar-hnr');
  uncertaintyBanner   = document.getElementById('uncertainty-banner');
  uncertaintyTitle    = document.getElementById('uncertainty-title');
  uncertaintyDesc     = document.getElementById('uncertainty-desc');
  uncertaintyBar      = document.getElementById('uncertainty-bar');
  uncertaintyPct      = document.getElementById('uncertainty-pct');
  reportRecordingId   = document.getElementById('report-recording-id');
  reportModelUsed     = document.getElementById('report-model-used');
  reportLlmStatus     = document.getElementById('report-llm-status');
  llmSourceBadge      = document.getElementById('llm-source-badge');
  summarySkeleton     = document.getElementById('summary-skeleton');
  summaryText         = document.getElementById('summary-text');
  btnRefreshReport    = document.getElementById('btn-refresh-report');

  // Bind Phase 5: Similarity Search DOM elements
  similarityResultsContainer = document.getElementById('similarity-results-container');
  similarityLoading          = document.getElementById('similarity-loading');
  similarityCountBadge       = document.getElementById('similarity-count-badge');

  // Bind NLP mode tabs
  tabAcoustic    = document.getElementById('tab-acoustic');
  tabLinguistic  = document.getElementById('tab-linguistic');

  // Bind acoustic / linguistic panels (for tab switching)
  // panelAcoustic is the main grid div (first child after the tab bar inside dashboard-section)
  panelAcoustic  = document.querySelector('#dashboard-section .grid.grid-cols-1.lg\\:grid-cols-3');
  panelLinguistic = document.getElementById('panel-linguistic');

  // Bind guide sidebar cards
  guideAcoustic   = document.getElementById('guide-acoustic');
  guideLinguistic  = document.getElementById('guide-linguistic');

  // Bind NLP Recorder elements
  btnNlpRecord         = document.getElementById('btn-nlp-record');
  btnNlpStop           = document.getElementById('btn-nlp-stop');
  nlpIconMic           = document.getElementById('nlp-icon-mic');
  nlpIconStop          = document.getElementById('nlp-icon-stop');
  nlpRecordStatus      = document.getElementById('nlp-record-status');
  nlpRecordTimer       = document.getElementById('nlp-record-timer');
  nlpRecordRing1       = document.getElementById('nlp-record-ring-1');
  nlpRecordRing2       = document.getElementById('nlp-record-ring-2');
  nlpWaveformIdle      = document.getElementById('nlp-waveform-idle');
  nlpWaveformBars      = document.getElementById('nlp-waveform-bars');
  nlpPlaybackPanel     = document.getElementById('nlp-playback-panel');
  nlpAudioPlayback     = document.getElementById('nlp-audio-playback');
  btnNlpAnalyze        = document.getElementById('btn-nlp-analyze');
  btnNlpAnalyzeText    = document.getElementById('btn-nlp-analyze-text');
  btnNlpAnalyzeSpinner = document.getElementById('btn-nlp-analyze-spinner');

  if (nlpAudioPlayback) {
    nlpAudioPlayback.addEventListener('loadedmetadata', () => {
      const duration = nlpAudioPlayback.duration;
      if (isFinite(duration)) {
        const timeEl = document.getElementById('nlp-player-time');
        if (timeEl) {
          const m = Math.floor(duration / 60);
          const s = Math.floor(duration % 60).toString().padStart(2, '0');
          timeEl.textContent = `${m}:${s}`;
        }
      }
    });

    nlpAudioPlayback.addEventListener('timeupdate', () => {
      const duration = nlpAudioPlayback.duration;
      const currentTime = nlpAudioPlayback.currentTime;
      if (isFinite(duration) && duration > 0) {
        const fillEl = document.getElementById('nlp-player-fill');
        if (fillEl) {
          fillEl.style.width = `${(currentTime / duration) * 100}%`;
        }
        const timeEl = document.getElementById('nlp-player-time');
        if (timeEl) {
          const m = Math.floor(currentTime / 60);
          const s = Math.floor(currentTime % 60).toString().padStart(2, '0');
          timeEl.textContent = `${m}:${s}`;
        }
      }
    });

    const updateNlpPlayIcon = (playing) => {
      const btn = document.getElementById('nlp-custom-play-btn');
      if (btn) {
        if (playing) {
          btn.innerHTML = `<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;
        } else {
          btn.innerHTML = `<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`;
        }
      }
    };

    nlpAudioPlayback.addEventListener('play', () => updateNlpPlayIcon(true));
    nlpAudioPlayback.addEventListener('pause', () => updateNlpPlayIcon(false));
    nlpAudioPlayback.addEventListener('ended', () => updateNlpPlayIcon(false));
  }

  // Bind NLP Results elements
  nlpResultsSection = document.getElementById('nlp-results-section');
  nlpLdiBar         = document.getElementById('nlp-ldi-bar');
  nlpLdiLabel       = document.getElementById('nlp-ldi-label');
  nlpSynBar         = document.getElementById('nlp-syn-bar');
  nlpSynLabel       = document.getElementById('nlp-syn-label');
  nlpCohBar         = document.getElementById('nlp-coh-bar');
  nlpCohLabel       = document.getElementById('nlp-coh-label');
  nlpMarkersList    = document.getElementById('nlp-markers-list');
  nlpTranscriptBox  = document.getElementById('nlp-transcript-box');
  nlpSummaryText    = document.getElementById('nlp-summary-text');
  nlpSummarySkeleton = document.getElementById('nlp-summary-skeleton');
  nlpErrorBox       = document.getElementById('nlp-error-box');
  nlpErrorText      = document.getElementById('nlp-error-text');

  // Bind Vision Recorder elements
  tabVisual = document.getElementById('tab-visual');
  panelVisual = document.getElementById('panel-visual');
  guideVisual = document.getElementById('guide-visual');
  webcamPreview = document.getElementById('webcam-preview');
  btnVisionRecord = document.getElementById('btn-vision-record');
  btnVisionStop = document.getElementById('btn-vision-stop');
  visionIconCam = document.getElementById('vision-icon-cam');
  visionIconStop = document.getElementById('vision-icon-stop');
  visionRecordStatus = document.getElementById('vision-record-status');
  visionRecordTimer = document.getElementById('vision-record-timer');
  visionRecordRing1 = document.getElementById('vision-record-ring-1');
  visionRecordRing2 = document.getElementById('vision-record-ring-2');
  btnVisionAnalyze = document.getElementById('btn-vision-analyze');
  btnVisionAnalyzeText = document.getElementById('btn-vision-analyze-text');
  btnVisionAnalyzeSpinner = document.getElementById('btn-vision-analyze-spinner');
  visionUploadResult = document.getElementById('vision-upload-result');

  // Bind Vision Results elements
  visionResultsSection = document.getElementById('vision-results-section');
  visionMarVal = document.getElementById('vision-mar-val');
  visionBlinkVal = document.getElementById('vision-blink-val');
  visionHypomimiaBadge = document.getElementById('vision-hypomimia-badge');
  visionErrorBox = document.getElementById('vision-error-box');
  visionErrorText = document.getElementById('vision-error-text');
  apiStatusDotVision = document.getElementById('api-status-dot-vision');
  apiStatusTextVision = document.getElementById('api-status-text-vision');

  // Render waveform bar elements
  buildWaveformBars();
  buildNlpWaveformBars();

  // Initialise cinematic video background error handlers
  initVideoBackgrounds();

  // ---------------------------------------------------------------------------
  // Wire Event Listeners with e.preventDefault() & e.stopPropagation()
  // ---------------------------------------------------------------------------

  // Auth form submit
  if (authForm) {
    authForm.addEventListener('submit', (e) => {
      e.preventDefault();
      e.stopPropagation();
      submitAuthForm(e);
    });
  }

  // Sign In / Sign Up button click
  if (btnLogin) {
    btnLogin.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      submitAuthForm(e);
    });
  }

  // Toggle between Sign In and Sign Up modes
  if (btnToggleRegister) {
    btnToggleRegister.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      isSignUpMode = !isSignUpMode;
      clearAuthError();
      updateAuthView();
    });
  }

  // One-click Bypass Authentication (Dev Demo)
  if (btnBypassAuth) {
    btnBypassAuth.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.info('[VoxTriad] Bypass Authentication (Dev Demo) activated.');
      onAuthSuccess({
        email: 'demo-clinician@voxtriad.local',
        id: 'demo-dev-user'
      });
    });
  }

  // Sign Out button
  if (btnLogout) {
    btnLogout.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      await handleSignOut();
    });
  }

  // ── Hamburger dropdown menu toggle ─────────────────────────────────────────
  const btnHamburger   = document.getElementById('btn-hamburger');
  const navDrawer      = document.getElementById('nav-drawer');
  const drawerOverlay  = document.getElementById('drawer-overlay');
  const closeMenuBtn   = document.getElementById('close-menu');
  const screenerToggle = document.getElementById('menu-screener-toggle');
  const screenerSub    = document.getElementById('menu-screener-sub');
  const screenerChevron = document.getElementById('screener-chevron');

  function toggleMenu() {
    if (!navDrawer || !drawerOverlay) return;
    const isClosed = navDrawer.classList.contains('-translate-x-full');
    if (isClosed) {
        navDrawer.classList.remove('-translate-x-full');
        drawerOverlay.classList.remove('hidden');
        setTimeout(() => {
            drawerOverlay.classList.remove('opacity-0');
            drawerOverlay.classList.add('opacity-100');
        }, 10);
    } else {
        navDrawer.classList.add('-translate-x-full');
        drawerOverlay.classList.remove('opacity-100');
        drawerOverlay.classList.add('opacity-0');
        setTimeout(() => {
            drawerOverlay.classList.add('hidden');
        }, 300);
    }
  }

  if (btnHamburger) btnHamburger.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleMenu();
  });
  if (closeMenuBtn) closeMenuBtn.addEventListener('click', toggleMenu);
  if (drawerOverlay) drawerOverlay.addEventListener('click', toggleMenu);

  // ── Screener accordion expand / collapse ───────────────────────────────────
  if (screenerToggle && screenerSub) {
    screenerToggle.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isExpanded = !screenerSub.classList.contains('hidden');
      screenerSub.classList.toggle('hidden', isExpanded);
      if (screenerChevron) {
        screenerChevron.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(180deg)';
      }
    });
  }

  // ── Screener sub-item navigation → wire to existing setAnalysisMode ────────
  const menuNavAcoustic   = document.getElementById('menu-nav-acoustic');
  const menuNavLinguistic = document.getElementById('menu-nav-linguistic');
  const menuNavVisual     = document.getElementById('menu-nav-visual');

  function handleMenuNav(mode) {
    setAnalysisMode(mode);
    switchView('dashboard');
    // Close dropdown after navigation
    if (navDropdown) navDropdown.classList.add('hidden');
    if (btnHamburger) btnHamburger.setAttribute('aria-expanded', 'false');
    // Scroll dashboard into view
    if (dashboardSection) dashboardSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  if (menuNavAcoustic)   menuNavAcoustic.addEventListener('click',   (e) => { e.preventDefault(); handleMenuNav('acoustic');   });
  if (menuNavLinguistic) menuNavLinguistic.addEventListener('click', (e) => { e.preventDefault(); handleMenuNav('linguistic'); });
  if (menuNavVisual)     menuNavVisual.addEventListener('click',     (e) => { e.preventDefault(); handleMenuNav('visual');     });

  // ── View Switching Logic ───────────────────────────────────────────────────
  const viewDashboard      = document.getElementById('view-dashboard');
  const viewPatientHistory = document.getElementById('view-patient-history');
  const btnOpenHistory     = document.getElementById('menu-patient-history');

  function switchView(viewName) {
    if (viewName === 'dashboard') {
      if (viewDashboard) viewDashboard.classList.remove('hidden');
      if (viewPatientHistory) viewPatientHistory.classList.add('hidden');
    } else if (viewName === 'history') {
      if (viewPatientHistory) viewPatientHistory.classList.remove('hidden');
      if (viewDashboard) viewDashboard.classList.add('hidden');
    }
  }

  if (btnOpenHistory) {
    btnOpenHistory.addEventListener('click', (e) => {
      e.preventDefault();
      switchView('history');
      // Close the dropdown when navigating
      if (navDropdown) navDropdown.classList.add('hidden');
      if (btnHamburger) btnHamburger.setAttribute('aria-expanded', 'false');
    });
  }

  // Close dropdown on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (navDropdown && !navDropdown.classList.contains('hidden')) {
        navDropdown.classList.add('hidden');
        if (btnHamburger) btnHamburger.setAttribute('aria-expanded', 'false');
      }
    }
  });

  // ── PDF Export buttons ─────────────────────────────────────────────────────
  const btnExportAcousticPdf = document.getElementById('btn-export-acoustic-pdf');
  const btnExportNlpPdf      = document.getElementById('btn-export-nlp-pdf');
  const btnExportVisionPdf   = document.getElementById('btn-export-vision-pdf');

  if (btnExportAcousticPdf) {
    btnExportAcousticPdf.addEventListener('click', (e) => {
      e.preventDefault();
      exportToPDF('results-section', 'Acoustic');
    });
  }
  if (btnExportNlpPdf) {
    btnExportNlpPdf.addEventListener('click', (e) => {
      e.preventDefault();
      exportToPDF('nlp-results-section', 'Linguistic');
    });
  }
  if (btnExportVisionPdf) {
    btnExportVisionPdf.addEventListener('click', (e) => {
      e.preventDefault();
      exportToPDF('vision-results-section', 'Visual');
    });
  }

  // Record button (toggle record / stop)
  if (btnRecord) {
    btnRecord.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();

      let pt = window.currentPatient || ((typeof window.vtGetPatient === 'function') ? window.vtGetPatient() : null);
      if (!pt || (!pt.id && !pt.manual_id)) {
        alert("⚠️ Please enter and confirm the patient details at the top of the screen before starting an analysis.");
        return;
      }

      if (recorderState === 'idle' || recorderState === 'stopped') {
        if (recorderState === 'stopped') resetRecorderUI();
        startRecording();
      } else if (recorderState === 'recording') {
        stopRecording();
      }
    });
  }

  // Stop button
  if (btnStop) {
    btnStop.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      stopRecording();
    });
  }

  // Upload button
  if (btnUpload) {
    btnUpload.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      await uploadAudio();
    });
  }

  // ── Custom Audio Player Listeners ──
  if (audioPlayback) {
    audioPlayback.addEventListener('loadedmetadata', () => {
      const duration = audioPlayback.duration;
      if (isFinite(duration)) {
        const timeEl = document.getElementById('player-time');
        if (timeEl) {
          const m = Math.floor(duration / 60);
          const s = Math.floor(duration % 60).toString().padStart(2, '0');
          timeEl.textContent = `${m}:${s}`;
        }
      }
    });

    audioPlayback.addEventListener('timeupdate', () => {
      const duration = audioPlayback.duration;
      const currentTime = audioPlayback.currentTime;
      if (isFinite(duration) && duration > 0) {
        const fillEl = document.getElementById('player-fill');
        if (fillEl) {
          fillEl.style.width = `${(currentTime / duration) * 100}%`;
        }
        const timeEl = document.getElementById('player-time');
        if (timeEl) {
          const m = Math.floor(currentTime / 60);
          const s = Math.floor(currentTime % 60).toString().padStart(2, '0');
          timeEl.textContent = `${m}:${s}`;
        }
      }
    });

    audioPlayback.addEventListener('play', () => {
      const playIcon = document.getElementById('play-icon');
      const pauseIcon = document.getElementById('pause-icon');
      if (playIcon) playIcon.classList.add('hidden');
      if (pauseIcon) pauseIcon.classList.remove('hidden');
    });

    audioPlayback.addEventListener('pause', () => {
      const playIcon = document.getElementById('play-icon');
      const pauseIcon = document.getElementById('pause-icon');
      if (playIcon) playIcon.classList.remove('hidden');
      if (pauseIcon) pauseIcon.classList.add('hidden');
    });

    audioPlayback.addEventListener('ended', () => {
      const playIcon = document.getElementById('play-icon');
      const pauseIcon = document.getElementById('pause-icon');
      if (playIcon) playIcon.classList.remove('hidden');
      if (pauseIcon) pauseIcon.classList.add('hidden');
    });
  }

  // Browse Audio File button — triggers hidden <input type="file">
  if (btnBrowseFile && audioFileInput) {
    btnBrowseFile.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();

      let pt = window.currentPatient || ((typeof window.vtGetPatient === 'function') ? window.vtGetPatient() : null);
      if (!pt || (!pt.id && !pt.manual_id)) {
        alert("⚠️ Please enter and confirm the patient details at the top of the screen before starting an analysis.");
        return;
      }

      audioFileInput.click();
    });

    // When a file is chosen from the OS picker
    audioFileInput.addEventListener('change', async (e) => {
      const errBanner = document.getElementById('audio-error-banner');
      if (errBanner) errBanner.classList.add('hidden');

      const file = e.target.files[0];
      if (!file) return;

      // Show converting status immediately
      if (recordStatus) {
        recordStatus.textContent = `Converting "${file.name}" to WAV…`;
      }

      // Reset any stale results
      setVisible(uploadResult, false);
      setVisible(resultsSection, false);

      try {
        // Decode the raw audio bytes into a Web Audio AudioBuffer
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Re-encode as WAV using the existing helper
        const wavBlob = audioBufferToWav(audioBuffer);

        // Assign the WAV blob — this is what uploadAudio() will send to FastAPI
        recordedBlob = wavBlob;

        // Show the playback panel with the converted WAV as preview source
        if (audioPlayback) {
          audioPlayback.src = URL.createObjectURL(wavBlob);
        }
        setVisible(playbackPanel, true);

        // Update status
        if (recordStatus) {
          recordStatus.textContent = `Ready: "${file.name}" (converted to WAV) — click Upload for Analysis.`;
        }

        // Ensure the upload button is active
        if (btnUpload) btnUpload.disabled = false;

        console.info('[VoxTriad] File converted to WAV:', file.name,
          `original ${(file.size / 1024).toFixed(1)} KB → WAV ${(wavBlob.size / 1024).toFixed(1)} KB`);

      } catch (err) {
        console.error('[VoxTriad] File conversion failed:', err);
        if (recordStatus) {
          recordStatus.textContent = `⚠ Could not decode "${file.name}". Please choose a valid audio file.`;
        }
      }

      // Reset so the same file can be reselected without issue
      audioFileInput.value = '';
    });
  }

  // Re-analyse button
  if (btnRefreshReport) {
    btnRefreshReport.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!_lastRecordingId) return;
      btnRefreshReport.disabled = true;
      if (recordStatus) recordStatus.textContent = 'Regenerating clinical summary & cohort similarity…';
      await Promise.allSettled([
        fetchClinicalReport(_lastRecordingId, {}, true),
        fetchSimilarRecordings(_lastRecordingId),
      ]);
      btnRefreshReport.disabled = false;
    });
  }

  // ── Mode Tab switcher ───────────────────────────────────────────────────────
  if (tabAcoustic) {
    tabAcoustic.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      setAnalysisMode('acoustic');
    });
  }
  if (tabLinguistic) {
    tabLinguistic.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      setAnalysisMode('linguistic');
    });
  }

  // ── NLP Recorder controls ───────────────────────────────────────────────────
  if (btnNlpRecord) {
    btnNlpRecord.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (nlpRecorderState === 'idle' || nlpRecorderState === 'stopped') {
        if (nlpRecorderState === 'stopped') resetNlpRecorderUI();
        startNlpRecording();
      } else if (nlpRecorderState === 'recording') {
        stopNlpRecording();
      }
    });
  }
  if (btnNlpStop) {
    btnNlpStop.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      stopNlpRecording();
    });
  }
  if (btnNlpAnalyze) {
    btnNlpAnalyze.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!nlpRecordedBlob) {
        if (nlpRecordStatus) nlpRecordStatus.textContent = 'Please record the passage before analysing.';
        return;
      }
      await uploadLinguisticAudio(nlpRecordedBlob);
    });
  }

  const btnNlpUploadProxy = document.getElementById('btn-nlp-upload-proxy');
  const nlpFileUpload = document.getElementById('nlp-file-upload');
  const btnNlpRerecord = document.getElementById('btn-nlp-rerecord');

  if (btnNlpUploadProxy && nlpFileUpload) {
    btnNlpUploadProxy.addEventListener('click', () => {
      nlpFileUpload.click();
    });
    
    nlpFileUpload.addEventListener('change', (e) => {
      const errorBanner = document.getElementById('nlp-error-banner');
      if (errorBanner) {
        errorBanner.classList.add('hidden');
      }

      const file = e.target.files[0];
      if (!file) return;
      
      nlpRecordedBlob = file;
      if (nlpAudioPlayback) {
        nlpAudioPlayback.src = URL.createObjectURL(file);
      }
      
      // Stop recording if active and reset UI state to simulate a successful recording
      if (nlpRecorderState === 'recording') {
        stopNlpRecording();
      }
      nlpRecorderState = 'stopped';
      setNlpRecordingUIActive(false);
      
      // Ensure the correct containers are visible
      const stateIdle = document.getElementById('ling-state-idle');
      const stateActive = document.getElementById('ling-state-active');
      if (stateIdle && stateActive) {
        stateIdle.classList.add('hidden');
        stateActive.classList.add('hidden');
        stateActive.classList.remove('flex');
      }
      
      // Hide active components
      setVisible(nlpRecordTimer, false);
      const recIndicator = document.getElementById('nlp-rec-indicator');
      if (recIndicator) recIndicator.classList.add('hidden');
      const listeningText = document.getElementById('nlp-listening-text');
      if (listeningText) listeningText.classList.add('hidden');
      const stopText = document.getElementById('nlp-stop-text');
      if (stopText) stopText.classList.add('hidden');
      if (btnNlpStop) btnNlpStop.classList.add('hidden');
      
      // Show playback panel
      if (nlpPlaybackPanel) {
        nlpPlaybackPanel.classList.remove('hidden');
        nlpPlaybackPanel.classList.add('flex');
      }
    });
  }

  if (btnNlpRerecord) {
    btnNlpRerecord.addEventListener('click', (e) => {
      e.preventDefault();
      resetNlpRecorderUI();
    });
  }

  // ── Vision Recorder controls ───────────────────────────────────────────────────
  if (tabVisual) {
    tabVisual.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      setAnalysisMode('visual');
    });
  }

  if (btnVisionRecord) {
    btnVisionRecord.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();

      let pt = window.currentPatient || ((typeof window.vtGetPatient === 'function') ? window.vtGetPatient() : null);
      if (!pt || (!pt.id && !pt.manual_id)) {
        alert("⚠️ Please enter and confirm the patient details at the top of the screen before starting an analysis.");
        return;
      }

      if (visionRecorderState === 'idle' || visionRecorderState === 'stopped') {
        if (visionRecorderState === 'stopped') resetVisionRecorderUI();
        startVisionRecording();
      } else if (visionRecorderState === 'recording') {
        stopVisionRecording();
      }
    });
  }
  if (btnVisionStop) {
    btnVisionStop.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      stopVisionRecording();
    });
  }
  if (btnVisionAnalyze) {
    btnVisionAnalyze.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!visionRecordedBlob) {
        if (visionRecordStatus) visionRecordStatus.textContent = 'Please record a video before analysing.';
        return;
      }
      await uploadVisionVideo(visionRecordedBlob);
    });
  }
  const btnVisionRerecord = document.getElementById('btn-vision-rerecord');
  if (btnVisionRerecord) {
    btnVisionRerecord.addEventListener('click', () => {
      resetVisionRecorderUI();
      startVisionRecording();
    });
  }

  // ── Patient Banner Edit Mode ───────────────────────────────────────────────────
  const btnEditPatient = document.getElementById('btn-edit-patient');
  const btnSavePatient = document.getElementById('btn-save-patient');
  const patientViewMode = document.getElementById('patient-view-mode');
  const patientEditMode = document.getElementById('patient-edit-mode');
  
  const nameDisplay = document.getElementById('patient-name-display');
  const ageDisplay = document.getElementById('patient-age-display');
  const genderDisplay = document.getElementById('patient-gender-display');
  const manualIdDisplay = document.getElementById('patient-manual-id-display');
  
  const nameInput = document.getElementById('patient-name-input');
  const ageInput = document.getElementById('patient-age-input');
  const genderInput = document.getElementById('patient-gender-input');
  const manualIdInput = document.getElementById('patient-manual-id');

  if (btnEditPatient && patientViewMode && patientEditMode) {
    btnEditPatient.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      let pt = window.currentPatient || ((typeof window.vtGetPatient === 'function') ? window.vtGetPatient() : null);
      if (pt) {
        if (manualIdInput) manualIdInput.value = pt.manual_id || '';
        nameInput.value = pt.fullName || '';
        ageInput.value = pt.age || '';
        // Title case to match option values or lower case
        const mappedSex = (pt.sex || 'Other').toLowerCase();
        Array.from(genderInput.options).forEach(opt => {
          if (opt.value.toLowerCase() === mappedSex) opt.selected = true;
        });
      }
      
      patientViewMode.classList.add('hidden');
      patientEditMode.classList.remove('hidden');
    });
  }

  if (btnSavePatient && patientViewMode && patientEditMode) {
    btnSavePatient.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();

      const originalHTML = btnSavePatient.innerHTML;
      btnSavePatient.disabled = true;

      try {
        let pt = window.currentPatient || ((typeof window.vtGetPatient === 'function') ? window.vtGetPatient() : null);
        if (!pt) {
          pt = {};
        }
        
        const manualId = manualIdInput ? manualIdInput.value.trim() : (pt.manual_id || '');
        const fullName = nameInput.value.trim() || pt.fullName;
        const age = parseInt(ageInput.value, 10) || pt.age;
        const sex = genderInput.options[genderInput.selectedIndex].value.toLowerCase();

        console.log("Save button clicked", { manual_id: manualId, name: fullName, age, sex });

        pt.manual_id = manualId;
        pt.fullName = fullName;
        pt.age = age;
        pt.sex = sex;
        
        // Update displays
        if (manualIdDisplay) manualIdDisplay.textContent = pt.manual_id;
        if (nameDisplay) nameDisplay.textContent = pt.fullName;
        if (ageDisplay) ageDisplay.textContent = `Age ${pt.age}`;
        if (genderDisplay) genderDisplay.textContent = pt.sex;

        // Visual feedback: change icon to checkmark temporarily
        btnSavePatient.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        
        // Determine if this is a new patient or an existing one
        const existingId = pt.id || localStorage.getItem('currentPatientId');

        if (existingId) {
          // ── EXISTING patient → PUT update ──────────────────────────────
          const payload = {
            patient_id: existingId,
            manual_id: manualId,
            full_name: fullName,
            age: age,
            sex: sex
          };

          const response = await fetch(`${API_BASE_URL}/api/patients/update`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.detail || 'Unknown error';
            alert(`Backend rejected: ${errMsg}`);
            throw new Error(`Update failed: ${errMsg}`);
          }
          console.info('[VoxTriad] Patient successfully updated on backend');
        } else {
          // ── NEW patient → POST create ─────────────────────────────────
          let auth_id = '';
          try {
              if (window.supabaseClient) {
                  const { data: sessionData } = await window.supabaseClient.auth.getSession();
                  if (sessionData?.session?.user) {
                      auth_id = sessionData.session.user.id;
                  }
              }
          } catch (e) {
              console.warn('Could not get session for auth_id', e);
          }

          const payload = {
            manual_id: manualId,
            full_name: fullName,
            age: age,
            sex: sex
          };
          if (auth_id) payload.auth_id = auth_id;

          const response = await fetch(`${API_BASE_URL}/api/patients/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.detail || 'Unknown error';
            alert(`Backend rejected: ${errMsg}`);
            throw new Error(`Create failed: ${errMsg}`);
          }
          const result = await response.json();
          const newId = result.patient?.id;
          if (newId) {
            pt.id = newId;
            localStorage.setItem('currentPatientId', newId);
            console.info('[VoxTriad] New patient created on backend, ID:', newId);
          } else {
            throw new Error("Backend did not return a valid patient ID.");
          }
        }

        // Update session storage (includes the new pt.id if just created)
        try {
          sessionStorage.setItem('vt_patient', JSON.stringify(pt));
        } catch (err) {
          console.warn('[VoxTriad] Could not save patient to sessionStorage', err);
        }
        
        // Expose globally so guards can verify the updated state (Task 1)
        window.currentPatient = pt;
        
        // Transition back to view mode after a short delay so user sees checkmark
        setTimeout(() => {
          patientEditMode.classList.add('hidden');
          patientViewMode.classList.remove('hidden');
          
          // Enable capture buttons after save
          if (btnRecord) btnRecord.disabled = false;
          if (btnBrowseFile) btnBrowseFile.disabled = false;
          if (btnVisionRecord) btnVisionRecord.disabled = false;
        }, 700);

      } catch (error) {
        console.error("Save process crashed:", error);
        alert("Error: " + error.message);
      } finally {
        setTimeout(() => {
          btnSavePatient.innerHTML = originalHTML;
          btnSavePatient.disabled = false;
        }, 700);
      }
    });
  }

  // Listen to Supabase auth state changes
  try {
    if (supabaseClient && supabaseClient.auth) {
      supabaseClient.auth.onAuthStateChange((event, session) => {
        console.info(`[VoxTriad] Auth state event: ${event}`);
        if (event === 'SIGNED_IN' && session?.user) {
          onAuthSuccess(session.user);
        } else if (event === 'SIGNED_OUT') {
          window.location.replace('login.html');
        }
      });
    }
  } catch (err) {
    console.warn('[VoxTriad] onAuthStateChange subscription failed:', err);
  }

  // Periodic API health check
  checkApiHealth();
  setInterval(checkApiHealth, 15000);

  // Step 3: Secure index.html – verify active Supabase session on load
  (async function checkExistingSession() {
    const authStatus = document.getElementById('auth-status');
    const authStatusText = document.getElementById('auth-status-text');

    const redirectToLogin = () => {
      if (authStatusText) authStatusText.textContent = 'Redirecting to login portal…';
      try {
        // Use relative path login.html rather than root-relative /login.html
        window.location.replace('login.html');
      } catch {
        window.location.href = 'login.html';
      }
    };

    try {
      const client = (typeof window !== 'undefined' && (window.supabaseClient || window.supabaseApp))
        ? (window.supabaseClient || window.supabaseApp)
        : supabaseClient;

      if (!client || !client.auth || typeof client.auth.getSession !== 'function') {
        console.warn('[VoxTriad] Supabase client is not available. Redirecting to login.html.');
        redirectToLogin();
        return;
      }

      // Enforce a 2500ms timeout on getSession so the page never hangs indefinitely
      const sessionPromise = client.auth.getSession();
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Session verification timed out')), 2500)
      );

      const result = await Promise.race([sessionPromise, timeoutPromise]).catch(err => ({
        data: { session: null },
        error: err
      }));

      const session = result?.data?.session;

      if (result?.error || !session || !session.user) {
        console.warn('[VoxTriad] No active session found or error:', result?.error);
        redirectToLogin();
        return;
      }

      console.info('[VoxTriad] Active session verified for user:', session.user.email);
      if (authStatus) authStatus.classList.add('hidden');
      onAuthSuccess(session.user);
    } catch (err) {
      console.error('[VoxTriad] Error verifying session, redirecting to login.html:', err);
      redirectToLogin();
    }
  })();
  } catch (fatalErr) {
    console.error('[VoxTriad] Fatal initialization error in initApp:', fatalErr);
    try {
      window.location.replace('login.html');
    } catch {
      window.location.href = 'login.html';
    }
  }
  
  // Disable capture buttons until a patient is saved
  if (btnRecord) btnRecord.disabled = true;
  if (btnBrowseFile) btnBrowseFile.disabled = true;
  if (btnVisionRecord) btnVisionRecord.disabled = true;

  // Initialize Lucide icons on page load
  if (window.lucide && window.lucide.createIcons) {
    window.lucide.createIcons();
  }
}

// Execute initApp on DOMContentLoaded or immediately if DOM is already ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// =============================================================================
// 15. WAV ENCODER (Web Audio API buffer -> pure 16-bit PCM audio/wav)
// =============================================================================

function audioBufferToWav(buffer) {
    let numOfChan = buffer.numberOfChannels,
        length = buffer.length * numOfChan * 2 + 44,
        bufferArray = new ArrayBuffer(length),
        view = new DataView(bufferArray),
        channels = [], i, sample, offset = 0, pos = 0;
    function setUint16(data) { view.setUint16(pos, data, true); pos += 2; }
    function setUint32(data) { view.setUint32(pos, data, true); pos += 4; }
    setUint32(0x46464952); setUint32(length - 8); setUint32(0x45564157);
    setUint32(0x20746d66); setUint32(16); setUint16(1); setUint16(numOfChan);
    setUint32(buffer.sampleRate); setUint32(buffer.sampleRate * 2 * numOfChan);
    setUint16(numOfChan * 2); setUint16(16); setUint32(0x61746164); setUint32(length - pos - 4);
    for (i = 0; i < buffer.numberOfChannels; i++) channels.push(buffer.getChannelData(i));
    while (pos < length) {
        for (i = 0; i < numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            view.setInt16(pos, sample, true); pos += 2;
        }
        offset++;
    }
    return new Blob([bufferArray], { type: "audio/wav" });
}

