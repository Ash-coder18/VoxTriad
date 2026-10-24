/**
 * VoxTriad – Supabase Client Configuration
 *
 * Central configuration file for Supabase authentication and database connection.
 * Included by both login.html and index.html.
 */

(function () {
  'use strict';

  // =============================================================================
  // SUPABASE CREDENTIALS
  // Resolved dynamically from window or backend /api/config (no hardcoded secrets)
  // =============================================================================

  let SUPABASE_URL = (typeof window !== 'undefined' && window.SUPABASE_URL) ? window.SUPABASE_URL : '';
  let SUPABASE_ANON_KEY = (typeof window !== 'undefined' && window.SUPABASE_ANON_KEY) ? window.SUPABASE_ANON_KEY : '';

  // Expose URL & Key on window for components that reference them
  window.SUPABASE_URL = SUPABASE_URL;
  window.SUPABASE_ANON_KEY = SUPABASE_ANON_KEY;

  // =============================================================================
  // CLIENT INITIALIZATION
  // =============================================================================

  /**
   * Global Supabase Client instance.
   * The official CDN exposes window.supabase with createClient function.
   */
  let clientInstance = null;

  try {
    if (
      typeof window !== 'undefined' &&
      window.supabase &&
      typeof window.supabase.createClient === 'function' &&
      SUPABASE_URL &&
      SUPABASE_ANON_KEY
    ) {
      clientInstance = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: true,
        },
      });
      console.info('[VoxTriad] Supabase client successfully initialized via config.js');
    } else {
      console.warn('[VoxTriad] Credentials or window.supabase not available. Using fallback mock client.');
    }
  } catch (err) {
    console.error('[VoxTriad] Exception while creating Supabase client in config.js:', err);
  }

  // Bulletproof fallback mock object if initialization fails or window.supabase is missing
  if (!clientInstance) {
    console.warn('[VoxTriad] Using safe bulletproof fallback supabase client object.');
    clientInstance = {
      auth: {
        getSession: async () => ({ data: { session: null }, error: new Error('Supabase not loaded') }),
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
        signOut: async () => ({ error: null }),
        signInWithPassword: async () => ({ data: { session: null }, error: new Error('Supabase client not initialized. Check credentials in config.js.') }),
        signUp: async () => ({ data: { session: null }, error: new Error('Supabase client not initialized. Check credentials in config.js.') }),
        signInWithOAuth: async () => ({ data: { session: null }, error: new Error('Supabase client not initialized. Check credentials in config.js.') })
      }
    };
  }

  // Expose client globally across the window scope
  window.supabaseClient = clientInstance;
  window.supabaseApp = clientInstance;

  // Expose `supabase` on window while preserving `createClient` if present
  if (typeof window !== 'undefined') {
    if (window.supabase && typeof window.supabase === 'object') {
      Object.assign(window.supabase, clientInstance);
    } else {
      window.supabase = clientInstance;
    }
  }

  // Dynamic runtime bootstrap from backend /api/config
  const apiBase = (typeof window !== 'undefined' && window.API_BASE_URL)
    ? window.API_BASE_URL
    : (typeof window !== 'undefined' && window.location.hostname === '127.0.0.1')
      ? 'http://127.0.0.1:8000'
      : 'http://localhost:8000';

  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    fetch(`${apiBase}/api/config`)
      .then((res) => (res.ok ? res.json() : null))
      .then((cfg) => {
        if (cfg && cfg.supabase_url && cfg.supabase_anon_key && window.supabase && typeof window.supabase.createClient === 'function') {
          const liveClient = window.supabase.createClient(cfg.supabase_url, cfg.supabase_anon_key, {
            auth: {
              persistSession: true,
              autoRefreshToken: true,
              detectSessionInUrl: true,
            },
          });
          window.SUPABASE_URL = cfg.supabase_url;
          window.SUPABASE_ANON_KEY = cfg.supabase_anon_key;
          window.supabaseClient = liveClient;
          window.supabaseApp = liveClient;
          if (window.supabase) Object.assign(window.supabase, liveClient);
          console.info('[VoxTriad] Supabase client successfully initialized dynamically from /api/config');
        }
      })
      .catch(() => {});
  }
})();

// Global variable for inline scripts that reference bare `supabase`
var supabase = window.supabaseClient;
