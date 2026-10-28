/**
 * VoxTriad – Supabase Client Configuration
 *
 * Central configuration file for Supabase authentication and database connection.
 * Included by login.html, screening.html, and dashboard.html.
 */

(function () {
  'use strict';

  // =============================================================================
  // SUPABASE CREDENTIALS
  // =============================================================================

  // ─── IMPORTANT ────────────────────────────────────────────────────────────
  // SUPABASE_URL and SUPABASE_ANON_KEY are public-safe (anon/publishable keys).
  // The service_role key is NEVER stored here - only in backend/.env (gitignored).
  // ──────────────────────────────────────────────────────────────────────────
  const SUPABASE_URL = 'https://eyjehbjsttlaqqjaqtdj.supabase.co';
  const SUPABASE_ANON_KEY = 'sb_publishable_L5YKKJYqj0UbUIQda3PvxQ_Cy0IMxwF';

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
      typeof window.supabase.createClient === 'function'
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
      console.warn('[VoxTriad] window.supabase or createClient not available. Using fallback mock client.');
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
})();

// Global variable for inline scripts that reference bare `supabase`
var supabase = window.supabaseClient;
