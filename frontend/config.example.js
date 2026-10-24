/**
 * VoxTriad – Frontend Configuration Template
 * 
 * Copy this file to `frontend/config.js` for local development:
 *   cp frontend/config.example.js frontend/config.js
 */

window.SUPABASE_URL = 'https://eyjehbjsttlaqqjaqtdj.supabase.co';
window.SUPABASE_ANON_KEY = 'sb_publishable_L5YKKJYqj0UbUIQda3PvxQ_Cy0IMxwF';

// =============================================================================
// CLIENT INITIALIZATION (Fallback / Standalone Mode)
// =============================================================================
(function () {
  'use strict';
  if (
    typeof window !== 'undefined' &&
    window.supabase &&
    typeof window.supabase.createClient === 'function'
  ) {
    try {
      var client = window.supabase.createClient(
        window.SUPABASE_URL,
        window.SUPABASE_ANON_KEY,
        {
          auth: {
            persistSession: true,
            autoRefreshToken: true,
            detectSessionInUrl: true,
          },
        }
      );
      window.supabaseClient = client;
      window.supabaseApp = client;
      if (typeof window.supabase === 'object') {
        Object.assign(window.supabase, client);
      }
    } catch (err) {
      console.warn('[VoxTriad] Supabase initialization error:', err);
    }
  }
})();

var supabase = window.supabaseClient;
