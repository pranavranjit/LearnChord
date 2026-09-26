# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: recruiters and hiring managers arriving from Pranav's LinkedIn post or
portfolio site to judge his machine-learning and audio-engineering ability.
They have a minute or two, may not play an instrument, and judge the work by
trying it: the tool itself is the live demo.

## Product Purpose

LiveChord takes any song, finds its chord progression from a 30-second preview,
lays it out on a timeline with guitar chord diagrams, and then listens through
the microphone while you play along, telling you whether you are on the right
chord. It exists as a public, working demonstration that Pranav can take an
audio-ML idea all the way to a deployed product people can use.

Success: a visitor gets from landing to chords on screen in one click and a few
seconds, understands in plain language what the app does, and leaves convinced
that its author builds things that work.

## Positioning

Chords are computed from the audio itself in seconds, not looked up in a chord
database, and the same engine then listens to the player live and scores the
match. Analysis, the real-time microphone path, the interface and the free-tier
deployment were all built by one person.

## Operating Context

- Visitors arrive through links in a LinkedIn post and the portfolio site's
  project card, landing on the Hugging Face Space
  (https://sunny2802-live-chord-ai.hf.space). A free Space sleeps after long
  inactivity, so the first visit can include a cold start. Render is a second
  deployment target (`render.yaml`).
- Desktop and phone browsers, including LinkedIn's in-app browser.
- Practising needs microphone permission over HTTPS, and headphones so the mic
  hears the instrument rather than the song.

## Capabilities and Constraints

- Search uses the iTunes Search API (no key); audio is Apple's 30-second
  preview clip, not the full song. YouTube and Jamendo backends exist behind
  `SEARCH_BACKEND` but are not the default.
- Chord vocabulary is the 24 major and minor triads; no sevenths, suspended,
  diminished or augmented chords.
- Guitar only: every chord is shown as a standard guitar voicing.
- Views: search with one-click example songs, song overview (chords with
  diagrams, progression), play-along timeline with Now/Next readouts, and
  practice in two modes (follow the song, drill one chord) scored by
  on-target share and best streak.
- Analyses run one at a time to fit a 512 MB host; results are cached for about
  an hour, and the example songs are pre-analysed at boot.
- Served by FastAPI; the interface is plain HTML, CSS and JavaScript with no
  build step (`static/`).

## Brand Commitments

- The product is called **LiveChord**. "Live Chord AI" and "LearnChord" are
  retired from user-facing copy; the GitHub repository keeps the LearnChord
  name.
- Keep the model story high-level: plain-language description of what happens,
  no model details, and no mention of the retired transformer experiment.
- Credit Pranav, with links to LinkedIn and the source on GitHub.

## Evidence on Hand

- The working app, with pre-analysed example songs: Let It Be, Wonderwall,
  Riptide, Perfect and Someone Like You.
- `static/og-image.png`, the link-preview image.
- Unverified: a note in `app.py` records that, across six songs with known
  chords, the share of true chords found rose from 81% to 100% when the
  analysis window was shortened to one second (chords found, not timing). The
  user will re-check it; do not show it until then.
- No testimonials, usage numbers, press or comparisons with other tools exist.
  Do not invent them.

## Product Principles

1. The demo is the proof: one click to a real result, with no setup and no
   sign-up.
2. Plain language over model detail, and nothing that overclaims: no "AI"
   hype, no unverified accuracy.
3. Be honest about limits: 30-second previews, major and minor chords only,
   and what the microphone needs.
4. Guitar-first: every chord on screen is playable, with its diagram.
