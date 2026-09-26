---
version: 1
slug: "static-index-html"
primary_target: "static/index.html"
related_targets: ["static/styles.css","static/script.js"]
---

# LiveChord app surface

Scope: the whole single-page app in `static/` (search, results, song overview,
play-along, how it works). Mode: Operate. Visitors: recruiters arriving from
LinkedIn who judge the build in a minute or two, and guitarists practising;
desktop and phone equally. Task: one click from landing to chords on screen,
then play along with the mic. Proof: the working demo and its five
pre-analysed presets; no invented claims, no unverified accuracy figure.
Constraints: keep every element ID and behaviour `script.js` relies on; plain
HTML/CSS/JS with no build step; controls stay familiar (field, buttons,
toggles), never knobs as inputs.

## Direction contract

THESIS: LiveChord is the practice amp you learn songs through: every control
lives on one silkscreened aluminium panel. It refuses the category default, a
dark app with a green accent, rounded cards and album-art tiles.

OWN-WORLD: Brushed aluminium panel #D8DCDF with silkscreen ink #14171C legends
in expanded caps; black tolex #1C1C1E frame with white piping; sparkle grille
cloth #A9BDBE field; jewel-red #D9342B lamps and LEDs are the only
indicators; black push keys; a bat toggle for the mic; printed 0-10 scales
with pointers for meters. Palette law: match green #2FB36A lights only when
your chord matches the target, nowhere else. One display face (Zilla Slab,
picked by the user in a live typeset round on 2026-09-26; Source Sans 3 sets
reading copy and numbers) prints headings, legends, key legends and chord
names, with chord names on a strict ladder: Now > Next / You / Target >
chord plates > progression > timeline.

STORY: The visitor sees an amp front and reads "Hear a song. See its chords.
Play along." They press a numbered preset, its LED blinks while three lamps
count the analysis, and the preset expands in place into the song's chords
with guitar shapes. Play along turns the same panel into a player; the mic
switch goes from STANDBY to LISTEN and the green lamp lights when they land
the chord. They leave believing the author ships things that work.

FIRST VIEWPORT: Tolex band with nav links. Full-width aluminium panel: the
LiveChord wordmark left, the state strip (1 Pick a song, 2 See its chords,
3 Play along) with lamps, the jewel power lamp and server status right; H1
and a one-sentence lede; the INPUT row (search field plus SEARCH key) is the
primary action; the PRESETS row of five numbered keys with LEDs, song over
artist. Grille cloth shows below the panel.

SIGNATURE INTERACTION AND MOTION: preset or result pressed, its LED blinks,
analysis lamps light in sequence with elapsed time, then the picked song
morphs (View Transitions) into the song panel, and again into the player
strip. Motion is mechanical and damped: keys depress 2px, lamps warm on over
about 180ms, the mic lever snaps, pointers ease like meter needles; only
lamps that mean "working" blink. Reduced motion: instant state changes.

FORM: Practice Amp (1960s-70s combo amp faceplate), position 6 of the ordered
list, seed key bde1d201. Raised by arcade (palette law), specimen (chord-name
size ladder) and miura (one song object through named states).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved

- A dark "Blackface" variant (black panel, white silkscreen) for dark-mode
  phones: build only if it reaches the light panel's finish.
- The recall figure noted in `app.py` stays off the page until re-checked.
