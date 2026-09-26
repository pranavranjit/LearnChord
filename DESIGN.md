---
name: LiveChord
description: A practice-amp faceplate for hearing a song, seeing its chords and playing along.
colors:
  aluminium: "#D8DCDF"
  aluminium-highlight: "#EEF0F1"
  aluminium-shadow: "#C3C9CD"
  tolex: "#1C1C1E"
  piping: "#ECE9E1"
  grille-cloth: "#A9BDBE"
  silkscreen-ink: "#14171C"
  ink-secondary: "#3E444C"
  ink-tertiary: "#525961"
  rule: "rgba(20, 23, 28, 0.32)"
  rule-soft: "rgba(20, 23, 28, 0.14)"
  wash: "rgba(20, 23, 28, 0.06)"
  key-face: "#1E2228"
  key-top: "#2F353D"
  key-skirt: "#090B0D"
  key-ink: "#F1F3F4"
  key-ink-muted: "#AEB6BC"
  field: "#F7F8F8"
  field-hint: "#596068"
  jewel-red: "#D9342B"
  jewel-hot: "#FF6A55"
  jewel-off: "#5E1D18"
  match-green: "#2FB36A"
  match-hot: "#8AF5B0"
  major-chip: "#F3F4F5"
  minor-chip-ink: "#EEF1F2"
  plate-ink: "#EDEFF0"
  plate-ink-muted: "#B4BCC2"
typography:
  display:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "clamp(2.1rem, 1.5rem + 2.3vw, 3.2rem)"
    fontWeight: 700
    lineHeight: 1.04
    letterSpacing: "-0.012em"
  headline:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "clamp(1.75rem, 1.25rem + 1.8vw, 2.75rem)"
    fontWeight: 700
    lineHeight: 1.04
    letterSpacing: "-0.015em"
  title:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "clamp(1.625rem, 1.2rem + 1.4vw, 2.25rem)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.02em"
  preset-title:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "1.2rem"
    fontWeight: 700
    lineHeight: 1.15
  chord-rung-1:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "clamp(4rem, 2.9rem + 4.4vw, 6rem)"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.01em"
  chord-rung-2:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "clamp(2.25rem, 1.8rem + 1.6vw, 3rem)"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.01em"
  chord-rung-3:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "1.875rem"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.01em"
  chord-rung-4:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.01em"
  chord-rung-5:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "0.8125rem"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.01em"
  body:
    fontFamily: "'Source Sans 3', 'Helvetica Neue', Helvetica, Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  lede:
    fontFamily: "'Source Sans 3', 'Helvetica Neue', Helvetica, Arial, sans-serif"
    fontSize: "clamp(1.0625rem, 1rem + 0.3vw, 1.1875rem)"
    fontWeight: 400
    lineHeight: 1.55
  numeral:
    fontFamily: "'Source Sans 3', 'Helvetica Neue', Helvetica, Arial, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 700
    letterSpacing: "0.04em"
    fontFeature: "'tnum'"
  label:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "0.8125rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.12em"
  key-legend:
    fontFamily: "'Zilla Slab', Rockwell, 'Roboto Slab', Georgia, serif"
    fontSize: "0.9375rem"
    fontWeight: 600
    letterSpacing: "0.1em"
rounded:
  plate: "2px"
  chip: "3px"
  control: "4px"
  key: "5px"
  well: "6px"
  cabinet: "12px"
  round: "50%"
spacing:
  side: "clamp(6px, 1.6vw, 24px)"
  gutter: "clamp(16px, 4vw, 52px)"
  content: "1180px"
  bank-gap: "12px"
  toggle-gap: "10px"
  group: "clamp(28px, 4.5vw, 44px)"
components:
  key:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.key-ink}"
    typography: "{typography.key-legend}"
    rounded: "{rounded.key}"
    padding: "0 1.6rem 3px"
    height: "58px"
  key-hover:
    backgroundColor: "#262B32"
  key-latched:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.key-ink}"
  key-disabled:
    backgroundColor: "#4E535B"
    textColor: "#D2D6D9"
  key-outline:
    backgroundColor: "transparent"
    textColor: "{colors.silkscreen-ink}"
    rounded: "{rounded.key}"
    padding: "0 1.2rem"
    height: "48px"
  key-outline-hover:
    backgroundColor: "{colors.wash}"
  input-window:
    backgroundColor: "{colors.field}"
    textColor: "{colors.silkscreen-ink}"
    rounded: "{rounded.key}"
    padding: "0 1rem 0 3rem"
    height: "58px"
  chord-plate:
    backgroundColor: "{colors.aluminium-highlight}"
    textColor: "{colors.silkscreen-ink}"
    typography: "{typography.chord-rung-3}"
    rounded: "{rounded.key}"
    padding: "0.85rem 0.9rem 0.95rem"
  chord-chip-major:
    backgroundColor: "{colors.major-chip}"
    textColor: "{colors.silkscreen-ink}"
    typography: "{typography.chord-rung-4}"
    rounded: "{rounded.control}"
    padding: "0.5rem 0.7rem"
  chord-chip-minor:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.minor-chip-ink}"
    typography: "{typography.chord-rung-4}"
    rounded: "{rounded.control}"
    padding: "0.5rem 0.7rem"
  state-button:
    backgroundColor: "transparent"
    textColor: "{colors.ink-tertiary}"
    rounded: "{rounded.control}"
    padding: "0 0.55rem"
    height: "40px"
  state-button-on:
    textColor: "{colors.silkscreen-ink}"
  counter:
    backgroundColor: "{colors.key-face}"
    textColor: "{colors.key-ink}"
    rounded: "{rounded.control}"
    padding: "0.5rem 0.75rem"
  instruction-plate:
    backgroundColor: "{colors.silkscreen-ink}"
    textColor: "{colors.plate-ink}"
    rounded: "{rounded.plate}"
    padding: "clamp(28px, 4.4vw, 52px)"
---

# Design System: LiveChord

## Overview

**Creative North Star: "The Practice Amp Faceplate"**

LiveChord is built as the front of a 1960s-70s combo practice amp. A brushed aluminium control panel sits in a black tolex cabinet with cream piping; below it, sparkle grille cloth carries a black engraved instruction plate. Every control is a real amp part translated into a familiar web control: black push keys with a 3px skirt, a recessed input window, jewel-red lamps, a bat-handle toggle for the mic, printed 0-10 scales with red pointers for meters. Legends, headings and chord names are printed in a slab serif, the way a manufacturer silkscreens a faceplate; reading copy and measured numbers sit in a plain sans.

The system is light only (`color-scheme: light`, theme colour set to the tolex). Density is that of an instrument panel: controls are grouped under ruled legends, rules are 1.5px ink, and nothing floats in rounded cards. Materials are drawn, not photographed: brushed metal, tolex grain, grille weave, sparkle and mottle are inline SVG data-URI textures on `:root`, so no bitmap ships for surface.

Motion is mechanical and damped. Keys depress 2px, lamps warm on over about 180ms, the mic lever snaps in 170ms, meter pointers ease over 320ms, and only lamps that mean "working" blink. The picked song is one object that morphs through three named states (Pick a song, See its chords, Play along) with the View Transitions API. Reduced motion makes every change instant.

**Key Characteristics:**
- Brushed aluminium panel inside a tolex cabinet with a 3px piping border and 12px corner.
- Silkscreen legends: slab capitals, 600 weight, 0.12em tracked.
- Two faces with fixed jobs: Zilla Slab prints (headings, legends, key legends, chord names on a strict five-rung ladder); Source Sans 3 reads (copy, the input, every measured number).
- Jewel red is the only indicator colour; match green exists for one state.
- Black push keys with a visible skirt; latched keys sit 2px lower.
- Textures are SVG, never raster.

## Colors

A cool, near-neutral metal-and-ink palette with exactly two chromatic voices, both of them light sources.

### Primary
- **Jewel Red** (jewel-red): the indicator colour. Lit lamps and preset LEDs, the power jewel, the timeline playhead and its active-block cap, scale pointers, the search caret, and the border of the fault strip. It is always a light or a pointer, never a fill for a surface or a button. Its lit lens glows through Jewel Hot (jewel-hot); its dark lens is Jewel Off (jewel-off).

### Secondary
- **Match Green** (match-green): lit only in the matched state, when the chord you play equals the target. It fills the large match lamp (highlight Match Hot, match-hot) and outlines the You and Target boxes with a soft green glow. When unlit, the match lamp is smoked grey glass, so green is invisible until earned.

### Neutral
- **Brushed Aluminium** (aluminium): the control panel, under the brushed-metal texture with an overlay blend and a top-lit gradient. **Aluminium Highlight** (aluminium-highlight) tops chord plates and the skip link; **Aluminium Shadow** (aluminium-shadow) is the timeline well and empty cover-art ground.
- **Tolex Black** (tolex): the page and cabinet, under the tolex grain texture. It also sets `theme-color`.
- **Cream Piping** (piping): the 3px border around the panel front, and the focus ring colour on tolex and on the plate.
- **Grille Cloth** (grille-cloth): the field below the panel, layered with sparkle, mottle and weave textures.
- **Silkscreen Ink** (silkscreen-ink): all primary text on aluminium, 1.5px section rules, outline-key borders, chord diagrams, share bars, meter fills. It is also the engraved instruction plate's ground.
- **Ink Secondary / Tertiary** (ink-secondary, ink-tertiary): lede and meta text; idle state labels, group notes, empty chord readouts.
- **Rule / Rule Soft / Wash** (rule, rule-soft, wash): ink at 32%, 14% and 6%, used for hairline dividers, bar tracks and hover washes on aluminium.
- **Key Face / Top / Skirt** (key-face, key-top, key-skirt) with **Key Ink** (key-ink) and **Key Ink Muted** (key-ink-muted): the black push keys, their top-lit gradient and 3px skirt, and the text printed on them. The time counter uses the same key face as a recessed readout.
- **Input Window** (field) with **Field Hint** (field-hint): the search field and its autocomplete list.
- **Major Chip** (major-chip) and **Minor Chip Ink** (minor-chip-ink): major chords print light like the panel; minor chords print on the key face like ink.
- **Plate Ink / Plate Ink Muted** (plate-ink, plate-ink-muted): headings and body copy on the instruction plate.

### Named Rules
**The Palette Law.** Match green lights only when the played chord matches the target. It appears nowhere else: not in success toasts, not in lamps, not in links, not in the favicon.

**The Only Indicator Rule.** Every lamp, LED and pointer is jewel red. A new status shows as a red lamp that is lit, dark or blinking, never as a new hue.

**The Major Light, Minor Dark Rule.** Wherever chords are listed (progression steps, timeline blocks, quality tags on plates), major chords print light (major-chip) and minor chords print on the key face (key-face, minor-chip-ink).

## Typography

**Display Font:** Zilla Slab (Google Fonts, weights 500, 600, 700; token `--face-display`), falling back to Rockwell, Roboto Slab, Georgia, serif.
**Body Font:** Source Sans 3 (Google Fonts, weights 400-800; token `--face`), falling back to Helvetica Neue, Helvetica, Arial, sans-serif.

**Character:** A sturdy slab for everything the maker would print on the faceplate, paired with a quiet humanist sans for everything the player reads or measures. The slab carries hierarchy through size, weight (600 for legends, 700 for headings and chord names) and tracking; there is no width axis and no `font-stretch`. The slab tops out at 700, so nothing is set heavier.

### Hierarchy
- **Display** (slab 700, clamp 2.1-3.2rem, line-height 1.04, -0.012em, max-width 18em): the one H1, set as three sentences that each hold together on a line.
- **Headline** (slab 700, clamp 1.75-2.75rem, 1.04, -0.015em, balanced wrap): the song title on the chords panel.
- **Title** (slab 700, clamp 1.625-2.25rem, 1.05, -0.02em): the instruction plate heading. The wordmark is slab 700 at clamp 1.25-1.625rem, -0.01em; the player strip title is slab 700 at 1.125rem, 1.15, -0.01em.
- **Preset title** (slab 700, 1.2rem, 1.15; 1rem on phones): preset key titles, over the artist in the sans (500, 0.875rem, 1.3).
- **Body** (sans 400, 1rem, 1.5): running copy. The lede steps up to clamp 1.0625-1.1875rem at 1.55 with a 60ch measure; group notes are 0.9375rem in ink-tertiary; plate paragraphs are 0.9375rem at 1.6, 44ch; the search input is sans 500 at 1.0625rem.
- **Label / Legend** (slab 600, 0.8125rem, line-height 1.2, 0.12em, uppercase): silkscreen legends over every control group and readout; the small legend is 0.6875rem. Printed tags, the Load pick, how-it-works step headings, spec terms and tolex nav links are slab 700 at 0.6875-0.8125rem, 0.12-0.14em uppercase.
- **Key legend** (slab 600, 0.9375rem, 0.1em, uppercase): push keys. Outline keys are slab 700 at 0.8125rem, 0.12em; toggle keys 0.8125rem; state strip buttons slab 700 at 0.6875rem, 0.12em; the bat-switch state word slab 700 at 0.9375rem, 0.1em.
- **Numerals** (sans 700, tabular): the counter (0.9375rem, 0.04em), gauge values (1.25rem), elapsed time, share values (0.75rem), scale numbers and tick labels (0.6875rem). Numbers that are part of a printed legend (preset numbers, state and step numerals) take the slab with their legend.

### The Chord Ladder
Chord names are slab 700, -0.01em, line-height 0.95, lining numerals, never wrapping. Sharps and flats are typeset as raised accidentals at 0.6em, lifted 0.5em. An empty readout drops to 500 in ink-tertiary. Five rungs, never mixed:
1. **Rung 1** (clamp 4-6rem): Now.
2. **Rung 2** (clamp 2.25-3rem): Next, You're playing, Song chord now.
3. **Rung 3** (1.875rem): chord plates.
4. **Rung 4** (1.25rem, line-height 1): progression steps and drill keys (drill keys at -0.02em).
5. **Rung 5** (0.8125rem): timeline blocks, dropping to 0.5625rem with -0.02em tracking on narrow blocks, and to hidden text on blocks too narrow for even that.

### Named Rules
**The Slab Prints, Sans Reads Rule.** Zilla Slab sets what the maker silkscreens: headings, legends, key legends, tags and chord names. Source Sans 3 sets what the player reads or measures: copy, the input, artists, and every counter, gauge, time and scale number. There is no third face.

**The Ladder Rule.** A chord name takes the rung of its role. A new chord readout joins an existing rung; it does not get a new size.

## Layout

The page is a cabinet: a tolex band (52px, nav right-aligned), then the panel front inset by `side`, then a centred footer. Inside the panel, content is capped at 1180px and padded by `gutter`. The panel head is a three-column grid (wordmark, state strip, power lamp) ruled off with 1.5px ink; under 860px the state strip drops to its own row, and under 560px only the active state keeps its text label.

Sections are groups under a legend that runs into a half-opacity 1.5px rule, with an optional note at the right. Rhythm is set with clamped values (group spacing clamp 28-44px; panel head margin clamp 20-34px, 28-60px on the search stage). Key banks use a 12px gap, toggles 10px.

Responsive behaviour reflows rather than shrinks: the five-key preset bank goes 5, 3, then 1 column (at 720px each preset becomes a 60px row with lamp, title over artist, and number). The play deck changes from a single row (play, Now, Next, counter) to a 2x2 grid; the practice row moves the match lamp above the You and Target boxes. The chord grid auto-fits 160px (140px on phones) columns capped by the song's chord count.

## Elevation & Depth

Depth is physical, not atmospheric. The panel is lit from above (a white-to-clear gradient and a white top edge, a darker bottom edge); the cabinet front throws the one large shadow onto the tolex. Keys stand proud on a 3px skirt; wells (input window, timeline, practice boxes, warm-up strip, counter) are recessed with inset shadows. Chord plates are the only lifted cards, and they lift on hover.

### Shadow Vocabulary
- **Cabinet** (`box-shadow: 0 22px 50px -12px rgba(0,0,0,0.65), 0 2px 4px rgba(0,0,0,0.4)`): the panel front on the tolex. Used once.
- **Key rest** (`box-shadow: inset 0 1px 1px rgba(255,255,255,0.14), 0 3px 8px rgba(8,10,12,0.28)`): push keys. Pressed: `0 1px 3px` with a 2px drop; latched: `inset 0 2px 4px rgba(0,0,0,0.55)`.
- **Well** (`box-shadow: inset 0 2px 5px rgba(20,23,28,0.16-0.25)`): input window and timeline track; practice boxes and warm-up at 0.1.
- **Plate** (`box-shadow: inset 0 1px 1px rgba(255,255,255,0.9), 0 2px 6px rgba(20,23,28,0.14)`): chord plates; hover raises to `0 10px 20px rgba(20,23,28,0.18)`.
- **Lamp glow** (`box-shadow: 0 0 12px 3px rgba(255,80,58,0.5)`): lit lenses only. The match lamp glows `0 0 22px 6px rgba(47,179,106,0.5)`.
- **Dropdown** (`box-shadow: 0 14px 30px rgba(20,23,28,0.25)`): the autocomplete list.

### Named Rules
**The Only Lit Glows Rule.** Glow belongs to a lit lens. Nothing else emits light.

**The Press Travels Down Rule.** Pressing moves a control down (keys 2px, outline keys 1px, chord plates 1px). Nothing scales up on press.

## Shapes

Corners are small and machined: 2px on the engraved plate, 3px on tags, quality chips and cover art, 4px on small controls, 5px on keys, the input window, chord plates and the timeline, 6px on the practice wells and bat switch, 12px only on the cabinet front. Lamps, jewels, the play key, the match lamp and state numerals are round. Borders are ink at 1.5px, or a 1px hairline. The instruction plate carries a second printed border inset 10px, like a rear-panel legend plate. Pointers and the playhead cap are clipped triangles.

## Components

### Push Keys
Black, heavy and mechanical.
- **Shape:** 5px corners, 58px tall (60px large, 52px in toggles, 48px drill keys); a 3px skirt drawn into the background gradient under the key face.
- **Default:** key-face gradient from key-top, key-ink legend in tracked slab capitals.
- **Hover / Press:** the face lightens one step; on press the key drops 2px and its shadow tightens (70ms). Focus is a 2px outline offset 3px.
- **Latched (toggles, drill keys):** a darker face, 2px lower, inset shadow, and its lamp lit.
- **Disabled:** a grey face (#4E535B) with no shadow; disabled presets fade to 55% while their loading LED still blinks.
- **Outline key:** transparent with a 1.5px ink border and ink legend, 48px (44px small); hover fills with wash, press drops 1px. Used for secondary moves such as back to chords.
- **Round key:** the play/pause key, 76px circle (64px on phones).

### Presets
Numbered push keys, 116px tall, with an LED top left, a number top right, then a slab title over the artist in the sans, both in mixed case. The LED blinks while the preset loads and stays lit once loaded.

### Input Window
A 58px recessed field (field ground, 1.5px ink border, inset well shadow) with a search icon at left and a jewel-red caret. Focus is a 2px ink outline offset 3px. The autocomplete list below is the same material, and its active row inverts to ink.

### Lamps and Jewels
- **Lamp** (10px, 8-9px in keys): a dark red lens that lights through an opacity fade (180ms). Steady when it means state; a 0.9s blink when it means working.
- **Power jewel** (30px): a two-tone bezel around a lens; lit steady when ready, a stronger glow while listening, a 1.4s blink while connecting and 0.7s when down.
- **Match lamp** (78px, 58px on phones): smoked glass unlit, match green lit, fading on in 120ms.

### State Strip
Three numbered states joined by short rules, each a 40px button with a lamp, a circled numeral and a legend. Idle states are ink-tertiary; the current state is ink with its lamp lit.

### Chord Plates
Aluminium-highlight plates with a 1.5px border at 50% ink: chord name on rung 3, a boxed quality tag (inverted for minor), a guitar diagram drawn in ink, the notes, and a share bar. Hover lifts 2px and darkens the border; touch devices do not lift.

### Timeline
A 68px recessed track (56px on phones) of blocks sized to duration, major light and minor dark, divided by 2px aluminium seams; the last block has no seam. The active block wears a 5px jewel-red cap; the playhead is a 3px red line with a triangular cap. Below, a printed tick scale labels every five seconds.

### Bat Switch
The mic control: a round two-tone bezel with a lever that flips between STANDBY and LISTEN (scaleY, 170ms), beside a slab state word and a small sans action line, framed by a 1.5px ink border. It is a switch (`aria-checked`), not a knob.

### Scales
Printed 0-10 scales drawn with gradients (minor ticks every tenth, major at 0, 5 and 10) with a red triangular pointer that eases 320ms and hides while idle. The level meter is a ten-segment bar filled in ink.

### Instruction Plate
A black engraved plate seated on the grille: 2px corners, an inset printed border, the plate title, three numbered steps and a spec table of legend terms and muted descriptions.

### Song Morph
The song object (result row, song card, player strip) shares one `view-transition-name` and morphs over 340ms with the ease-out curve. The old page holds still and opaque while the new page fades in over it in 200ms, so the view never dips toward the tolex.

## Do's and Don'ts

### Do:
- **Do** keep match green (match-green) to the matched state only: the match lamp and the You and Target outlines.
- **Do** show every new status as a jewel-red lamp that is lit, dark or blinking, and blink only for work in progress.
- **Do** set chord names from the five-rung ladder, with raised accidentals and major-light, minor-dark printing.
- **Do** label control groups with a silkscreen legend (slab 600, 0.8125rem, 0.12em, uppercase) running into a ruled line.
- **Do** set new headings, legends, key legends and chord names in the slab (`--face-display`), and new copy and measured numbers in the sans (`--face`, tabular figures for numbers).
- **Do** make new controls from existing parts: push key, outline key, input window, bat switch, printed scale.
- **Do** draw new textures and cover fallbacks as SVG in the panel palette, as `sleeve.svg` and the `:root` textures are.
- **Do** make presses travel down and lamps fade on (120-220ms, `cubic-bezier(0.16, 1, 0.3, 1)`), and make every change instant under reduced motion.

### Don't:
- **Don't** use green anywhere outside the matched state, or add a second indicator hue.
- **Don't** introduce a third typeface, a chord-name size off the ladder, a slab weight above 700, or `font-stretch`.
- **Don't** add a dark theme by inverting tokens; the page is light only, and a black-panel variant is not part of this system.
- **Don't** use knobs or sliders as inputs; controls stay familiar (field, keys, toggles, switch).
- **Don't** give glow to anything but a lit lens, or scale controls up on press.
- **Don't** ship raster surface textures.
