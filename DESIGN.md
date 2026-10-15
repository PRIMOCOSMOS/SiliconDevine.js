---
name: SiliconDevine
description: An immersive keynote stage for real numerical crystal models.
colors:
  bg: "#06080d"
  surface: "#10151e"
  field: "#19202c"
  line: "#2c3647"
  muted: "#abb7ca"
  text: "#edf2fa"
  accent: "#aecfff"
  primary-fill: "#dce9ff"
  primary-ink: "#162638"
  native-field: "#161e2b"
  native-line: "#293346"
  native-muted: "#a7b4c9"
  native-ink: "#142642"
typography:
  brand:
    fontFamily: "Tektur Variable, sans-serif"
    fontSize: "22px"
    fontWeight: 500
    lineHeight: 1.1
    letterSpacing: "-.025em"
  title:
    fontFamily: "Tektur Variable, ZCOOL QingKe HuangYou, sans-serif"
    fontSize: "clamp(26px, 3vw, 48px)"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "-.035em"
  display:
    fontFamily: "Tektur Variable, ZCOOL QingKe HuangYou, sans-serif"
    fontWeight: 400
  section:
    fontSize: "18px"
    fontWeight: 500
  body:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, sans-serif"
    fontSize: "12px"
    lineHeight: 1.8
  label:
    fontSize: "11px"
  code:
    fontFamily: "Consolas, monospace"
    fontSize: "11px"
    lineHeight: 1.9
rounded:
  control: "10px"
  input: "9px"
  panel: "16px"
  pill: "24px"
spacing:
  tight: "8px"
  compact: "12px"
  group: "20px"
  mobile-gutter: "18px"
  desktop-gutter: "40px"
components:
  button-primary:
    backgroundColor: "{colors.primary-fill}"
    textColor: "{colors.primary-ink}"
    rounded: "{rounded.control}"
    padding: "11px 15px"
  button-secondary:
    backgroundColor: "{colors.field}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "11px 15px"
  button-header:
    backgroundColor: "transparent"
    textColor: "#b5c3d9"
    rounded: "{rounded.pill}"
    padding: "10px 13px"
  input:
    backgroundColor: "#090f18"
    textColor: "#e5edfa"
    rounded: "{rounded.input}"
    padding: "11px 12px"
  navigation-item:
    backgroundColor: "transparent"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    padding: "10px"
  navigation-item-current:
    backgroundColor: "#304360"
    textColor: "#fff"
    rounded: "{rounded.control}"
    padding: "10px"
  inspector:
    backgroundColor: "#101620"
    textColor: "{colors.text}"
    padding: "25px 28px 40px"
    width: "380px"
---

# Design System: SiliconDevine

## Overview

**Creative North Star: "The Crystal Keynote Stage"**

The real model is the protagonist. An uninterrupted near-black stage, silver-white text and ice-blue controls bring keynote clarity to scientific crystal rendering. Controls gather into a compact suspended transport; source selection and detailed interpretation appear only when requested.

The separate model library uses editorial scale and restrained architecture diagrams. The Windows launcher pairs an orbit-and-crystal brand stage with a configuration area. Appearance motion gives these surfaces presence while remaining separate from numerical execution. The launcher orbit is identity artwork, never a representation of captured tensor values.

**Key Characteristics:**
- A full-window model stage with a compact bottom transport.
- A separate categorized model library and contextual inspector.
- Silver-white hierarchy, ice-blue actions and dark tonal surfaces.
- Genuine crystal geometry and numerical colors retain scientific meaning.
- Responsive, DPI-aware controls and deliberate appearance motion.

Recorded from the current `demo/style.css`, `demo/appearance.ts`, `demo/index.html`, `launcher_ui.py`, and workbench, library, inspector, mobile and launcher captures. This replaces the rejected persistent-sidebar layout. Token values describe the rebuilt interface; native variants are explicitly named rather than treated as drift.

## Colors

The shell combines black, cool graphite, silver-white and pale ice blue. The workbench and native launcher share the palette strategy while retaining the platform-specific values recorded above.

### Primary
- **Ice Blue** (`accent`): web focus, input accents and optical emphasis.
- **Frosted Action** (`primary-fill`, `primary-ink`): connect and rebuild actions. The launcher uses the same fill with `native-ink`; playback and selected header controls use closely related pale fills.

### Neutral
- **Black Stage** (`bg`): root canvas, model stage and native brand stage.
- **Deep Surface** (`surface`): native status surface.
- **Control Surface** (`field`, `native-field`): secondary actions and native fields.
- **Blue Graphite Line** (`line`, `native-line`): the shared boundary vocabulary; individual browser fields and sections also have local stroke values.
- **Silver White** (`text`): primary working text.
- **Cool Secondary Text** (`muted`, `native-muted`): supporting information.

The library's small family diagrams add subdued violet, mint and warm sand to distinguish categories. These categorical illustrations do not redefine the numerical renderer's palette.

**The Numerical Color Rule.** Shell and category accents must not replace the model's numerical color mapping. Negative, zero, positive, active, unknown and masked values retain the scientific renderer's existing semantics and legend.

## Typography

Tektur supplies the Latin identity and family subtitles. Model titles, library statements and family headings pair Tektur with bundled ZCOOL QingKe HuangYou for Chinese glyphs, using regular weight. The launcher loads bundled Tektur for its wordmark and ZCOOL QingKe HuangYou for Chinese display headings. Body text and controls keep the Windows UI text stack with Chinese fallback. Consolas marks code and endpoint values; mathematical content retains equation typography.

The working hierarchy rises from compact labels to model titles. The desktop model title is fluid, becomes 30px below 1100px and 27px below 760px. The library uses a much larger editorial statement, with a 48–84px fluid size and 58px mobile size. Supporting prose uses 1.8–1.9 line height, and the library introduction is bounded to 38ch.

Native fonts use point sizes and DPI scaling; they are not interchangeable with browser pixel values. Chinese launcher display headings use 40pt on the identity stage and 31pt above configuration. The bundled font licenses remain with the application in THIRD_PARTY_LICENSES.

**The Model Title Rule.** On the workbench, the active model name leads the text hierarchy; transport, metadata and hierarchy controls remain smaller and quieter.

## Layout

The default workbench has no persistent source rail. A compact global header sits over a full-window workspace organized as title/navigation, model stage and bottom transport. The desktop header is 82px high; the workspace fills the remaining dynamic viewport, with a 660px minimum height. The centered transport is bounded to 790px.

Opening the library replaces the workspace with a full-page selection surface. Its desktop layout pairs a large introduction with expandable model families, followed by source/import and IDE connection areas. Returning or choosing a model restores the stage.

The inspector is requested from global navigation. On desktop it occupies the right edge, and the workspace reserves room for it instead of placing it over the model. It is 380px wide, reducing to 340px at 1100px. It contains explanation, graph index, parameters, tensor windows, numerical legend and rendering details.

At 760px and below, the global header wraps, title/navigation stack, transport controls wrap and the library becomes a vertical flow. An opened inspector sits below the stage and transport in document flow. Presentation mode simplifies the header, title and transport while keeping navigation and playback available.

The native launcher places orbit/crystal identity on the left and configuration on the right. Below a DPI-scaled 1000px window width, the identity stage hides so configuration remains usable. Only configuration scrolls; status and the full-width primary launch action remain in the footer.

**The Receding Controls Rule.** Keep the main stage clear by placing model selection on its own surface and detailed interpretation in the requested inspector.

## Elevation & Depth

The model supplies meaningful spatial depth. The transport floats on a subtle dark translucent gradient and a soft shadow (`0 18px 50px -25px #000`). A faint horizontal optical line anchors the stage. The library has a restrained radial light behind its editorial introduction; the inspector uses a dark tonal surface.

**The Stage Depth Rule.** Concentrate spatial expression in the model and purposeful optical framing; preserve quiet working surfaces around it.

## Shapes

Standard browser actions have softly rounded corners; fields are slightly tighter. Header actions and playback use pill forms, while the dock and inspector use broader panel corners. The inspector rounds its upper-left corner on desktop and both upper corners on mobile. Family rows use separators and outlined diagrams rather than repeated cards.

Icons are inline stroke SVGs with consistent thin geometry. The native launcher uses a locally bundled, antialiased orbit/crystal body with three small moving light markers. It chooses a 400px, 600px or 800px image for the display scale, with drawn geometry as a missing-asset fallback. Native buttons use image-backed surfaces with opaque matte corners and distinct focus artwork while keeping native text and input behavior.

## Components

### Buttons

Pale primary actions use dark, medium-to-bold text. Secondary actions use graphite fills, brighten on hover and deepen on press. Header actions combine stroke icons and labels, with a pale fill for the expanded state. Playback remains the highest-emphasis action in the transport. Disabled actions retain disabled behavior and lower emphasis.

Keyboard focus uses a two-pixel ice-blue outline with a four-pixel offset; the stage uses its own subtle inset outline. Native fields change border color on focus; native button surfaces include explicit idle, active, pressed, disabled and focus states.

### Inputs / Fields

Fields pair a dark fill with a fine boundary and a visible label. Endpoint and coordinate values use monospace. Selects retain native option behavior. Native configuration locks during capture, and keyboard focus scrolls hidden configuration fields into view.

### Model Families

Each library family is an expandable row with an architecture diagram, title, small family subtitle and rotating expand icon. Expanded content introduces the family and presents compact model buttons. These are selection controls, not numerical data views.

### Navigation and Inspector

Global controls switch library, inspector and presentation states. Module index entries wrap long names, use a subdued hover fill and a stronger current-item fill. The inspector close control returns focus to its trigger. Escape closes the library and inspector and exits presentation mode.

Provenance, tensor windows, color mapping and rendering information use disclosures. Their text remains compact and readable; they do not occupy the default stage.

### Suspended Transport

Playback, overview, zoom, label visibility, speed and fullscreen share the upper row. Calculation progress and current local computation sit beneath. The dock is centered and width-bounded, and its content wraps on mobile. Its background gradient and soft shadow distinguish it from the stage without recreating a dashboard shell.

### Crystal Stage and Appearance Motion

Model replacement uses an 800ms opacity/brightness/scale reveal; library opening uses a 620ms clip reveal. Both use `cubic-bezier(.16,1,.3,1)`. Control colors transition in 200ms, family expand icons rotate in 400ms, and loading status pulses only while loading.

The workbench removes shell animation under `prefers-reduced-motion` and initially pauses model playback. Hidden-page CSS animations pause. The native projected light trail checks Windows animation preferences and stops its timer when hidden; the high-resolution optical body stays static. The native capture line conveys indeterminate activity, not a measured completion percentage.

## Do's and Don'ts

### Do:
- **Do** let the actual crystal model dominate the default workspace.
- **Do** put model selection in the library and interpretation in the requested inspector.
- **Do** preserve numerical color meaning and source provenance.
- **Do** use silver-white hierarchy, ice-blue actions and quiet graphite controls.
- **Do** pair Tektur Latin identity with ZCOOL QingKe HuangYou Chinese display headings, keeping controls in the UI text face.
- **Do** preserve responsive flow, keyboard access and native DPI-aware sizing.
- **Do** distinguish launcher identity artwork from scientific model data.
- **Do** keep appearance motion separate from computation and respect the implemented motion preferences.

### Don't:
- **Don't** restore the rejected persistent-sidebar dashboard layout.
- **Don't** add ornamental HUD clutter around the scientific structure.
- **Don't** recolor numerical tensors to match category or shell accents.
- **Don't** imply a measured capture percentage with an indeterminate indicator.


## Large model extension · v0.6

Structure mode preserves the established vertical crystal stage. Repeated stacks, attention rings, expert fans and recurrence loops identify functional templates. Etched thin matrices distinguish parameters from data cubes without changing numeric tint or weight-line linkage. Structure-only semantic color is explicitly separated from numerical color. At most one active scope is built; hover locks local flows and click enters a referenced template. Logical partition labels and a persistent mode badge prevent schematic animation being read as measured inference.


## Optical refinement · v0.6.1

The library uses a silver split torus around a faceted optical core. This artwork is branding, separate from scientific tensors. Room reflections and restrained cold light supply material depth without postprocessing bloom. The workbench reveals the same dark optical field through an alpha renderer; world-space labels have transparent backgrounds.

The library artwork renders at 20 fps with DPR capped at 1.5, initializes only when visible, and stops offscreen or when the document is hidden. Full, quiet and off settings persist locally. Reduced-motion overrides full. Native artwork shares the exported scene at 400/600/800 pixels; only a short projected light trail updates, with cached image geometry. Native system animation settings and the visible motion toggle control its timer.

To regenerate native artwork, serve the project on port 5198, run tests/render-keynote-assets.mjs, then tests/build-optical-assets.py. Both desktop and mobile layouts are checked alongside native 100% and 150% scaling.


## Numerical mechanism bridge · v0.7

The large-model structure remains a bounded source-configured overview. Composite modules expose a miniature dependency skeleton of their actual template; independent instances retain explicit repeat counts. Internal fragments carry functional colors and bounded moving connections. They denote structure, never invented activation values.

A computation view is a peer of its structural scope, reached by one toggle or a terminal operator click. It uses fully evaluated small PyTorch graphs with persistent data-origin labeling. Long graphs have named computation stages; complete graphs remain available. Stages preserve real boundary tensor values and source dependencies. Only one view is allocated.

Explicit focus uses a 720ms quartic ease-out camera flight. Orbit interaction or wheel zoom cancels it. Hover changes computation without moving the camera. Optional 12-second stage tours pause over the model and stop on direct camera input. Reduced-motion uses immediate camera changes. RoPE pair arcs, routing candidate bars, weighted crystal contributions, normalization and attention streams explain the active mathematics.


## Side annotations and repeated templates · v0.7.1

Composite geometry contains only its dependency skeleton. Its explanation lives in one nearby world-space typographic plate. Parallel branches share a row, sequential stages retain reading order; repetition combines a compact sequence with three translucent receding sections and an exact independent-instance count. Ghost sections represent repeated templates, not sampled numeric tensors or tied weights.

Both side lanes are available for repeated blocks. Annotation scaling targets readable screen size, with collision checks against tensor and composite volumes. Hover shows only the relevant region; all/none controls remain. Provenance and active-region text share normal document flow under the title, with no absolute offsets. Navigation wraps on narrow or inspector-constrained widths.

## Industrial precision in the model stage (2026-10-08)

The reference imagery contributes an abstract visual grammar: silver registration marks, segmented circular boundaries, quiet graphite surfaces and a restrained yellow selection signal. No scenery, fictional machines, logos or decorative data are imported.

`src/render/industrialTheme.ts` owns architectural colors. Numerical colors stay in the existing numerical palette. Tensor plates distinguish weights from activations; ownership frames use open corners and three registration ticks. Independent branch towers use broken orbital arcs around actual intermediate activation tensors. All tower rails and ticks are batched into one draw with a shared animation clock. Motion emphasizes the active module only.

Captions remain fixed in model coordinates beside their tensors. Their size is chosen once for the scope, with automatic visibility filtering; rotating the camera never repacks their anchors. Architecture rules use executed dependencies and source scopes, without model-name or encoder/decoder-field switches. Small models with captured composition open in architecture view as well.


### Machined metal supports (2026-10-09)

Metal is expressed through opaque geometry and reflected light. Each tensor rests inside a bevelled silver rim with a darker lower flange. Parameter rims use warm alloy; activation rims use cool aluminium. The middle remains open and the numerical crystals remain transparent. A shared prefiltered studio reflection map is created once per viewer; the model stage does not display a room or image background. Subtle roughness grain and directional lighting reveal the bevel as the camera rotates.

All rims in one scope share one instanced draw, one rounded geometry and one physical material. Selected ownership changes instance colors without making metal transparent. Scene disposal releases the studio target; scope changes release rim geometry and roughness textures. No global tone mapping is added, preserving the existing unlit value colors.

### Fixed world-space typography (2026-10-09)

Annotations are transparent text geometry beside each tensor, with no backing crystal, metal slab or leader line. A small registration accent aligns with the near edge. Both position and orientation are fixed in model coordinates; camera movement never repacks or billboards the text. Visibility filtering remains separate from anchoring. Reflective metal belongs to tensor supports, while annotation glyphs remain unlit for contrast.

### Mathematical structure for architecture designers (2026-10-13)

Architecture and mathematical structure are the primary views. Mathematics stops at meaningful functions, vectors and matrices: projection, dot product, channel gating, attention, normalization and residual connection. Concrete Python helper invocations form source-backed units. Masks and coordinate preparation belong to their functions. A multiply is only a dot product when its captured vector-axis reduction proves that interpretation.

Clicking a mathematical unit focuses it and shows its explanation. The separate raw-call view exposes implementation steps for debugging. Aggregation preserves captured tensors, external consumers and parameter indices; reorganizing a projected vector retains its W-line correspondence and bias. Motion labels use the visible mathematical function rather than renderer implementation names.

Engineering preparation is absent as a standalone layer in default views. Its captured children and coordinate provenance belong to the mathematical consumer, retaining that consumer's own visual operator and parameters. Data validity and causal visibility remain functional constraints; routing, feature concatenation and Gaussian distributions retain design meaning. Scope boundaries may expose captured effective inputs without inventing an operator or replacing their values.

The docked Model Inputs section presents the Python calling contract independently of the selected scope. Input labels use bound argument names and pytree paths, including aliased tensors, keyword arguments and nested containers. Concrete example dimensions, source symbols, optional defaults and unknown dimensions remain distinct. Input explanation never overlays the 3D model.

### Guided import recovery (2026-10-14)

Automatic parsing owns the import workflow. It reads module relationships, unique construction examples, annotated configuration defaults, command-line defaults and example forward inputs before considering a form. Generated examples pass the real PyTorch forward. Conflicting source evidence remains explicit ambiguity.

Blocked imports offer an inline recovery action, without automatically opening a modal. The form asks only unresolved constructor fields or the failed input; it preserves resolved settings. Confirming an ambiguous entry resumes automatic parsing. The quiet Import Diagnostics action lives within the IDE connection section. A usable structural view with unknown dimensions never triggers a supplementary form or request banner. Failed recaptures preserve the last graph. Technical details stay secondary.

The native launcher shows dependency names, its selected interpreter, environment switching and copyable installation instructions. Installation remains explicit in the user's terminal. Recovery controls appear before configuration fields; status and primary actions remain visible below. Per-project forms persist in application data and update the current local server. Writes require same-origin requests and a per-session token; the read-only SDK bridge stays read-only.

Finish review: desktop and 430px forms, inline input errors, keyboard controls, and native 100%/150% layouts checked. The validation message stays beside the sticky submit action. Existing visual identity and model geometry are unchanged.
